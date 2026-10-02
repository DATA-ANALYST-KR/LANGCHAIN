import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from langsmith import traceable
import json

MAX_ITERATIONS = 10
MODEL = "gemini-3.5-flash-lite" #"gemini-3.6-flash"


# ============================================================
# TOOLS: REGULAR PYTHON FUNCTIONS
# ============================================================


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(
        f"    >> Executing "
        f"get_product_price(product='{product}')"
    )

    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }

    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(
    price: float,
    discount_tier: str,
) -> float:
    """Apply a discount tier to a price and return the final price.

    Available tiers: bronze, silver, gold.
    """

    print(
        f"    >> Executing apply_discount("
        f"price={price}, "
        f"discount_tier='{discount_tier}')"
    )

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    discount = discount_percentages.get(
        discount_tier,
        0,
    )

    return round(
        price * (1 - discount / 100),
        2,
    )


# ============================================================
# MANUAL FUNCTION SCHEMAS
# ============================================================
#
# Without LangChain's @tool decorator, the function schemas must
# be manually defined.
#
# These schemas describe the Python functions to Gemini.
# They do not execute the functions.


tools_for_llm = [
    {
        "name": "get_product_price",
        "description": (
            "Look up the price of a product "
            "in the catalog."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": (
                        "The product name, e.g. "
                        "'laptop', 'headphones', or 'keyboard'"
                    ),
                }
            },
            "required": ["product"],
        },
    },
    {
        "name": "apply_discount",
        "description": (
            "Apply a discount tier to a price "
            "and return the final price. "
            "Available tiers: bronze, silver, gold."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The original price",
                },
                "discount_tier": {
                    "type": "string",
                    "description": (
                        "The discount tier: "
                        "'bronze', 'silver', or 'gold'"
                    ),
                },
            },
            "required": [
                "price",
                "discount_tier",
            ],
        },
    },
]


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================
#
# Gemini's generateContent API accepts the system instruction
# separately from the normal conversation contents.


SYSTEM_INSTRUCTION = (
    "You are a helpful shopping assistant. "
    "You have access to a product catalog tool "
    "and a discount tool.\n\n"
    "STRICT RULES: You must follow these exactly:\n"
    "1. NEVER guess or assume any product price. "
    "You MUST call get_product_price first "
    "to get the real price.\n"
    "2. Only call apply_discount AFTER you have received "
    "a price from get_product_price. Pass the exact price "
    "returned by get_product_price. Do NOT pass "
    "a made-up number.\n"
    "3. NEVER calculate discounts yourself using math. "
    "Always use the apply_discount tool.\n"
    "4. If the user does not specify a discount tier, "
    "ask them which tier to use. Do NOT assume one."
)


# ============================================================
# GEMINI CLIENT
# ============================================================


client = genai.Client()


# ============================================================
# LANGSMITH TRACE SERIALIZATION
# ============================================================
#
# Gemini returns provider-specific Content and Part objects.
# The functions below create readable copies for LangSmith.
#
# These functions affect only tracing.
# They do not modify the native objects sent to Gemini.


def serialize_gemini_part(part) -> str:
    """Convert a Gemini Part into readable text for LangSmith."""

    # Ordinary text response
    if getattr(part, "text", None):
        return part.text

    # Function call requested by Gemini
    function_call = getattr(
        part,
        "function_call",
        None,
    )

    if function_call is not None:
        function_name = function_call.name

        function_args = dict(
            function_call.args or {}
        )

        return (
            "FUNCTION CALL\n"
            f"Name: {function_name}\n"
            f"Arguments: "
            f"{json.dumps(function_args, indent=2)}"
        )

    # Result returned by the Python function
    function_response = getattr(
        part,
        "function_response",
        None,
    )

    if function_response is not None:
        function_name = function_response.name
        response_data = function_response.response

        return (
            "FUNCTION RESPONSE\n"
            f"Name: {function_name}\n"
            f"Response: "
            f"{json.dumps(response_data, indent=2)}"
        )

    # Fallback for any other Gemini Part type
    if hasattr(part, "model_dump"):
        serialized_part = part.model_dump(
            mode="json",
            exclude_none=True,
        )

        return json.dumps(
            serialized_part,
            indent=2,
        )

    return str(part)


def serialize_gemini_message(message) -> dict:
    """Convert Gemini Content into a readable LangSmith message."""

    parts = message.parts or []

    # Gemini calls model-generated messages "model".
    # LangSmith calls model-generated messages "assistant".
    if message.role == "model":
        trace_role = "assistant"
    else:
        trace_role = message.role

    content_parts = [
        serialize_gemini_part(part)
        for part in parts
    ]

    return {
        "role": trace_role,
        "content": "\n\n".join(content_parts),
    }

def process_gemini_trace_inputs(
    inputs: dict,
) -> dict:
    """Control how Gemini inputs appear in LangSmith."""

    # Add the system instruction as a visible system message
    # in the trace-only representation.

    trace_messages = [
        {
            "role": "system",
            "content": inputs["system_instruction"],
        }
    ]

    # Add the complete accumulated conversation history.

    trace_messages.extend(
        serialize_gemini_message(message)
        for message in inputs["messages"]
    )

    return {
        "model": inputs["model"],
        "messages": trace_messages,
        "tool_schemas": inputs["tools"],
    }


def process_gemini_trace_outputs(
    output,
) -> dict:
    """Control how Gemini outputs appear in LangSmith."""

    if (
        output is None
        or not getattr(output, "candidates", None)
    ):
        return {
            "messages": [],
            "error": "Gemini returned no candidates",
        }

    ai_message = output.candidates[0].content

    formatted_message = serialize_gemini_message(
        ai_message
    )

    return {
        "messages": [
            formatted_message
        ],
        "model_version": getattr(
            output,
            "model_version",
            None,
        ),
        "response_id": getattr(
            output,
            "response_id",
            None,
        ),
    }


# ============================================================
# TRACED GEMINI CALL
# ============================================================
#
# Without LangChain's chat-model wrapper, the Gemini call is
# traced manually.
#
# Passing all request components as explicit arguments allows
# LangSmith to display:
#
# - Model name
# - System instruction
# - Tool schemas
# - Complete accumulated message history
# - Gemini output


@traceable(
    name="Gemini Generate Content",
    run_type="llm",
    process_inputs=process_gemini_trace_inputs,
    process_outputs=process_gemini_trace_outputs,
)
def gemini_chat_traced(
    messages,
    system_instruction,
    tools,
    model,
):
    gemini_tools = types.Tool(
        function_declarations=tools
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[gemini_tools],
    )

    return client.models.generate_content(
        model=model,
        contents=messages,
        config=config,
    )


# ============================================================
# MANUAL AGENT LOOP
# ============================================================


@traceable(name="Gemini Agent Loop")
def run_agent(question: str):
    # Map function names returned by Gemini to the actual
    # executable Python functions.

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # Local conversation history.
    #
    # The list will accumulate:
    #
    # 1. Original user question
    # 2. Gemini function-call response
    # 3. Function result
    # 4. Additional Gemini calls and results

    messages = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=question
                )
            ],
        )
    ]

    for iteration in range(
        1,
        MAX_ITERATIONS + 1,
    ):
        print(f"\n--- Iteration {iteration} ---")

        # Send the complete accumulated message history.
        #
        # The static system instruction and tool schemas are also
        # supplied during every invocation.

        response = gemini_chat_traced(
            messages=messages,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=tools_for_llm,
            model=MODEL,
        )

        # Preserve Gemini's complete response Content object.
        #
        # This can include:
        #
        # - Text
        # - Function calls
        # - Function metadata
        # - Gemini thought-signature metadata

        ai_message = response.candidates[0].content

        # Extract function calls selected by Gemini.

        tool_calls = response.function_calls or []

        # If Gemini did not request a function, treat the response
        # as the final answer and stop the loop.

        if not tool_calls:
            print(
                f"\nFinal Answer: {response.text}"
            )

            return response.text

        # Execute only the first function call.
        #
        # This forces one function execution per iteration,
        # matching the instructor's workflow.

        tool_call = tool_calls[0]

        tool_name = tool_call.name

        tool_args = dict(
            tool_call.args or {}
        )

        print(
            f"  [Tool Selected] {tool_name} "
            f"with args: {tool_args}"
        )

        # Find the actual Python function.

        tool_to_use = tools_dict.get(
            tool_name
        )

        if tool_to_use is None:
            raise ValueError(
                f"Tool '{tool_name}' not found"
            )

        # Execute the regular Python function directly.
        #
        # This replaces LangChain's:
        #
        # tool_to_use.invoke(tool_args)

        observation = tool_to_use(
            **tool_args
        )

        print(
            f"  [Tool Result] {observation}"
        )

        # Append Gemini's complete model response.
        #
        # This matches the instructor's:
        #
        # messages.append(ai_message)

        messages.append(
            ai_message
        )

        # Create Gemini's provider-specific function response.
        #
        # Part.from_function_response() accepts:
        #
        # - name
        # - response
        #
        # The currently installed SDK does not accept an id
        # argument for this method.

        function_response = (
            types.Part.from_function_response(
                name=tool_name,
                response={
                    "result": observation
                },
            )
        )

        # Append the function result to the same local history.
        #
        # Gemini represents a function result as a
        # function_response Part within a user-role Content object.
        #
        # Although the native role is user, the specialized
        # function_response Part identifies this as a tool result.

        messages.append(
            types.Content(
                role="user",
                parts=[
                    function_response
                ],
            )
        )

    print(
        "ERROR: Max iterations reached "
        "without a final answer"
    )

    return None


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================


if __name__ == "__main__":
    print(
        "Hello Gemini Raw Function Calling!"
    )

    print()

    result = run_agent(
        "What is the price of a laptop "
        "after applying a gold discount?"
    )