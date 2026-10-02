# Tool calls will be parsed from raw model text instead of
# structured function-call JSON.

import inspect
import re

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from langsmith import traceable


MAX_ITERATIONS = 10
MODEL = "gemini-3.6-flash"


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

    # Arguments are extracted from raw text, so price may initially
    # be a string. Convert it before performing the calculation.

    price = float(price)

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


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# ============================================================
# DYNAMIC TOOL DESCRIPTIONS
# ============================================================
#
# CHANGE 3:
# There are no JSON function schemas in this version.
#
# The tool names, signatures, and descriptions are inserted into
# the prompt as ordinary text.
#
# inspect reads information directly from the Python functions.


def get_tool_descriptions(tools_dict):
    descriptions = []

    for tool_name, tool_function in tools_dict.items():
        # @traceable wraps the original Python function.
        #
        # __wrapped__ gives inspect access to the original function
        # rather than the decorator's wrapper function.

        original_function = getattr(
            tool_function,
            "__wrapped__",
            tool_function,
        )

        signature = inspect.signature(
            original_function
        )

        docstring = (
            inspect.getdoc(original_function)
            or ""
        )

        descriptions.append(
            f"{tool_name}{signature} - {docstring}"
        )

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(
    tools
)

tool_names = ", ".join(
    tools.keys()
)


# ============================================================
# REACT PROMPT
# ============================================================


react_prompt = f"""
STRICT RULES: You must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price. Do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use. Do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


# ============================================================
# GEMINI CLIENT
# ============================================================


client = genai.Client()


# ============================================================
# LANGSMITH OUTPUT SERIALIZATION
# ============================================================
#
# This affects only how Gemini's output appears in LangSmith.
# The native Gemini response returned to run_agent is unchanged.


def process_gemini_outputs(response):
    if response is None:
        return {
            "messages": [],
            "error": "Gemini returned no response",
        }

    return {
        "messages": [
            {
                "role": "assistant",
                "content": response.text or "",
            }
        ],
        "model_version": getattr(
            response,
            "model_version",
            None,
        ),
        "response_id": getattr(
            response,
            "response_id",
            None,
        ),
    }


# ============================================================
# TRACED GEMINI CALL
# ============================================================
#
# CHANGE 4:
# No tools are passed through the Gemini API.
#
# Gemini does not know that these Python tools exist through native
# function calling. Gemini only sees their plain-text descriptions
# inside the ReAct prompt.
#
# messages remains in the instructor's simple dictionary format so
# LangSmith can render the input cleanly. The helper translates it
# into Gemini Content objects immediately before the API request.


@traceable(
    name="Gemini Generate Content",
    run_type="llm",
    process_outputs=process_gemini_outputs,
)
def gemini_chat_traced(
    model,
    messages,
    stop_sequences,
):
    gemini_contents = [
        types.Content(
            role=message["role"],
            parts=[
                types.Part(
                    text=message["content"]
                )
            ],
        )
        for message in messages
    ]

    config = types.GenerateContentConfig(
        stop_sequences=stop_sequences,
    )

    return client.models.generate_content(
        model=model,
        contents=gemini_contents,
        config=config,
    )


# ============================================================
# MANUAL REACT AGENT LOOP
# ============================================================


@traceable(name="Gemini ReAct Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # CHANGE 5:
    # One complete prompt replaces separate SystemMessage and
    # HumanMessage objects.

    prompt = react_prompt.format(
        question=question
    )

    # The scratchpad contains previous Gemini outputs and the real
    # observations returned by the Python tools.

    scratchpad = ""

    for iteration in range(
        1,
        MAX_ITERATIONS + 1,
    ):
        print(f"\n--- Iteration {iteration} ---")

        # Every iteration resends:
        #
        # Original ReAct prompt
        # + all previous model outputs
        # + all previous tool observations

        full_prompt = (
            prompt
            + scratchpad
        )

        # The stop sequence prevents Gemini from inventing its own
        # Observation.
        #
        # Generation stops immediately before "\nObservation".
        # Python executes the real function and appends the genuine
        # observation afterward.

        response = gemini_chat_traced(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            stop_sequences=[
                "\nObservation"
            ],
        )

        output = response.text or ""

        print(
            f"LLM Output:\n{output}"
        )

        # ========================================================
        # CHECK FOR FINAL ANSWER
        # ========================================================

        print(
            "  [Parsing] Looking for Final Answer "
            "in LLM output..."
        )

        final_answer_match = re.search(
            r"Final Answer:\s*(.+)",
            output,
        )

        if final_answer_match:
            final_answer = (
                final_answer_match
                .group(1)
                .strip()
            )

            print(
                f"  [Parsed] Final Answer: "
                f"{final_answer}"
            )

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Final Answer: {final_answer}"
            )

            return final_answer

        # ========================================================
        # PARSE ACTION AND ACTION INPUT
        # ========================================================
        #
        # CHANGE 6:
        # Tool calls are extracted from ordinary text using regex.
        #
        # This is more fragile than native function calling because
        # parsing fails if Gemini does not follow the requested format.

        print(
            "  [Parsing] Looking for Action and "
            "Action Input in LLM output..."
        )

        action_match = re.search(
            r"Action:\s*(.+)",
            output,
        )

        action_input_match = re.search(
            r"Action Input:\s*(.+)",
            output,
        )

        if (
            not action_match
            or not action_input_match
        ):
            print(
                "  [Parsing] ERROR: Could not parse "
                "Action/Action Input from LLM output"
            )

            break

        tool_name = (
            action_match
            .group(1)
            .strip()
        )

        tool_input_raw = (
            action_input_match
            .group(1)
            .strip()
        )

        print(
            f"  [Tool Selected] {tool_name} "
            f"with args: {tool_input_raw}"
        )

        # Split comma-separated arguments.
        #
        # Examples accepted:
        #
        # laptop
        #
        # 1299.99, gold
        #
        # price=1299.99, discount_tier=gold

        raw_args = [
            value.strip()
            for value
            in tool_input_raw.split(",")
        ]

        # If Gemini returns key=value format, remove the key.
        # Also remove surrounding quotation marks.

        args = [
            value
            .split("=", 1)[-1]
            .strip()
            .strip("'\"")
            for value in raw_args
        ]

        print(
            f"  [Tool Executing] "
            f"{tool_name}({args})..."
        )

        # ========================================================
        # EXECUTE SELECTED TOOL
        # ========================================================

        if tool_name not in tools:
            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: "
                f"{list(tools.keys())}"
            )
        else:
            observation = str(
                tools[tool_name](*args)
                
            )

        print(
            f"  [Tool Result] {observation}"
        )

        # ========================================================
        # UPDATE THE SCRATCHPAD
        # ========================================================
        #
        # CHANGE 7:
        # History is one growing string.
        #
        # This replaces:
        #
        # messages.append(ai_message)
        # messages.append(tool_result)
        #
        # The complete scratchpad is resent as part of full_prompt
        # during the next iteration.

        scratchpad += (
            f"{output}\n"
            f"Observation: {observation}\n"
            f"Thought:"
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
        "Hello Gemini Manual ReAct Agent!"
    )

    print()

    result = run_agent(
        "What is the price of a laptop "
        "after applying a gold discount?"
    )