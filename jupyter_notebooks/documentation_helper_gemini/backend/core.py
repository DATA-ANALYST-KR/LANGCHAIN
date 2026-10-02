import truststore

truststore.inject_into_ssl()

from pathlib import Path

ENV_PATH = Path(
    "../../langchain-course/.env"
).resolve()

from dotenv import load_dotenv

load_dotenv(dotenv_path=ENV_PATH)

import os

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = os.environ["DOCUMENTATION_INDEX_NAME"]
NAMESPACE = "langchain-docs"


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    output_dimensionality=1536,
)


llm = ChatGoogleGenerativeAI(
    model=os.getenv(
        "GEMINI_CHAT_MODEL",
        "gemini-3.5-flash-lite",
    ),
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0,
)


vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4},
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve LangChain documentation to help answer a query."""

    retrieved_docs = retriever.invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n"
        f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


SYSTEM_PROMPT = """You are a LangChain documentation assistant.
Always use the retrieve_context tool before answering.
Answer using the retrieved documentation.
If the retrieved documentation does not contain the answer, say so.
"""


agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)


def run_llm(query: str):
    """Run the documentation agent and return its answer and context."""

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    })

    result_messages = result["messages"]
    answer = result_messages[-1].content

    context = []

    for message in result_messages:
        if (
            isinstance(message, ToolMessage)
            and message.name == "retrieve_context"
            and message.artifact
        ):
            context.extend(message.artifact)

    return {
        "answer": answer,
        "context": context,
    }
