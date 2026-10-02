import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(
        description="The URL of the source"
    )


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(
        description="The agent's answer to the query"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer",
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_retries=2,
)

tools = [TavilySearch()]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Search for 3 job postings for an AI engineer "
                        "using LangChain in the Bay Area on LinkedIn "
                        "and list their details."
                    )
                )
            ]
        }
    )

    structured_response = result["structured_response"]

    print(structured_response.answer)

    print("\nSources:")
    for source in structured_response.sources:
        print(source.url)


if __name__ == "__main__":
    main()
