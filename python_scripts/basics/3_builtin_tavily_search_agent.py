import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


search = TavilySearch(max_results=5)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_retries=2,
)

tools = [search]

agent = create_agent(
    model=llm,
    tools=tools,
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

    print(result)


if __name__ == "__main__":
    main()