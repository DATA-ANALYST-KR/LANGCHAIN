import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0).bind_tools(
    tools
)