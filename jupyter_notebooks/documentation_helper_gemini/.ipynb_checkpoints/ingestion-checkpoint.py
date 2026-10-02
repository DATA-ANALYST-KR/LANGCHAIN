import truststore

truststore.inject_into_ssl()

from pathlib import Path

ENV_PATH = Path("../../langchain-course/.env").resolve()

from dotenv import load_dotenv

load_dotenv(dotenv_path=ENV_PATH)

import asyncio
import os

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone


START_URL = "https://docs.langchain.com/oss/python/langchain/overview"
INDEX_NAME = os.environ["DOCUMENTATION_INDEX_NAME"]
NAMESPACE = "langchain-docs"

BATCH_SIZE = 10
WAIT_SECONDS = 65
MAX_CONCURRENT_BATCHES = 1
DELETE_EXISTING_RECORDS = False


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    output_dimensionality=1536,
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)


def delete_namespace_records():
    """Delete all existing records before a full rebuild."""

    if not DELETE_EXISTING_RECORDS:
        return

    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )

    index_description = pinecone_client.describe_index(
        INDEX_NAME
    )

    pinecone_index = pinecone_client.Index(
        host=index_description.host
    )

    pinecone_index.delete(
        delete_all=True,
        namespace=NAMESPACE,
    )

    print(
        f"Deleted all records from namespace: {NAMESPACE}"
    )


async def index_documents_async(
    documents: list[Document],
    batch_size: int = BATCH_SIZE,
    wait_seconds: int = WAIT_SECONDS,
    max_concurrent_batches: int = MAX_CONCURRENT_BATCHES,
):
    """Upload document batches with controlled Gemini concurrency."""

    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    print("=" * 72)
    print("VECTOR STORAGE PHASE")
    print("=" * 72)
    print(
        f"Preparing to add {len(documents)} chunks "
        f"across {len(batches)} batches."
    )

    semaphore = asyncio.Semaphore(
        max_concurrent_batches
    )

    async def add_batch(batch, batch_number):
        async with semaphore:
            try:
                print(
                    f"Processing batch "
                    f"{batch_number}/{len(batches)}..."
                )

                await vectorstore.aadd_documents(batch)

                print(
                    f"Batch {batch_number}/{len(batches)} "
                    f"uploaded successfully."
                )

                if batch_number < len(batches):
                    print(
                        f"Waiting {wait_seconds} seconds "
                        f"before the next batch."
                    )
                    await asyncio.sleep(wait_seconds)

                return True

            except Exception as error:
                print(
                    f"Batch {batch_number}/{len(batches)} "
                    f"failed: {error}"
                )
                return False

    tasks = [
        add_batch(batch, batch_number)
        for batch_number, batch in enumerate(
            batches,
            start=1,
        )
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    successful_batches = sum(
        result is True
        for result in results
    )

    print(
        f"Successful batches: "
        f"{successful_batches}/{len(batches)}"
    )


async def main():
    print("=" * 72)
    print("DOCUMENT CRAWLING PHASE")
    print("=" * 72)

    tavily_crawl = TavilyCrawl(
        max_depth=2,
        extract_depth="advanced",
    )

    crawl_results = tavily_crawl.invoke({
        "url": START_URL,
    })

    documents = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]},
        )
        for result in crawl_results["results"]
        if result.get("raw_content")
    ]

    print("Documents crawled:", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    delete_namespace_records()

    await index_documents_async(chunks)


if __name__ == "__main__":
    asyncio.run(main())
