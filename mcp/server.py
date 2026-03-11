import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

load_dotenv(Path(__file__).parent.parent / ".env")

from mcp.server.fastmcp import FastMCP
from data_pipeline.storage import query_hybrid

mcp = FastMCP("spring_docs")

@mcp.tool()
def get_docs(query: str, category: str = None) -> str:
    """
    Get documents for a query.
    Supported documentation categories:
    - spring boot reference documentation
    - spring data redis reference documentation
    - spring data jpa reference documentation
    - spring security reference documentation
    - spring cloud gateway reference documentation

    category can be one of the following:
    - spring-boot
    - spring-data-redis
    - spring-data-jpa
    - spring-security
    - spring-cloud-gateway
    - None (default : all categories)
    
    use english for query
    
    Args:
        query: The search query string.
    Returns:
        List of documents.
    """
    docs = query_hybrid(query, k=5, category=category, use_reranker=False)
    results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]
    return json.dumps(results, ensure_ascii=False)

@mcp.tool()
def get_docs_with_reranker(query: str, category: str = None) -> str:
    """
    Get documents for a query.
    Supported documentation categories:
    - spring boot reference documentation
    - spring data redis reference documentation
    - spring data jpa reference documentation
    - spring security reference documentation
    - spring cloud gateway reference documentation

    category can be one of the following:
    - spring-boot
    - spring-data-redis
    - spring-data-jpa
    - spring-security
    - spring-cloud-gateway
    - None (default : all categories)
    
    use english for query
    10 request limit per minute (Api free plan)
    
    Args:
        query: The search query string.
    Returns:
        List of documents.
    """
    docs = query_hybrid(query, k=5, category=category, use_reranker=True)
    results = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]
    return json.dumps(results, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()