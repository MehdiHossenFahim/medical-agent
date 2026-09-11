"""
MedicalWebSearchTool: general medical knowledge (definitions, symptoms, causes,
treatments/cures) via a web search API. Uses Tavily by default because it returns
a synthesized `answer` field in addition to raw results, which keeps this tool
simple. Swap `_search_tavily` for a SerpAPI/Bing implementation if you prefer --
the public `medical_web_search_tool(query)` signature is all the agents rely on.
"""
import os

TOOL_NAME = "MedicalWebSearchTool"
TOOL_DESCRIPTION = (
    "Searches the web for GENERAL medical knowledge: definitions, symptoms, causes, "
    "risk factors, treatments/cures, prevention. Use this for conceptual/medical "
    "questions that are NOT about the Heart Disease, Cancer, or Diabetes dataset "
    "statistics (for those, use the matching *DBTool instead)."
)


def _search_tavily(query: str, max_results: int = 5) -> str:
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "MedicalWebSearchTool is not configured: set TAVILY_API_KEY in your "
            "environment (see .env.example)."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=f"{query} (medical)",
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
    )

    parts = []
    if response.get("answer"):
        parts.append(response["answer"])

    for result in response.get("results", [])[:3]:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        if content:
            parts.append(f"- {title}: {content} ({url})")

    return "\n\n".join(parts) if parts else "No relevant medical information was found."


def medical_web_search_tool(query: str) -> str:
    """Answer a general medical knowledge question using web search."""
    try:
        return _search_tavily(query)
    except Exception as exc:
        return f"Web search failed ({exc})."
