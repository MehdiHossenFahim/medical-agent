"""
Shared core used by all three DB-specific tools (Heart Disease, Cancer, Diabetes).

Each tool needs to:
  1. Connect to its own SQLite database
  2. Turn a natural-language question into SQL, run it, and turn the result back into
     natural language

Rather than duplicating this logic three times, `build_sql_agent()` builds a LangChain
SQL agent (SQLDatabaseToolkit + create_sql_agent) bound to a single database, and each
tool module in this package just points it at a different .db file / table.

LLM provider: controlled by the LLM_PROVIDER env var ("openai" or "groq", default
"openai"). Groq's OpenAI-compatible tool-calling models (e.g. openai/gpt-oss-20b) work
here via `langchain-groq`'s ChatGroq -- no code changes needed, just set:
    LLM_PROVIDER=groq
    GROQ_API_KEY=...
    GROQ_MODEL=openai/gpt-oss-20b   (default)
in your .env.

Notes:
- The agent is created lazily (on first call) and cached, since building the toolkit
  requires an LLM call to inspect schema and is a bit slow to import-time construct.
- agent_type="tool-calling" is used (rather than the OpenAI-specific "openai-tools")
  so this works with any tool-calling-capable chat model, Groq included.
"""
import os
from functools import lru_cache

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


@lru_cache(maxsize=None)
def get_llm(provider: str = LLM_PROVIDER):
    """Return a LangChain chat model for the configured provider. Cached so the
    same instance is reused across tool calls."""
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set (see .env.example).")
        return ChatGroq(model=DEFAULT_GROQ_MODEL, api_key=api_key, temperature=0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set (see .env.example).")
        return ChatOpenAI(model=DEFAULT_OPENAI_MODEL, api_key=api_key, temperature=0)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'groq'.")


@lru_cache(maxsize=None)
def build_sql_agent(db_path: str, provider: str = LLM_PROVIDER):
    """Build (and cache) a LangChain SQL agent scoped to a single SQLite file."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run `python scripts/csv_to_sqlite.py` "
            f"first (after populating data/ -- see data/README.md)."
        )
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = get_llm(provider)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",  # generic, works with OpenAI, Groq, etc.
        verbose=False,
    )
    return agent


def ask_database(db_path: str, question: str, provider: str = LLM_PROVIDER) -> str:
    """Route a natural-language question to the SQL agent for the given database
    and return its natural-language answer."""
    agent = build_sql_agent(db_path, provider=provider)
    try:
        result = agent.invoke({"input": question})
        return result["output"] if isinstance(result, dict) and "output" in result else str(result)
    except Exception as exc:  # keep the outer agent alive even if a query fails
        return f"I couldn't answer that from the database ({exc})."
