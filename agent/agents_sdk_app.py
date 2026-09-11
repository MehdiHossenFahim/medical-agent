"""
Main multi-tool agent built with the OpenAI Agents SDK (`pip install openai-agents`).

Each of the four tools below is a thin `@function_tool` wrapper around the plain
Python functions in tools/*.py:
  - HeartDiseaseDBTool  -> tools.heart_tool.heart_disease_db_tool   (LangChain SQL agent)
  - CancerDBTool        -> tools.cancer_tool.cancer_db_tool          (LangChain SQL agent)
  - DiabetesDBTool      -> tools.diabetes_tool.diabetes_db_tool      (LangChain SQL agent)
  - MedicalWebSearchTool-> tools.web_search_tool.medical_web_search_tool (Tavily search)

Those tools' internal LangChain SQL agents pick their LLM (OpenAI or Groq) via
LLM_PROVIDER, independent of this file -- see tools/db_core.py.

The TOP-LEVEL router agent below is built with the OpenAI Agents SDK, which talks to
an OpenAI-compatible chat-completions endpoint. By default that's OpenAI itself; if
LLM_PROVIDER=groq, we instead point it at Groq's OpenAI-compatible endpoint
(https://api.groq.com/openai/v1) with a Groq API key, so the whole project -- DB
tools AND the router -- can run entirely on a free Groq key (e.g. openai/gpt-oss-20b),
no OpenAI key required.

Run:
    python agent/agents_sdk_app.py "What is the average cholesterol in the heart dataset?"
    python agent/agents_sdk_app.py "What are the symptoms of type 2 diabetes?"
"""
import os
import sys

# Allow running this file directly (`python agent/agents_sdk_app.py`) as well as
# as a module (`python -m agent.agents_sdk_app`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, function_tool, set_tracing_disabled

from tools.heart_tool import heart_disease_db_tool, TOOL_DESCRIPTION as HEART_DESC
from tools.cancer_tool import cancer_db_tool, TOOL_DESCRIPTION as CANCER_DESC
from tools.diabetes_tool import diabetes_db_tool, TOOL_DESCRIPTION as DIABETES_DESC
from tools.web_search_tool import medical_web_search_tool, TOOL_DESCRIPTION as WEB_DESC


@function_tool
def HeartDiseaseDBTool(question: str) -> str:
    return heart_disease_db_tool(question)


HeartDiseaseDBTool.description = HEART_DESC


@function_tool
def CancerDBTool(question: str) -> str:
    return cancer_db_tool(question)


CancerDBTool.description = CANCER_DESC


@function_tool
def DiabetesDBTool(question: str) -> str:
    return diabetes_db_tool(question)


DiabetesDBTool.description = DIABETES_DESC


@function_tool
def MedicalWebSearchTool(query: str) -> str:
    return medical_web_search_tool(query)


MedicalWebSearchTool.description = WEB_DESC


ROUTER_INSTRUCTIONS = """
You are a medical multi-tool assistant with access to four tools:

1. HeartDiseaseDBTool     - SQL over the Heart Disease dataset
2. CancerDBTool           - SQL over the Cancer Prediction dataset
3. DiabetesDBTool         - SQL over the Diabetes dataset
4. MedicalWebSearchTool   - web search for general medical knowledge

Routing rules:
- If the question asks for statistics, counts, averages, correlations, or specific
  records from one of the three datasets (heart disease, cancer, diabetes), call the
  matching *DBTool. If it's unclear which dataset, ask a brief clarifying question.
- If the question asks for a definition, symptoms, causes, risk factors, treatment,
  or cure -- i.e. general medical knowledge not tied to one of the datasets -- call
  MedicalWebSearchTool.
- If a question needs both (e.g. "what's the average glucose in the diabetes dataset
  and what does that mean clinically?"), call both tools and combine the answer.
- Always answer in clear, concise natural language. Briefly mention which
  tool/dataset the answer is based on.
- Never fabricate dataset numbers -- only report numbers a tool actually returned.
"""


def _build_router_model():
    """Return the `model=` argument for Agent(...), based on LLM_PROVIDER.

    - "openai" (default): just pass the model name string; the SDK talks to
      OpenAI directly using OPENAI_API_KEY from the environment.
    - "groq": build an OpenAIChatCompletionsModel bound to an AsyncOpenAI client
      whose base_url points at Groq's OpenAI-compatible endpoint, so the whole
      agent runs on a Groq API key with no OpenAI key needed.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "groq":
        from openai import AsyncOpenAI
        from agents import OpenAIChatCompletionsModel

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set (see .env.example).")

        groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
        )
        model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        # The SDK's built-in tracing always tries to upload to OpenAI's dashboard
        # using OPENAI_API_KEY, regardless of which model you're actually chatting
        # with. Since we have no OpenAI key here, disable tracing to avoid noisy
        # (harmless) 401s -- can also be set globally via OPENAI_AGENTS_DISABLE_TRACING=1.
        set_tracing_disabled(True)
        return OpenAIChatCompletionsModel(model=model_name, openai_client=groq_client)

    # default: plain OpenAI
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


main_agent = Agent(
    name="MedicalMultiToolAgent",
    instructions=ROUTER_INSTRUCTIONS,
    tools=[HeartDiseaseDBTool, CancerDBTool, DiabetesDBTool, MedicalWebSearchTool],
    model=_build_router_model(),
)


def ask(question: str) -> str:
    result = Runner.run_sync(main_agent, question)
    return result.final_output


def chat_loop():
    print("Medical Multi-Tool Agent (OpenAI Agents SDK). Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        print(f"Agent: {ask(question)}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
    else:
        chat_loop()
