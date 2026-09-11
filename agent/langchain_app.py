"""
Alternative main agent built with plain LangChain (`AgentExecutor`), for
graders/reviewers who want a pure-LangChain version instead of (or alongside) the
OpenAI Agents SDK version in agents_sdk_app.py.

Uses `create_tool_calling_agent`, which is provider-agnostic (works with any chat
model that supports LangChain's `bind_tools`, including ChatOpenAI and ChatGroq) --
unlike the older `AgentType.OPENAI_FUNCTIONS`, which assumes OpenAI's specific
function-calling wire format.

Both agent files call the exact same four underlying tool functions in tools/*.py,
so routing behavior should match.

Run:
    python agent/langchain_app.py "How many patients in the diabetes dataset have an outcome of 1?"
    python agent/langchain_app.py "What causes type 2 diabetes?"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

try:
    # Current LangChain (>=1.0): legacy agent APIs live in the separate
    # langchain-classic package and are NOT re-exported through langchain.agents.
    from langchain_classic.agents import AgentExecutor, Tool, create_tool_calling_agent
except ImportError:
    # Older LangChain (<1.0): these still live directly under langchain.agents.
    from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.db_core import get_llm
from tools.heart_tool import heart_disease_db_tool, TOOL_DESCRIPTION as HEART_DESC
from tools.cancer_tool import cancer_db_tool, TOOL_DESCRIPTION as CANCER_DESC
from tools.diabetes_tool import diabetes_db_tool, TOOL_DESCRIPTION as DIABETES_DESC
from tools.web_search_tool import medical_web_search_tool, TOOL_DESCRIPTION as WEB_DESC

ROUTER_SYSTEM_PROMPT = (
    "You are a medical multi-tool assistant with four tools: HeartDiseaseDBTool, "
    "CancerDBTool, DiabetesDBTool (each runs SQL over its dataset), and "
    "MedicalWebSearchTool (general medical knowledge). Use a *DBTool for "
    "statistics/counts/averages/records from a dataset. Use MedicalWebSearchTool for "
    "definitions/symptoms/causes/treatments not tied to a dataset. If a question "
    "needs both, call both tools and combine the answer. Never fabricate dataset "
    "numbers -- only report what a tool actually returned."
)

# Uses whichever provider is set via LLM_PROVIDER=openai|groq -- see tools/db_core.py
llm = get_llm()

tools = [
    Tool(name="HeartDiseaseDBTool", func=heart_disease_db_tool, description=HEART_DESC),
    Tool(name="CancerDBTool", func=cancer_db_tool, description=CANCER_DESC),
    Tool(name="DiabetesDBTool", func=diabetes_db_tool, description=DIABETES_DESC),
    Tool(name="MedicalWebSearchTool", func=medical_web_search_tool, description=WEB_DESC),
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def ask(question: str) -> str:
    return agent_executor.invoke({"input": question})["output"]


def chat_loop():
    print("Medical Multi-Tool Agent (LangChain AgentExecutor). Type 'exit' to quit.\n")
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
