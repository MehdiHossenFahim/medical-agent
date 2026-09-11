"""
Top-level entry point.

Usage:
    python main.py                                  # interactive chat, OpenAI Agents SDK
    python main.py --backend langchain              # interactive chat, LangChain AgentExecutor
    python main.py "your question here"             # single question, OpenAI Agents SDK
    python main.py --backend langchain "question"   # single question, LangChain AgentExecutor
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--backend",
        choices=["agents_sdk", "langchain"],
        default="agents_sdk",
        help="Which agent implementation to use (default: agents_sdk).",
    )
    parser.add_argument("question", nargs="*", help="Optional single question (non-interactive mode).")
    args = parser.parse_args()

    if args.backend == "langchain":
        from agent.langchain_app import ask, chat_loop
    else:
        from agent.agents_sdk_app import ask, chat_loop

    if args.question:
        print(ask(" ".join(args.question)))
    else:
        chat_loop()


if __name__ == "__main__":
    main()
