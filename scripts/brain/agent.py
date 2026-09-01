"""Terminal chat agent that drives Microduck through the bridge API.

Needs AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.
Start the sim first: uv run scripts/infer_policy.py --walking <onnx> --new-cmd-obs --bridge 8630
"""

import os
import sys

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI

from tools import ALL_TOOLS

SYSTEM_PROMPT = """You control Microduck, a small bipedal robot, through tools.
The robot balances itself; you only choose what it does.
Speeds are capped by the bridge, so prefer moderate values (walking 0.2 m/s
is normal). Walks stop on their own after their duration. Check status if
unsure what the robot is doing. Answer briefly after acting."""


def make_agent():
    model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    return create_agent(model, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


def main():
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        if not os.environ.get(var):
            sys.exit(f"Missing environment variable: {var}")

    agent = make_agent()
    messages = []
    print("Microduck brain ready. Type what the robot should do, or 'quit'.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": line})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(messages[-1].content)


if __name__ == "__main__":
    main()
