"""Terminal chat agent that drives Microduck through the bridge API.

Needs AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.
Start the sim first: uv run scripts/infer_policy.py --walking <onnx> --new-cmd-obs --bridge 8630
Run: UV_PROJECT_ENVIRONMENT=~/.venvs/microduck_brain uv run --project scripts/brain python scripts/brain/agent.py
"""

import os
import sys
from urllib.parse import urlsplit

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from tools import ALL_TOOLS

SYSTEM_PROMPT = """You control Microduck, a small bipedal robot, through tools.
The robot balances itself. You only choose what it does.
Speeds are capped by the bridge, so prefer moderate values (walking 0.2 m/s
is normal). The walk and gesture tools return when the action is finished,
so call them one after another for a sequence ("turn, then walk forward")
and each step happens in order. Check status if unsure what the robot is
doing. Answer briefly after acting."""


def azure_endpoint(raw: str) -> str:
    """Strip any path from an Azure endpoint URL.

    A Foundry project URL carries a project path that the OpenAI client
    must not see. It appends its own /openai/deployments/... path.
    """
    parts = urlsplit(raw)
    return f"{parts.scheme}://{parts.netloc}"


def make_agent():
    """Build the LangChain agent, wired to Azure OpenAI and the bridge tools."""
    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint(os.environ["AZURE_OPENAI_ENDPOINT"]),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

    return create_agent(model, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


def run_turn(agent, messages: list) -> tuple[list, str]:
    """Run one agent turn. Returns the new history and the reply text.

    On failure the pending user message is dropped so history stays
    consistent with what the model actually saw, and the reply is the error.
    """
    try:
        result = agent.invoke({"messages": messages})

    except Exception as exc:  # noqa: BLE001  any tool or model failure becomes the reply
        return messages[:-1], f"Error: {exc}"

    messages = result["messages"]

    return messages, messages[-1].text


def _invoke_turn(agent, messages: list) -> list:
    """Terminal variant of run_turn: prints the reply."""
    messages, reply = run_turn(agent, messages)
    print(reply)

    return messages


def _check_env() -> None:
    """Exit early if any required Azure environment variable is missing."""
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        if not os.environ.get(var):
            sys.exit(f"Missing environment variable: {var}")


def _read_line() -> str | None:
    """Read one non-empty line of user input, reprompting on blank lines.

    Returns None on EOF, interrupt, or 'quit'/'exit'.
    """
    while True:
        try:
            line = input("> ").strip()

        except (EOFError, KeyboardInterrupt):
            return None

        if line.lower() in ("quit", "exit"):
            return None

        if line:
            return line


def _chat_loop(agent) -> None:
    """Read commands from the terminal and run each through the agent."""
    messages = []
    print("Microduck brain ready. Type what the robot should do, or 'quit'.")

    while True:
        line = _read_line()

        if line is None:
            break

        messages.append({"role": "user", "content": line})
        messages = _invoke_turn(agent, messages)


def main():
    """Check env vars, build the agent, and run the terminal chat loop."""
    _check_env()
    agent = make_agent()
    _chat_loop(agent)


if __name__ == "__main__":
    main()
