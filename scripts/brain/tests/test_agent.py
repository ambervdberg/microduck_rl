"""Pure tests for agent helpers. No network, no env vars needed."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import _invoke_turn, _read_line, azure_endpoint  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


def test_foundry_project_url_reduces_to_scheme_and_host():
    raw = "https://my-resource.services.ai.azure.com/api/projects/my-project"
    assert azure_endpoint(raw) == "https://my-resource.services.ai.azure.com"


def test_bare_azure_openai_url_is_unchanged():
    assert azure_endpoint("https://x.openai.azure.com/") == "https://x.openai.azure.com"


class _FailingAgent:
    def invoke(self, _payload):
        raise RuntimeError("boom")


class _OkAgent:
    def invoke(self, payload):
        return {"messages": [*payload["messages"], AIMessage(content="done")]}


def test_invoke_turn_drops_pending_user_message_on_error(capsys):
    messages = [{"role": "user", "content": "earlier"}, {"role": "user", "content": "walk"}]
    result = _invoke_turn(_FailingAgent(), messages)

    assert result == [{"role": "user", "content": "earlier"}]
    assert "Error: boom" in capsys.readouterr().out


def test_invoke_turn_prints_block_structured_reply(capsys):
    messages = [{"role": "user", "content": "walk"}]
    result = _invoke_turn(_OkAgent(), messages)

    assert result[-1].text == "done"
    assert capsys.readouterr().out.strip() == "done"


def test_read_line_ignores_blank_lines_and_reprompts():
    with patch("builtins.input", side_effect=["", "  ", "walk forward"]):
        assert _read_line() == "walk forward"


def test_read_line_quit_and_exit_end_the_session():
    with patch("builtins.input", side_effect=["quit"]):
        assert _read_line() is None

    with patch("builtins.input", side_effect=["EXIT"]):
        assert _read_line() is None
