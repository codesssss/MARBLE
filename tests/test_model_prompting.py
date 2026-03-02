from types import SimpleNamespace
import unittest
import os
from unittest.mock import MagicMock, patch

from litellm.types.utils import Message

from marble.llms.model_prompting import model_prompting


def _mock_completion_response() -> SimpleNamespace:
    message = Message(role="assistant", content="ok")
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestModelPrompting(unittest.TestCase):
    @patch("marble.llms.model_prompting.litellm.completion")
    def test_model_prompting_uses_custom_base_url(
        self, mock_completion: MagicMock
    ) -> None:
        mock_completion.return_value = _mock_completion_response()

        message = model_prompting(
            llm_model={
                "model": "openai/gpt-4o-mini",
                "base_url": "https://example.com/v1",
            },
            messages=[{"role": "system", "content": "test prompt"}],
            return_num=1,
            max_token_num=128,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]

        self.assertIsInstance(message, Message)
        _, kwargs = mock_completion.call_args
        self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(kwargs["base_url"], "https://example.com/v1")

    @patch("marble.llms.model_prompting.litellm.completion")
    def test_model_prompting_uses_legacy_together_fallback(
        self, mock_completion: MagicMock
    ) -> None:
        mock_completion.return_value = _mock_completion_response()

        model_prompting(
            llm_model="together_ai/TA-sample-model",
            messages=[{"role": "system", "content": "test prompt"}],
            return_num=1,
            max_token_num=128,
            temperature=0.0,
            top_p=None,
            stream=None,
        )

        _, kwargs = mock_completion.call_args
        self.assertEqual(kwargs["base_url"], "https://api.ohmygpt.com/v1")
        self.assertEqual(kwargs["model"], "together_ai/TA-sample-model")

    @patch("marble.llms.model_prompting.litellm.completion")
    def test_model_prompting_uses_env_base_url_when_not_configured(
        self, mock_completion: MagicMock
    ) -> None:
        mock_completion.return_value = _mock_completion_response()

        with patch.dict(os.environ, {"MARBLE_LLM_BASE_URL": "https://env-proxy/v1"}):
            model_prompting(
                llm_model="openai/gpt-4o-mini",
                messages=[{"role": "system", "content": "test prompt"}],
                return_num=1,
                max_token_num=128,
                temperature=0.0,
                top_p=None,
                stream=None,
            )

        _, kwargs = mock_completion.call_args
        self.assertEqual(kwargs["base_url"], "https://env-proxy/v1")
        self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
