import os

import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff


def _resolve_model_and_base_url(
    llm_model: Union[str, Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    """
    Resolve model name and base URL from llm config.

    Supports:
    - llm_model as model string
    - llm_model as dict, e.g. {"model": "...", "base_url": "..."}
    """
    if isinstance(llm_model, dict):
        model_name = llm_model.get("model", "gpt-3.5-turbo")
        if not isinstance(model_name, str):
            model_name = "gpt-3.5-turbo"
        configured_base_url = llm_model.get("base_url")
        base_url = (
            configured_base_url
            if isinstance(configured_base_url, str) and configured_base_url
            else None
        )
    else:
        model_name = llm_model
        base_url = None

    # Optional global override from environment.
    if base_url is None:
        env_base_url = os.getenv("MARBLE_LLM_BASE_URL")
        if env_base_url:
            base_url = env_base_url

    # Keep backward compatibility for legacy together_ai routing.
    if base_url is None and "together_ai/TA" in model_name:
        base_url = "https://api.ohmygpt.com/v1"

    return model_name, base_url


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: Union[str, Dict[str, Any]],
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    # litellm.set_verbose=True
    model_name, base_url = _resolve_model_and_base_url(llm_model)
    completion = litellm.completion(
        model=model_name,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
    )
    message_0: Message = completion.choices[0].message
    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
