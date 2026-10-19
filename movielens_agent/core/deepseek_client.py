"""Minimal DeepSeek Chat Completions client for server-side agent turns.

The API key is read from the process environment for each request and is never
sent to the browser or included in exceptions. The transport stays small so the
course application does not need a separate SDK dependency.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"


class DeepSeekError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def complete(messages, tools=None):
    """Return one response message, model ID, finish reason and usage."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise DeepSeekError("missing_api_key", "未配置 DEEPSEEK_API_KEY；请在启动服务前设置环境变量。")

    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    payload = {"model": model, "messages": messages, "stream": False,
               "thinking": {"type": "disabled"}, "temperature": 0.2,
               "max_tokens": 1200}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(API_URL, body, {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as exc:
        # Provider responses are not reflected to the browser: they can include
        # request details and should not be treated as trusted instructions.
        if exc.code in (401, 403):
            raise DeepSeekError("authentication_failed", "DeepSeek API 鉴权失败，请检查 API Key。") from exc
        if exc.code == 429:
            raise DeepSeekError("rate_limited", "DeepSeek API 请求受限，请稍后重试。") from exc
        raise DeepSeekError("provider_error", f"DeepSeek API 返回 HTTP {exc.code}。") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise DeepSeekError("provider_unavailable", "无法连接 DeepSeek API，请检查网络后重试。") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise DeepSeekError("invalid_provider_response", "DeepSeek API 返回了无法解析的响应。") from exc

    try:
        choice = result["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("missing assistant message")
        return {
            "message": message,
            "model": result.get("model", model),
            "finish_reason": choice.get("finish_reason"),
            "usage": result.get("usage"),
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DeepSeekError("invalid_provider_response", "DeepSeek API 响应缺少必要字段。") from exc
