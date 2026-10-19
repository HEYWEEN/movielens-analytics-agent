"""Minimal DeepSeek Chat Completions client for server-side agent turns.

The API key is read from the process environment for each request and is never
sent to the browser or included in exceptions. The transport stays small so the
course application does not need a separate SDK dependency.
"""

import codecs
import json
import os
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"


class DeepSeekError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _sse_data(response):
    """Yield complete SSE data records without buffering the whole response."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending, data = "", []
    # HTTPResponse.read1 returns currently available bytes. read would wait for
    # the requested buffer to fill and make short answers appear non-streaming.
    read = getattr(response, "read1", None) or response.read
    while True:
        try:
            chunk = read(4096)
        except (OSError, TimeoutError, HTTPException) as exc:
            raise DeepSeekError("provider_stream_interrupted", "DeepSeek 输出中断，请重试。") from exc
        pending += decoder.decode(chunk, final=not chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if not chunk:
            if pending.startswith("data:"):
                data.append(pending[5:].removeprefix(" ").rstrip("\r"))
            if data:
                yield "\n".join(data)
            return


def _stream_completion(response, model, on_delta):
    content, calls, usage, finish_reason = [], {}, None, None
    finished = False
    for data in _sse_data(response):
        if data.strip() == "[DONE]":
            finished = True
            break
        chunk = json.loads(data)
        if not isinstance(chunk, dict) or "error" in chunk:
            raise DeepSeekError("invalid_provider_response", "DeepSeek API 返回了无效的流响应。")
        model = chunk.get("model") or model
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise ValueError("missing streaming choices")
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("invalid streaming choice")
            if choice.get("index", 0) != 0:
                continue  # Only one answer is requested.
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ValueError("invalid streaming delta")
            if delta.get("role", "assistant") != "assistant":
                raise ValueError("unexpected streaming role")
            text = delta.get("content")
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError("invalid streaming content")
                if text:
                    content.append(text)
                    on_delta(text)
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, dict):
                    raise ValueError("invalid streaming tool call")
                index = fragment.get("index")
                if type(index) is not int or index < 0:
                    raise ValueError("missing streaming tool index")
                call = calls.setdefault(index, {"id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
                for key in ("id", "type"):
                    if fragment.get(key) is not None:
                        if not isinstance(fragment[key], str):
                            raise ValueError("invalid streaming tool field")
                        if key == "id":
                            call[key] += fragment[key]
                        else:
                            call[key] = fragment[key]
                function = fragment.get("function") or {}
                if not isinstance(function, dict):
                    raise ValueError("invalid streaming function")
                for key in ("name", "arguments"):
                    if function.get(key) is not None:
                        if not isinstance(function[key], str):
                            raise ValueError("invalid streaming function field")
                        call["function"][key] += function[key]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if not finished or finish_reason is None:
        raise DeepSeekError("provider_stream_interrupted", "DeepSeek 输出尚未完成，连接已中断，请重试。")
    message = {"role": "assistant", "content": "".join(content) if content else None}
    if calls:
        ordered = [calls[index] for index in sorted(calls)]
        if any(not call["id"] or call["type"] != "function" or
               not call["function"]["name"] for call in ordered):
            raise ValueError("incomplete streaming tool call")
        message["tool_calls"] = ordered
    return {"message": message, "model": model,
            "finish_reason": finish_reason, "usage": usage}


def complete(messages, tools=None, on_delta=None):
    """Return a completion; optionally forward real provider text deltas."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise DeepSeekError("missing_api_key", "未配置 DEEPSEEK_API_KEY；请在启动服务前设置环境变量。")

    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    streaming = on_delta is not None
    payload = {"model": model, "messages": messages, "stream": streaming,
               "thinking": {"type": "disabled"}, "temperature": 0.2,
               "max_tokens": 1200}
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(API_URL, body, {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if streaming else "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            if streaming:
                return _stream_completion(response, model, on_delta)
            result = json.load(response)
    except HTTPError as exc:
        # Provider responses are not reflected to the browser: they can include
        # request details and should not be treated as trusted instructions.
        exc.close()
        if exc.code in (401, 403):
            raise DeepSeekError("authentication_failed", "DeepSeek API 鉴权失败，请检查 API Key。") from exc
        if exc.code == 429:
            raise DeepSeekError("rate_limited", "DeepSeek API 请求受限，请稍后重试。") from exc
        raise DeepSeekError("provider_error", f"DeepSeek API 返回 HTTP {exc.code}。") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise DeepSeekError("provider_unavailable", "无法连接 DeepSeek API，请检查网络后重试。") from exc
    except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
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
