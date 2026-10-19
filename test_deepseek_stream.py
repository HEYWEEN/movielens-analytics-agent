"""Provider SSE decoding and end-to-end incremental HTTP delivery checks."""

import json
import os
import threading
import unittest
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import server
from movielens_agent.core import deepseek_client


def sse(delta=None, finish=None, **values):
    chunk = {"model": "deepseek-flash", "choices": [{"index": 0,
        "delta": delta or {}, "finish_reason": finish}], **values}
    return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\r\n\r\n").encode()


class FragmentedResponse:
    def __init__(self, fragments):
        self.fragments = iter(fragments)

    def read1(self, _size):
        fragment = next(self.fragments, b"")
        if isinstance(fragment, Exception):
            raise fragment
        return fragment

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DeepSeekStreamingTests(unittest.TestCase):
    def complete(self, fragments, callback):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(deepseek_client, "urlopen", return_value=FragmentedResponse(fragments)) as request:
            result = deepseek_client.complete([{"role": "user", "content": "你好"}], on_delta=callback)
        return result, request.call_args.args[0]

    def test_utf8_network_fragments_empty_choices_and_usage(self):
        wire = (b": keep-alive\r\n\r\n" + sse({"role": "assistant", "content": "你"}) +
                sse({"content": "好。"}) +
                b'data: {"choices":[],"usage":{"total_tokens":7}}\r\n\r\n' +
                sse(finish="stop") + b"data: [DONE]\r\n\r\n")
        # Every UTF-8 character and SSE delimiter spans multiple network reads.
        deltas = []
        result, request = self.complete([wire[i:i + 1] for i in range(len(wire))], deltas.append)
        self.assertEqual(deltas, ["你", "好。"])
        self.assertEqual(result["message"], {"role": "assistant", "content": "你好。"})
        self.assertEqual(result["usage"], {"total_tokens": 7})
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(json.loads(request.data)["stream"])
        self.assertEqual(request.get_header("Accept"), "text/event-stream")

    def test_interleaved_tool_arguments_are_assembled_by_index(self):
        first = {"tool_calls": [
            {"index": 1, "id": "call_b", "type": "function",
             "function": {"name": "get_user_clusters", "arguments": '{"user_'}},
            {"index": 0, "id": "call_a", "type": "function",
             "function": {"name": "predict_high_rating", "arguments": '{"user_id":635,'}}]}
        second = {"tool_calls": [
            {"index": 0, "function": {"arguments": '"movie_id":1}'}},
            {"index": 1, "function": {"arguments": 'id":635}'}}]}
        deltas = []
        result, _ = self.complete([sse(first), sse(second), sse(finish="tool_calls"),
                                   b"data: [DONE]\n\n"], deltas.append)
        self.assertEqual(deltas, [])
        self.assertIsNone(result["message"]["content"])
        calls = result["message"]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_a", "call_b"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"user_id": 635, "movie_id": 1})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"user_id": 635})

    def test_eof_or_network_failure_cannot_report_partial_answer_as_complete(self):
        for tail in ([], [IncompleteRead(b"broken")]):
            with self.subTest(tail=tail):
                deltas = []
                with self.assertRaises(deepseek_client.DeepSeekError) as raised:
                    self.complete([sse({"content": "部分回答"}), *tail], deltas.append)
                self.assertEqual(raised.exception.code, "provider_stream_interrupted")
                self.assertEqual(deltas, ["部分回答"])

    def test_done_without_finish_reason_is_interruption(self):
        with self.assertRaises(deepseek_client.DeepSeekError) as raised:
            self.complete([sse({"content": "未完成"}), b"data: [DONE]\n\n"], lambda text: None)
        self.assertEqual(raised.exception.code, "provider_stream_interrupted")

    def test_invalid_json_is_explicit_and_http_errors_never_reflect_provider_body(self):
        with self.assertRaises(deepseek_client.DeepSeekError) as raised:
            self.complete([b"data: {not-json}\n\n"], lambda text: None)
        self.assertEqual(raised.exception.code, "invalid_provider_response")
        for status, code in ((401, "authentication_failed"), (429, "rate_limited"), (500, "provider_error")):
            with self.subTest(status=status), \
                 patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
                 patch.object(deepseek_client, "urlopen", side_effect=HTTPError(
                     "https://api.deepseek.com", status, "private-test-key", {}, None)):
                with self.assertRaises(deepseek_client.DeepSeekError) as raised:
                    deepseek_client.complete([], on_delta=lambda text: None)
                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("private-test-key", str(raised.exception))

    def test_each_tool_round_resets_text_and_forwards_real_deltas(self):
        first = [sse({"content": "我先读取报告。"}), sse({"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function", "function": {
                "name": "inspect_workspace", "arguments": "{}"}}]}),
            sse(finish="tool_calls"), b"data: [DONE]\n\n"]
        second = [sse({"content": "发现"}), sse({"content": "现有结果。"}),
                  sse(finish="stop"), b"data: [DONE]\n\n"]
        events = []
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(deepseek_client, "urlopen", side_effect=[FragmentedResponse(first), FragmentedResponse(second)]), \
             patch.object(server, "latest_analysis_task", return_value=None), \
             patch.object(server, "latest_source_task", return_value=None), \
             patch.object(server, "_tool_dispatch", return_value={"status": "completed"}):
            result = server.agent_chat_llm("看看现有结果", on_event=events.append)
        self.assertEqual(result["reply"], "发现现有结果。")
        self.assertEqual([event["type"] for event in events], ["phase", "response_start", "delta",
            "tool_start", "tool_result", "phase", "response_start", "delta", "delta"])
        self.assertEqual([event["text"] for event in events if event["type"] == "delta"],
                         ["我先读取报告。", "发现", "现有结果。"])

    def test_interrupted_tool_request_is_not_dispatched(self):
        wire = [sse({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
            "function": {"name": "inspect_workspace", "arguments": "{}"}}]}),
            sse(finish="length"), b"data: [DONE]\n\n"]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(deepseek_client, "urlopen", return_value=FragmentedResponse(wire)), \
             patch.object(server, "latest_analysis_task", return_value=None), \
             patch.object(server, "latest_source_task", return_value=None), \
             patch.object(server, "_tool_dispatch") as dispatch:
            result = server.agent_chat_llm("看看现有结果", on_event=lambda event: None)
        dispatch.assert_not_called()
        self.assertEqual(result["error"], "incomplete_model_response")

    def test_http_delta_arrives_before_provider_finishes(self):
        release, timed_out = threading.Event(), threading.Event()
        class ProviderHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse({"content": "你好"}))
                self.wfile.flush()
                if not release.wait(4):
                    timed_out.set()
                self.wfile.write(sse({"content": "，欢迎。"}) + sse(finish="stop") + b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, *args):
                pass

        provider = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
        website = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threads = [threading.Thread(target=httpd.serve_forever, daemon=True) for httpd in (provider, website)]
        for thread in threads:
            thread.start()
        try:
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
                 patch.object(deepseek_client, "API_URL", f"http://127.0.0.1:{provider.server_port}/"), \
                 patch.object(server, "latest_analysis_task", return_value=None), \
                 patch.object(server, "latest_source_task", return_value=None):
                request = Request(f"http://127.0.0.1:{website.server_port}/api/agent/stream",
                    json.dumps({"message": "你好"}).encode(), {"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=8) as response:
                    events = []
                    while not events or events[-1]["type"] != "delta":
                        events.append(json.loads(response.readline()))
                    self.assertFalse(timed_out.is_set(), "website buffered until provider completion")
                    self.assertEqual(events[-1], {"type": "delta", "text": "你好"})
                    release.set()
                    events.extend(json.loads(line) for line in response)
                self.assertEqual([event["type"] for event in events],
                                 ["phase", "response_start", "delta", "delta", "answer"])
                self.assertEqual(events[-1]["reply"], "你好，欢迎。")
        finally:
            release.set()
            for httpd in (provider, website):
                httpd.shutdown()
                httpd.server_close()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
