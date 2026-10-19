"""Agent boundary checks that do not require a live DeepSeek account."""

import json
import os
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen

import server
from movielens_agent.core import deepseek_client


class DeepSeekAgentTests(unittest.TestCase):
    def test_http_exposes_configuration_and_missing_key_state(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_port}"
        try:
            with patch.dict(os.environ, {}, clear=True):
                with urlopen(base + "/api/config") as response:
                    config = json.load(response)
                with urlopen(base + "/api/source-tasks") as response:
                    sources = json.load(response)
                request = Request(base + "/api/agent/chat",
                                  json.dumps({"message": "预测 UserID 1 对 MovieID 2"}).encode(),
                                  {"Content-Type": "application/json"}, method="POST")
                with urlopen(request) as response:
                    answer = json.load(response)
            self.assertFalse(config["agent"]["configured"])
            self.assertIn("latest_source_task_id", config)
            self.assertIsInstance(sources["source_tasks"], list)
            self.assertEqual(answer["error"], "missing_api_key")
            self.assertNotIn("DEEPSEEK_API_KEY", json.dumps(config))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_missing_key_is_explicit_and_never_calls_provider(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(server, "deepseek_complete") as call:
            response = server.agent_chat_llm("预测 UserID 1 对 MovieID 2 的高分概率")
        self.assertEqual(response["status"], "unavailable")
        self.assertEqual(response["error"], "missing_api_key")
        call.assert_not_called()

    def test_classification_request_invokes_registered_tool_and_returns_evidence(self):
        first = {"message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "predict_high_rating", "arguments": '{"user_id":1,"movie_id":2}'}}]},
            "model": "deepseek-flash", "finish_reason": "tool_calls"}
        second = {"message": {"role": "assistant", "content": "两种模型已完成实际预测。"},
                  "model": "deepseek-flash", "finish_reason": "stop"}
        result = {"status": "completed", "task": "classification", "results": [{"algorithm": "logistic_regression"}]}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(server, "deepseek_complete", side_effect=[first, second]) as call, \
             patch.object(server, "_tool_dispatch", return_value=result) as dispatch:
            response = server.agent_chat_llm("预测 UserID 1 对 MovieID 2", analysis_task_id="012345abcdef")
        self.assertEqual(response["status"], "complete")
        self.assertEqual(response["result"], result)
        self.assertEqual(response["tool_events"][0]["tool"], "predict_high_rating")
        dispatch.assert_called_once_with("predict_high_rating", {"user_id": 1, "movie_id": 2},
                                         None, "012345abcdef", None, None, {})
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args.args[0][-1]["role"], "tool")

    def test_factual_answer_without_tool_is_not_presented_as_verified(self):
        response = {"message": {"role": "assistant", "content": "该用户属于群体 1。"},
                    "model": "deepseek-flash", "finish_reason": "stop"}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(server, "deepseek_complete", return_value=response):
            actual = server.agent_chat_llm("用户 1 属于哪个群体？")
        self.assertEqual(actual["error"], "evidence_required")
        self.assertNotIn("群体 1。", actual["reply"])

    def test_provider_request_uses_environment_key_and_returns_model_id(self):
        provider = {"choices": [{"message": {"role": "assistant", "content": "你好"},
                                  "finish_reason": "stop"}], "model": "deepseek-flash"}
        context = MagicMock()
        context.__enter__.return_value = context
        context.__exit__.return_value = None
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(deepseek_client, "urlopen", return_value=context) as open_request, \
             patch.object(deepseek_client.json, "load", return_value=provider):
            actual = deepseek_client.complete([{"role": "user", "content": "你好"}])
        request = open_request.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
        self.assertEqual(json.loads(request.data)["model"], "deepseek-flash")
        self.assertEqual(actual["message"]["content"], "你好")

    def test_analysis_configuration_is_validated_before_scheduling(self):
        result = server.start_analysis_task("012345abcdef", {"n_clusters": 11})
        self.assertEqual(result["error"], "invalid_configuration")
        result = server.start_analysis_task("012345abcdef", {
            "target_dimensions": 3, "information_threshold": 0.9})
        self.assertEqual(result["error"], "invalid_configuration")

    def test_query_cannot_relabel_published_model_with_new_parameters(self):
        published = {"configuration": {"n_clusters": 5, "target_dimensions": 2,
                                          "information_threshold": None}}
        with patch.object(server, "analysis_report", return_value=published):
            mismatch = server._configuration_mismatch("012345abcdef", {"n_clusters": 8})
            same = server._configuration_mismatch("012345abcdef", {"target_dimensions": 2})
        self.assertEqual(mismatch["error"], "invalid_request")
        self.assertIn("重新训练", mismatch["reason"])
        self.assertIsNone(same)

    def test_selected_analysis_must_match_clean_data_source(self):
        task = {"status": "complete", "source_task_id": "aaaaaaaaaaaa"}
        with patch.object(server, "analysis_status", return_value=task):
            selected, error = server._selected_analysis("bbbbbbbbbbbb", "cccccccccccc")
        self.assertIsNone(selected)
        self.assertEqual(error["error"], "source_mismatch")

    def test_selected_analysis_uses_matching_source_when_id_is_omitted(self):
        task = {"status": "complete", "source_task_id": "aaaaaaaaaaaa"}
        with patch.object(server, "latest_analysis_task", return_value="bbbbbbbbbbbb") as latest, \
             patch.object(server, "analysis_status", return_value=task):
            selected, error = server._selected_analysis(None, "aaaaaaaaaaaa")
        self.assertEqual(selected, "bbbbbbbbbbbb")
        self.assertIsNone(error)
        latest.assert_called_once_with("aaaaaaaaaaaa")

    def test_explicit_configuration_conflict_prevents_new_run(self):
        with patch.object(server, "start_analysis_task") as start:
            result = server._tool_dispatch("start_analysis", {"n_clusters": 8},
                                           "012345abcdef", None,
                                           selected_analysis_config={"n_clusters": 4})
        self.assertEqual(result["error"], "configuration_conflict")
        start.assert_not_called()

    def test_model_cannot_override_page_selected_source_task(self):
        with patch.object(server, "start_analysis_task") as start:
            result = server._tool_dispatch("start_analysis", {"source_task_id": "bbbbbbbbbbbb"},
                                           "aaaaaaaaaaaa", None)
        self.assertEqual(result["error"], "configuration_conflict")
        start.assert_not_called()

    def test_selected_user_ids_reach_analysis_tool(self):
        first = {"message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "compare_reduction", "arguments": "{}"}}]},
            "model": "deepseek-flash", "finish_reason": "tool_calls"}
        second = {"message": {"role": "assistant", "content": "已查询两种降维结果。"},
                  "model": "deepseek-flash", "finish_reason": "stop"}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(server, "deepseek_complete", side_effect=[first, second]), \
             patch.object(server, "_tool_dispatch", return_value={"status": "completed"}) as dispatch:
            response = server.agent_chat_llm("分析选中的用户", selected_user_ids=[635, 1])
        self.assertEqual(response["tool_events"][0]["arguments"]["user_ids"], [635, 1])
        self.assertEqual(dispatch.call_args.args[1]["user_ids"], [635, 1])

    def test_analysis_task_http_route_receives_explicit_configuration(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(f"http://127.0.0.1:{httpd.server_port}/api/analysis/tasks",
                              json.dumps({"source_task_id": "012345abcdef",
                                          "analysis_config": {"n_clusters": 4}}).encode(),
                              {"Content-Type": "application/json"}, method="POST")
            with patch.object(server, "start_analysis_task", return_value={
                "status": "queued", "analysis_task_id": "abcdef123456"}) as start:
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
            start.assert_called_once_with("012345abcdef", {"n_clusters": 4})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
