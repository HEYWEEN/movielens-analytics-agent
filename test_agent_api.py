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
    def test_compact_report_preserves_nested_measured_values(self):
        value = {'report': {'classification': {'algorithms': {'logistic_regression': {
            'test': {'f1': 0.7454}, 'params': {'C': 1.0}}}}}}
        compact = server._compact_tool_evidence(value)
        self.assertEqual(compact['report']['classification']['algorithms']['logistic_regression']['test']['f1'], 0.7454)

    def test_stream_route_delivers_live_events_and_final_answer(self):
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        def fake_chat(*args, **kwargs):
            kwargs['on_event']({'type': 'phase', 'message': '正在理解你的问题…'})
            kwargs['on_event']({'type': 'tool_start', 'tool': 'inspect_workspace', 'arguments': {}})
            return {'status': 'complete', 'reply': '已读取现有结果。'}
        try:
            request = Request(f'http://127.0.0.1:{httpd.server_port}/api/agent/stream',
                              json.dumps({'message': '看看现有结果', 'user_ids': []}).encode(),
                              {'Content-Type': 'application/json'}, method='POST')
            with patch.object(server, 'agent_chat_llm', side_effect=fake_chat):
                with urlopen(request) as response:
                    self.assertIn('application/x-ndjson', response.headers['Content-Type'])
                    events = [json.loads(line) for line in response]
            self.assertEqual([event['type'] for event in events], ['phase', 'tool_start', 'answer'])
            self.assertEqual(events[-1]['reply'], '已读取现有结果。')
        finally:
            httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)

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
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.args[:2], ("predict_high_rating", {"user_id": 1, "movie_id": 2}))
        self.assertEqual(dispatch.call_args.args[3], "012345abcdef")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args.args[0][-1]["role"], "tool")

    def test_general_analysis_question_can_be_answered_without_forced_ids(self):
        response = {"message": {"role": "assistant", "content": "该用户属于群体 1。"},
                    "model": "deepseek-flash", "finish_reason": "stop"}
        response['message']['content'] = '聚类用于理解用户偏好；你希望看整体还是某位用户？'
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-test-key"}), \
             patch.object(server, "deepseek_complete", return_value=response):
            actual = server.agent_chat_llm("聚类分析能帮助我做什么？")
        self.assertEqual(actual['status'], 'complete')
        self.assertEqual(actual['evidence_status'], 'general_explanation')
        self.assertIn('聚类用于', actual['reply'])

    def test_empty_optional_user_list_does_not_block_greeting(self):
        answer = {'message': {'role': 'assistant', 'content': '你好！想了解哪些电影数据？'},
                  'model': 'deepseek-flash', 'finish_reason': 'stop'}
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'private-test-key'}), \
             patch.object(server, 'deepseek_complete', return_value=answer) as model:
            result = server.agent_chat_llm('你好', selected_user_ids=[])
        self.assertEqual(result['status'], 'complete')
        model.assert_called_once()

    def test_model_can_discover_results_then_read_report_in_one_turn(self):
        def tool(name, call_id):
            return {'message': {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}]},
                'model': 'deepseek-flash', 'finish_reason': 'tool_calls'}
        final = {'message': {'role': 'assistant', 'content': '已读取现有报告，下面解释两种方法的差别。'},
                 'model': 'deepseek-flash', 'finish_reason': 'stop'}
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'private-test-key'}), \
             patch.object(server, 'deepseek_complete', side_effect=[
                 tool('inspect_workspace', 'call_1'), tool('get_analysis_report', 'call_2'), final]), \
             patch.object(server, '_tool_dispatch', return_value={'status': 'completed'}) as dispatch:
            result = server.agent_chat_llm('现有结果说明了什么？')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual([event['tool'] for event in result['tool_events']],
                         ['inspect_workspace', 'get_analysis_report'])

    def test_missing_result_can_be_explained_by_model(self):
        first = {'message': {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'call_1', 'type': 'function', 'function': {
                'name': 'get_user_clusters', 'arguments': '{"user_id":1}'}}]},
                 'model': 'deepseek-flash', 'finish_reason': 'tool_calls'}
        final = {'message': {'role': 'assistant', 'content': '用户 1 的训练期历史不足，因此没有群体归属；可以查看整体分群。'},
                 'model': 'deepseek-flash', 'finish_reason': 'stop'}
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'private-test-key'}), \
             patch.object(server, 'deepseek_complete', side_effect=[first, final]), \
             patch.object(server, '_tool_dispatch', return_value={
                 'status': 'insufficient_history', 'reason': '训练期历史不足'}):
            result = server.agent_chat_llm('看看这个用户的喜好')
        self.assertIn('整体分群', result['reply'])
        self.assertEqual(result['status'], 'unavailable')

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
