"""Regression tests for the NanoLang tool runtime (nanow_tools.py).

Exercised entirely offline against a throwaway workspace root:
  * path sandbox (escaping the workspace is refused)
  * file CRUD round-trip (write/read/append/list/move/delete)
  * tolerant tool-call parser (JSON envelope, TOOL: line, func call)
  * bash opt-in, deny-list screening, and workspace cwd
  * graceful failure when a tool is unknown or disabled

Run:
    python3 -m unittest tests.test_tools -v
"""

from __future__ import annotations
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import nanow_tools as tools  # noqa: E402


class TestParser(unittest.TestCase):
    def test_tool_call_json_envelope(self):
        call = tools.parse_tool_call(
            'Sure.\n<tool_call>{"name": "read_file", "args": {"path": "a.txt"}}</tool_call>')
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "a.txt"})

    def test_tool_line_syntax(self):
        call = tools.parse_tool_call('TOOL: write_file {"path": "n.txt", "content": "hi"}')
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args["content"], "hi")

    def test_function_call_syntax(self):
        call = tools.parse_tool_call('read_file(path="notes.txt")', known={"read_file"})
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "notes.txt"})

    def test_plain_text_is_not_a_call(self):
        self.assertIsNone(tools.parse_tool_call("The capital of France is Paris."))

    def test_kwargs_coercion(self):
        parsed = tools._parse_kwargs("path='a', n=3, flag=true, name=hello")
        self.assertEqual(parsed, {"path": "a", "n": 3, "flag": True, "name": "hello"})

    def test_tool_line_prose_recovers_real_tool(self):
        call = tools.parse_tool_call(
            'TOOL: please use read_file {"path": "notes.txt"} now',
            known={"read_file", "write_file"})
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args["path"], "notes.txt")

    def test_tool_line_prose_without_real_tool_is_none(self):
        self.assertIsNone(tools.parse_tool_call(
            "TOOL: Read the file notes.txt", known={"read_file"}))

    def test_name_with_json_object(self):
        call = tools.parse_tool_call(
            'write_file {"path": "todo.txt", "content": "buy milk"}',
            known={"write_file", "read_file"})
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args, {"path": "todo.txt", "content": "buy milk"})


class FakeRunner:
    """Stands in for SmolLM2Runner: replays canned step outputs."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.reg = tools.ToolRegistry(root=self._tmp.name)

    def test_loop_executes_call_then_answers(self):
        runner = FakeRunner([
            'TOOL: write_file {"path": "a.txt", "content": "hi"}',
            "Done: a.txt now contains hi.",
        ])
        out = tools.run_agent(runner, self.reg, "create a.txt", max_steps=3, verbose=False)
        self.assertTrue(out["calls"][0]["ok"])
        self.assertEqual(out["calls"][0]["name"], "write_file")
        self.assertEqual(self.reg.call("read_file", path="a.txt").output, "hi")
        self.assertIn("Done", out["answer"])
        # The observation must be fed back into the next prompt.
        self.assertIn("Observation:", runner.prompts[1])

    def test_prompt_parts_recompose(self):
        prefix, suffix = tools.build_agent_prompt_parts(self.reg, "do x")
        self.assertEqual(prefix + suffix, tools.build_agent_prompt(self.reg, "do x"))
        self.assertTrue(suffix.startswith("Task:"))
        self.assertNotIn("do x", prefix)

    def test_compact_prompt_is_shorter_and_keeps_protocol(self):
        full_prefix, _ = tools.build_agent_prompt_parts(self.reg, "x")
        comp_prefix, _ = tools.build_agent_prompt_parts(self.reg, "x", compact=True)
        self.assertLess(len(comp_prefix), len(full_prefix))
        self.assertIn("TOOL:", comp_prefix)
        self.assertIn("Observation:", comp_prefix)
        self.assertEqual(comp_prefix.count("Task:"), 1)  # a single worked example

    def test_compact_describe_omits_gloss_but_keeps_signatures(self):
        comp = self.reg.describe(compact=True)
        self.assertIn("read_file(path)", comp)
        self.assertNotIn("Read a UTF-8 text file", comp)
        self.assertIn("Read a UTF-8 text file", self.reg.describe())

    def test_run_agent_honours_compact_flag(self):
        runner = FakeRunner(["no tool here"])
        tools.run_agent(runner, self.reg, "noop", compact=True, verbose=False)
        self.assertIn("You are an agent with tools.", runner.prompts[0])
        self.assertNotIn("Available tools (call with", runner.prompts[0])
        full = FakeRunner(["no tool here"])
        tools.run_agent(full, self.reg, "noop", compact=False, verbose=False)
        self.assertIn("Available tools (call with", full.prompts[0])

    def test_identical_call_repeats_are_deduped(self):
        # A greedy model can latch onto one call; the loop must stop early and
        # answer with the first result instead of re-running the same call.
        runner = FakeRunner([
            'TOOL: write_file {"path": "loop.txt", "content": "once"}',
            'TOOL: write_file {"path": "loop.txt", "content": "once"}',
        ])
        out = tools.run_agent(runner, self.reg, "create loop.txt", max_steps=4,
                              verbose=False)
        self.assertEqual(len(out["calls"]), 1)
        self.assertIn("wrote", out["answer"])
        self.assertEqual(self.reg.call("read_file", path="loop.txt").output, "once")

    def test_prime_prefixes_cue_and_parses_continuation(self):
        runner = FakeRunner([
            'read_file {"path": "notes.txt"}',   # continuation after "TOOL: "
            "The secret word is PINEAPPLE.",
        ])
        self.reg.call("write_file", path="notes.txt", content="PINEAPPLE")
        out = tools.run_agent(runner, self.reg, "read notes.txt", max_steps=2,
                              verbose=False, prime=True)
        self.assertEqual(out["calls"][0]["name"], "read_file")
        self.assertTrue(out["calls"][0]["ok"])
        self.assertIn("TOOL: ", runner.prompts[0])


class TestSandboxAndCrud(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.reg = tools.ToolRegistry(root=self._tmp.name)

    def test_crud_round_trip(self):
        self.assertTrue(self.reg.call("write_file", path="notes.txt", content="hello").ok)
        self.assertTrue(self.reg.call("append_file", path="notes.txt", content=" world").ok)
        self.assertEqual(self.reg.call("read_file", path="notes.txt").output, "hello world")
        listing = self.reg.call("list_dir").output
        self.assertIn("notes.txt", listing)
        self.assertTrue(self.reg.call("delete_file", path="notes.txt").ok)
        self.assertFalse(self.reg.call("read_file", path="notes.txt").ok)

    def test_nested_create_and_move(self):
        self.assertTrue(self.reg.call("write_file", path="a/b/c.txt", content="x").ok)
        self.assertTrue(self.reg.call("move_file", src="a/b/c.txt", dst="top.txt").ok)
        self.assertEqual(self.reg.call("read_file", path="top.txt").output, "x")

    def test_path_escape_is_refused(self):
        result = self.reg.call("read_file", path="../../etc/passwd")
        self.assertFalse(result.ok)
        self.assertIn("escapes workspace", result.output)

    def test_unknown_tool_reports_error(self):
        result = self.reg.call("nope_tool")
        self.assertFalse(result.ok)
        self.assertIn("unknown tool", result.output)


class TestBash(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_bash_disabled_by_default(self):
        reg = tools.ToolRegistry(root=self._tmp.name)
        result = reg.call("run_bash", command="echo hi")
        self.assertFalse(result.ok)
        self.assertIn("disabled", result.output)

    def test_bash_runs_in_workspace(self):
        reg = tools.ToolRegistry(root=self._tmp.name, allow_bash=True)
        result = reg.call("run_bash", command="pwd && echo hi")
        self.assertTrue(result.ok)
        self.assertIn(os.path.realpath(self._tmp.name), result.output)
        self.assertIn("hi", result.output)

    def test_deny_list_blocks_destructive(self):
        reg = tools.ToolRegistry(root=self._tmp.name, allow_bash=True)
        for command in ["rm -rf /", "sudo reboot", "mkfs.ext4 /dev/sda"]:
            result = reg.call("run_bash", command=command)
            self.assertFalse(result.ok, command)
            self.assertIn("refused", result.output, command)


class TestWebSearch(unittest.TestCase):
    def test_search_disabled_without_injection(self):
        reg = tools.ToolRegistry(root=tempfile.mkdtemp())
        self.assertFalse(reg.call("web_search", query="x").ok)

    def test_search_uses_injected_callable(self):
        reg = tools.ToolRegistry(root=tempfile.mkdtemp(),
                                 search=lambda q: f"result for {q}")
        result = reg.call("web_search", query="llm")
        self.assertTrue(result.ok)
        self.assertIn("result for llm", result.output)

    def test_describe_lists_tools(self):
        reg = tools.ToolRegistry(root=tempfile.mkdtemp())
        desc = reg.describe()
        for name in ["read_file", "write_file", "run_bash", "web_search"]:
            self.assertIn(name, desc)

    def test_agent_prompt_has_protocol_and_fewshot(self):
        reg = tools.ToolRegistry(root=tempfile.mkdtemp())
        prompt = tools.build_agent_prompt(reg, "do a thing")
        self.assertIn("TOOL:", prompt)
        self.assertIn("write_file", prompt)
        self.assertIn("Observation:", prompt)   # few-shot shows the full loop
        self.assertIn("Task: do a thing", prompt)

    def test_fewshot_can_be_disabled(self):
        reg = tools.ToolRegistry(root=tempfile.mkdtemp())
        self.assertNotIn("Example 1",
                         tools.build_agent_prompt(reg, "x", fewshot=False))


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.reg = tools.ToolRegistry(root=tempfile.mkdtemp())

    def test_unknown_argument_rejected(self):
        result = self.reg.call("read_file", path="a.txt", q="x")
        self.assertFalse(result.ok)
        self.assertIn("unknown argument", result.output)

    def test_missing_required_argument(self):
        result = self.reg.call("write_file", content="hi")
        self.assertFalse(result.ok)
        self.assertIn("missing required", result.output)

    def test_optional_argument_defaults(self):
        self.assertTrue(self.reg.call("write_file", path="a.txt").ok)

    def test_signature_format(self):
        self.assertEqual(self.reg.signature("write_file"), "write_file(path, content?)")
        self.assertEqual(self.reg.signature("move_file"), "move_file(src, dst)")


class TestRouter(unittest.TestCase):
    def test_create_file(self):
        call = tools.route_task("create the file todo.txt with the content buy milk")
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args, {"path": "todo.txt", "content": "buy milk"})

    def test_read_file(self):
        call = tools.route_task("read the file notes.txt and report its contents")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "notes.txt"})

    def test_append_extracts_content(self):
        call = tools.route_task("append the text bye to hello.txt")
        self.assertEqual(call.name, "append_file")
        self.assertEqual(call.args["path"], "hello.txt")
        self.assertEqual(call.args["content"], "the text bye")

    def test_other_verbs(self):
        self.assertEqual(tools.route_task("delete report.txt").name, "delete_file")
        self.assertEqual(tools.route_task("list the files in the folder").name, "list_dir")
        self.assertEqual(tools.route_task("run the shell command ls -la").name, "run_bash")
        self.assertEqual(tools.route_task("search the web for best llm 2026").name,
                         "web_search")
        self.assertEqual(tools.route_task("move notes.txt to archive.txt").name, "move_file")
        self.assertEqual(tools.route_task("create a directory data").name, "make_dir")

    def test_no_match_returns_none(self):
        self.assertIsNone(tools.route_task("what is the meaning of life"))

    def test_plan_sequence_inherits_path(self):
        plan = tools.route_plan("create todo.txt with the content buy milk then read it")
        self.assertEqual([c.name for c in plan], ["write_file", "read_file"])
        self.assertEqual(plan[1].args["path"], "todo.txt")

    def test_plan_splits_on_semicolon(self):
        plan = tools.route_plan("list the files; then delete report.txt")
        self.assertEqual([c.name for c in plan], ["list_dir", "delete_file"])

    def test_plan_skips_non_tool_part(self):
        self.assertEqual([c.name for c in tools.route_plan("read report.txt and summarize it")],
                         ["read_file"])

    def test_plan_empty_when_no_tool(self):
        self.assertEqual(tools.route_plan("tell me a joke"), [])

    def test_route_planner_runs_full_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = tools.ToolRegistry(root=tmp)
            runner = FakeRunner(["Done: todo.txt says buy milk."])
            out = tools.run_agent(
                runner, reg, "create todo.txt with the content buy milk then read it",
                planner="route", verbose=False)
            self.assertEqual([c["name"] for c in out["calls"]],
                             ["write_file", "read_file"])
            self.assertTrue(all(c["ok"] for c in out["calls"]))

    def test_route_planner_executes_and_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = tools.ToolRegistry(root=tmp)
            runner = FakeRunner(["Done: I created todo.txt."])
            out = tools.run_agent(
                runner, reg, "create the file todo.txt with the content buy milk",
                planner="route", verbose=False)
            self.assertEqual(out["planner"], "route")
            self.assertTrue(out["calls"][0]["ok"])
            self.assertEqual(reg.call("read_file", path="todo.txt").output, "buy milk")
            self.assertIn("Done", out["answer"])


if __name__ == "__main__":
    unittest.main()
