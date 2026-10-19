"""NanoLang tool runtime -- file CRUD, bash, HTTP, and web search for agents.

This is the "Use It" half of the tool-use surface: a tiny, sandboxed,
standard-library-only registry plus a tolerant parser for the model's calls.
The model never touches the host directly. It emits a call, the runner parses
it, the registry executes it under a workspace root, and the observation is
fed back into the prompt for the next step.

Guardrails (anti-halu / least privilege):
  * every path is resolved and confined to the workspace root;
  * bash is opt-in and screened by a destructive-command deny-list;
  * commands and HTTP fetches have timeouts and capped output.

The three-layer design (parser -> registry -> sandbox) mirrors how production
agent runtimes separate "what the model asked for" from "what the host allows".
Design notes: the JSON tool-call envelope follows the convention used by
OpenAI/Anthropic function calling; the deny-list is informed by the OWASP LLM
Top 10 "excessive agency" risk. Stdlib only, no external deps.

Run the standalone CLI:
    python3 nanow_tools.py list
    python3 nanow_tools.py read notes.txt
    python3 nanow_tools.py write notes.txt "hello"
    python3 nanow_tools.py bash "ls -la"
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Results and calls
# ---------------------------------------------------------------------------

MAX_OUTPUT = 4000


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    """Cap text, keeping the head and tail so both context ends survive."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} chars truncated] ...\n{text[-tail:]}"


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict = field(default_factory=dict)

    def render(self) -> str:
        tag = "ok" if self.ok else "error"
        return f"[{tag}] {truncate(self.output)}"


@dataclass
class ToolCall:
    name: str
    args: dict


# ---------------------------------------------------------------------------
# Tolerant parser for model-emitted tool calls
# ---------------------------------------------------------------------------

_TOOLCALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TOOL_LINE = re.compile(
    r"^[ \t]*TOOL[ \t]*:[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*(.*)$", re.M)
_FUNC_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(([^{}()]*)\)")
_NAME_JSON = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(\{[^{}]*\})")
_KWARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)(?=\s*,\s*[A-Za-z_]|\s*$)")


def _balanced_objects(text: str) -> list[str]:
    """Return every top-level balanced ``{...}`` substring in ``text``."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
    return out


def _coerce(value: str):
    """Turn a textual argument into bool/int/float/str without surprises."""
    v = value.strip().strip('"').strip("'")
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_kwargs(raw: str) -> dict:
    """Parse ``a=1, b='x', c="y"`` style arguments (best effort, never raises)."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval("{" + raw + "}")
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError, TypeError):
        pass
    args = {}
    for key, value in _KWARG.findall(raw):
        args[key] = _coerce(value)
    return args


def _from_json_obj(obj: dict) -> Optional[ToolCall]:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not isinstance(name, str):
        return None
    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(args, str):
        args = _parse_kwargs(args)
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=name, args=args)


def _find_known_name(text: str, known: set) -> Optional[str]:
    """Earliest known tool name *used as a call* (followed by ``(`` or ``{``)."""
    best, best_pos = None, len(text) + 1
    for name in known:
        match = re.search(r"\b" + re.escape(name) + r"\b\s*(?:[({]|$)", text)
        if match and match.start() < best_pos:
            best, best_pos = name, match.start()
    return best


def parse_tool_call(text: str, known: Optional[set] = None) -> Optional[ToolCall]:
    """Extract the first tool call from model output, or ``None``.

    Accepts, in priority order:
      1. ``<tool_call>{"name": ..., "args": {...}}</tool_call>``
      2. ``TOOL: name {"arg": value}`` on its own line
      3. ``{"name": ..., "args": {...}}`` written as bare JSON
      4. ``name(arg=value, ...)`` function-call syntax
    """
    if not text:
        return None

    block = _TOOLCALL_BLOCK.search(text)
    if block:
        for obj_s in _balanced_objects(block.group(1)):
            try:
                call = _from_json_obj(json.loads(obj_s))
            except json.JSONDecodeError:
                call = None
            if call:
                return call

    line = _TOOL_LINE.search(text)
    if line:
        name, rest = line.group(1), line.group(2).strip().lstrip(":").strip()
        if known is not None and name not in known:
            # A weak model often writes prose after the cue ("TOOL: Read the
            # file ..."). Recover by finding a real tool name in the text.
            alt = _find_known_name(text, known)
            if alt is None:
                return None
            name, rest = alt, text.split(alt, 1)[1].lstrip(": ")
        for obj_s in _balanced_objects(rest):
            try:
                obj = json.loads(obj_s)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            named = _from_json_obj(obj)          # {"name": ..., "args": {...}}
            if named and named.name:
                return named
            return ToolCall(name=name, args=obj)  # {"path": ...} is the args bag
        return ToolCall(name=name, args=_parse_kwargs(rest))

    for name, obj_s in _NAME_JSON.findall(text):
        if known is None or name in known:
            try:
                obj = json.loads(obj_s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return ToolCall(name=name, args=obj)

    for obj_s in _balanced_objects(text):
        try:
            call = _from_json_obj(json.loads(obj_s))
        except json.JSONDecodeError:
            continue
        if call and (known is None or call.name in known):
            return call

    for name, raw in _FUNC_CALL.findall(text):
        if known is None or name in known:
            return ToolCall(name=name, args=_parse_kwargs(raw))
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Destructive patterns refused before execution. This is a seatbelt, not a
# sandbox: it stops the obvious footguns an agent or a copy-paste would fire.
DEFAULT_DENY = [
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(?:\s|$)",
    r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/(?:\s|$)",
    r"\bmkfs\b",
    r"\bdd\s+[^|;]*of=/dev/",
    r":\(\)\s*\{.*\}\s*;?\s*:",           # fork bomb
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r">\s*/dev/(sd|nvme|hd)",
    r"\bchmod\s+-R\s+777\s+/",
    r"\b(curl|wget)\b[^|;]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b",
    r"\bsudo\b",
    r"\bgit\s+push\b",
    r"\b(crontab|systemctl|mount)\b",
]


class ToolError(Exception):
    """Raised by a tool handler for a user-visible failure."""


# Per-tool argument schema: name -> {arg: (type, required)}. The registry
# validates calls against this *before* dispatch, so a malformed call fails
# with a useful message instead of a raw TypeError (or a bad side effect).
SCHEMAS: dict[str, dict[str, tuple[str, bool]]] = {
    "list_dir":    {"path": ("str", False)},
    "read_file":   {"path": ("str", True)},
    "write_file":  {"path": ("str", True), "content": ("str", False)},
    "append_file": {"path": ("str", True), "content": ("str", False)},
    "delete_file": {"path": ("str", True)},
    "make_dir":    {"path": ("str", True)},
    "move_file":   {"src": ("str", True), "dst": ("str", True)},
    "http_get":    {"url": ("str", True)},
    "web_search":  {"query": ("str", True)},
    "run_bash":    {"command": ("str", True)},
}


class ToolRegistry:
    """A sandboxed set of tools an agent may call.

    ``root`` confines every file operation. ``search`` is an injected callable
    (e.g. ``WebRetriever.ground``) so this module stays free of the runner.
    """

    def __init__(self, root: str | os.PathLike = "nanow_workspace",
                 allow_bash: bool = False, bash_timeout: float = 20.0,
                 max_output: int = MAX_OUTPUT,
                 search: Optional[Callable[[str], str]] = None,
                 deny_patterns: Optional[list[str]] = None):
        self.root = Path(os.path.realpath(Path(root)))
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_bash = allow_bash
        self.bash_timeout = bash_timeout
        self.max_output = max_output
        self.search = search
        self.deny = [re.compile(p) for p in (deny_patterns or DEFAULT_DENY)]
        self._tools: dict[str, tuple[str, str, bool, Callable]] = {}
        self._errors: list[str] = []
        self._register_all()

    # -- registration -------------------------------------------------------

    def tool(self, name: str, summary: str, params: str = "", dangerous: bool = False):
        def deco(fn):
            self._tools[name] = (summary, params, dangerous, fn)
            return fn
        return deco

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self, compact: bool = False) -> str:
        """Tool list for the prompt.

        ``compact`` drops the per-tool gloss and the protocol hint (which then
        lives in the header). The gloss is what teaches a *human*; the model
        only needs the name and its parameter names, so the long form costs
        ~100 tokens of prefill for no measurable selection benefit.
        """
        if compact:
            return "Tools:\n" + " ".join(self.signature(n) for n in self.names)
        lines = ["Available tools (call with: TOOL: <name> {\"arg\": value}):"]
        for name in self.names:
            summary, params, dangerous, _ = self._tools[name]
            flag = " [dangerous]" if dangerous else ""
            lines.append(f"- {self.signature(name)}: {summary}{flag}")
        return "\n".join(lines)

    def signature(self, name: str) -> str:
        """Human/library signature like ``write_file(path, content?)``."""
        schema = SCHEMAS.get(name)
        if not schema:
            return f"{name}(...)"
        parts = [k if req else f"{k}?" for k, (_, req) in schema.items()]
        return f"{name}({', '.join(parts)})"

    def required_args(self, name: str) -> list[str]:
        return [k for k, (_, req) in SCHEMAS.get(name, {}).items() if req]

    def _validate(self, name: str, args: dict) -> dict:
        """Check and normalize call arguments against the schema."""
        schema = SCHEMAS.get(name)
        if not schema:
            return args
        unknown = [k for k in args if k not in schema]
        if unknown:
            raise ToolError(f"unknown argument(s) {unknown} for {self.signature(name)}")
        missing = [k for k, (_, req) in schema.items() if req and k not in args]
        if missing:
            raise ToolError(f"missing required argument(s) {missing} for {self.signature(name)}")
        out = {}
        for key, value in args.items():
            if schema[key][0] == "str":
                out[key] = "" if value is None else str(value)
            else:
                out[key] = value
        return out

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, **args) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(False, f"unknown tool '{name}'. Known: {', '.join(self.names)}")
        try:
            args = self._validate(name, args)
            return entry[3](**args)
        except ToolError as exc:
            return ToolResult(False, str(exc))
        except TypeError as exc:
            return ToolResult(False, f"bad arguments for '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            return ToolResult(False, f"{type(exc).__name__}: {exc}")

    # -- path sandbox -------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = Path(os.path.expanduser(str(path)))
        if not p.is_absolute():
            p = self.root / p
        p = Path(os.path.realpath(p))
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path escapes workspace: {path!r}")
        return p

    def _cap(self, text: str) -> str:
        return truncate(text, self.max_output)

    # -- tools --------------------------------------------------------------

    def _register_all(self):
        self.tool("list_dir", "List files in a directory", "path='.'")(self._list_dir)
        self.tool("read_file", "Read a UTF-8 text file", "path")(self._read_file)
        self.tool("write_file", "Create or overwrite a file", "path, content")(self._write_file)
        self.tool("append_file", "Append text to a file", "path, content")(self._append_file)
        self.tool("delete_file", "Delete a file", "path")(self._delete_file)
        self.tool("make_dir", "Create a directory (and parents)", "path")(self._make_dir)
        self.tool("move_file", "Rename or move a file", "src, dst")(self._move_file)
        self.tool("http_get", "Fetch a URL and return its text", "url")(self._http_get)
        self.tool("web_search", "Search the live web for a query",
                  "query")(self._web_search)
        self.tool("run_bash", "Run a shell command inside the workspace",
                  "command", dangerous=True)(self._run_bash)

    def _list_dir(self, path: str = ".") -> ToolResult:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"no such directory: {path}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        rows = []
        for child in sorted(target.iterdir()):
            kind = "dir " if child.is_dir() else "file"
            size = "" if child.is_dir() else f" {child.stat().st_size:>8} B"
            rows.append(f"{kind}{size}  {child.name}")
        return ToolResult(True, "\n".join(rows) or "(empty directory)",
                          {"count": len(rows)})

    def _read_file(self, path: str) -> ToolResult:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        data = target.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"not a UTF-8 text file: {path} ({len(data)} bytes)")
        return ToolResult(True, self._cap(text), {"bytes": len(data)})

    def _write_file(self, path: str, content: str = "") -> ToolResult:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        n = len(str(content))
        return ToolResult(True, f"wrote {n} bytes to {target.relative_to(self.root)}",
                          {"bytes": n})

    def _append_file(self, path: str, content: str = "") -> ToolResult:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(str(content))
        return ToolResult(True, f"appended {len(str(content))} bytes to "
                                f"{target.relative_to(self.root)}")

    def _delete_file(self, path: str) -> ToolResult:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"no such file: {path}")
        if target.is_dir():
            raise ToolError("delete_file refuses directories; remove files first")
        target.unlink()
        return ToolResult(True, f"deleted {target.relative_to(self.root)}")

    def _make_dir(self, path: str) -> ToolResult:
        target = self._resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"created directory {target.relative_to(self.root)}")

    def _move_file(self, src: str, dst: str) -> ToolResult:
        source, dest = self._resolve(src), self._resolve(dst)
        if not source.exists():
            raise ToolError(f"no such file: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.replace(dest)
        return ToolResult(True, f"moved {source.relative_to(self.root)} -> "
                                f"{dest.relative_to(self.root)}")

    def _http_get(self, url: str = "", max_bytes: int = 100_000) -> ToolResult:
        if not str(url).startswith(("http://", "https://")):
            raise ToolError("http_get needs an http(s) URL")
        req = urllib.request.Request(str(url), headers={
            "User-Agent": "NanoLang/0.1 (educational tool runtime)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(max_bytes)
            text = raw.decode("utf-8", errors="replace")
        return ToolResult(True, self._cap(text), {"bytes": len(raw)})

    def _web_search(self, query: str = "") -> ToolResult:
        if self.search is None:
            raise ToolError("web_search disabled (start the runner with --online)")
        if not str(query).strip():
            raise ToolError("web_search needs a non-empty query")
        out = self.search(str(query))
        if not out:
            return ToolResult(False, "no results")
        return ToolResult(True, self._cap(out))

    def _screen_command(self, command: str) -> Optional[str]:
        for pattern in self.deny:
            if pattern.search(command):
                return pattern.pattern
        return None

    def _run_bash(self, command: str = "") -> ToolResult:
        if not self.allow_bash:
            raise ToolError("run_bash disabled (start with --allow-bash)")
        command = str(command).strip()
        if not command:
            raise ToolError("run_bash needs a command")
        blocked = self._screen_command(command)
        if blocked:
            return ToolResult(False, f"refused: command matches deny-list ({blocked})")
        proc = subprocess.run(command, shell=True, cwd=self.root, timeout=self.bash_timeout,
                              capture_output=True, text=True, executable="/bin/bash")
        combined = (proc.stdout or "") + (proc.stderr or "")
        tail = f"\n[exit {proc.returncode}]"
        return ToolResult(proc.returncode == 0, self._cap(combined) + tail,
                          {"exit": proc.returncode})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

FEWSHOT = (
    "Example 1\n"
    "Task: read the file notes.txt and report its contents\n"
    'TOOL: read_file {"path": "notes.txt"}\n'
    "Observation: [ok] the secret word is PINEAPPLE\n"
    "Answer: The file says: the secret word is PINEAPPLE.\n\n"
    "Example 2\n"
    "Task: create the file hello.txt with the content hi\n"
    'TOOL: write_file {"path": "hello.txt", "content": "hi"}\n'
    "Observation: [ok] wrote 2 bytes to hello.txt\n"
    "Answer: Done. I created hello.txt.\n\n"
    "Example 3\n"
    "Task: append the text bye to the file hello.txt\n"
    'TOOL: append_file {"path": "hello.txt", "content": " bye"}\n'
    "Observation: [ok] appended 4 bytes to hello.txt\n"
    "Answer: Done. I appended ' bye' to hello.txt.\n"
)

# One example carries the format; the other two mostly repeat it. With
# constrained decoding forcing the tool name and its argument keys, the model
# does not need three demonstrations to learn the pattern.
COMPACT_FEWSHOT = (
    "Example\n"
    "Task: read the file notes.txt\n"
    'TOOL: read_file {"path": "notes.txt"}\n'
    "Observation: [ok] the secret word is PINEAPPLE\n"
    "Answer: The file says: the secret word is PINEAPPLE.\n"
)


def build_agent_prompt_parts(registry: ToolRegistry, task: str,
                             fewshot: bool = True,
                             compact: bool = False) -> tuple[str, str]:
    """Return ``(static_prefix, task_suffix)`` for the agent prompt.

    The prefix holds the tool preamble and few-shot examples; it does not depend
    on the task, so it is the part worth caching a KV prefill for.

    ``compact`` shrinks that prefix from ~444 to ~230 tokens (shorter header,
    one-line tool list, a single worked example), which cuts the one-time
    prefill and the on-disk cache proportionally.
    """
    if compact:
        prefix = (
            "You are an agent with tools.\n"
            'Reply with one line: TOOL: <name> {"arg": "value"}\n'
            "or the final answer. Wait for Observation after each call.\n\n"
            + registry.describe(compact=True) + "\n\n"
            + (COMPACT_FEWSHOT + "\n" if fewshot else "")
        )
        return prefix, f"Task: {task}\n"
    prefix = (
        "You are an agent that can use tools.\n"
        f"{registry.describe()}\n\n"
        "To use a tool, write one line exactly like:\n"
        'TOOL: read_file {"path": "notes.txt"}\n'
        "Then stop and wait for the Observation. When you have the final answer, "
        "write it without any TOOL line.\n\n"
        + (FEWSHOT + "\n" if fewshot else "")
    )
    return prefix, f"Task: {task}\n"


def build_agent_prompt(registry: ToolRegistry, task: str, fewshot: bool = True) -> str:
    """Compose the tool protocol preamble for a completion-style model.

    ``fewshot`` adds three worked examples. For a small model this does far more
    for tool *selection* than any prompt wording alone.
    """
    prefix, suffix = build_agent_prompt_parts(registry, task, fewshot)
    return prefix + suffix


def _first_quoted(text: str) -> Optional[str]:
    m = re.search(r"""["']([^"']+)["']""", text)
    return m.group(1) if m else None


def _first_path(text: str) -> Optional[str]:
    m = re.search(r"([\w./-]+\.[A-Za-z0-9]{1,8})", text)
    return m.group(1) if m else None


def _content_phrase(text: str) -> Optional[str]:
    """Extract the content after the most specific 'with/containing' marker."""
    markers = ["with the content", "with content", "containing", "contains",
               "content:", "saying", "that says", "with"]
    for marker in markers:
        m = re.search(r"\b" + re.escape(marker) + r"\b\s*:?\s*(.+)$", text, re.I)
        if m:
            value = m.group(1).strip().rstrip(".")
            if value:
                quoted = _first_quoted(value)
                return quoted if quoted else value
    return None


def route_task(task: str, registry: Optional[ToolRegistry] = None) -> Optional[ToolCall]:
    """Deterministic intent router: map a plain-language task to a tool call.

    This is the "small classifier before the LLM" pattern. It is deliberately
    simple and transparent (regexes over the common verbs) and it is what makes
    the agent dependable on a small model that cannot be trusted to pick the
    tool on its own. Returns ``None`` when nothing matches.
    """
    t = task.strip()
    low = t.lower()

    def has(*words: str) -> bool:
        return any(w in low for w in words)

    url = re.search(r"(https?://\S+)", t)
    if url and has("fetch", "download", "http"):
        return ToolCall("http_get", {"url": url.group(1).rstrip(".,")})
    if url and has("get", "read") and "http" in url.group(1):
        return ToolCall("http_get", {"url": url.group(1).rstrip(".,")})
    if has("search", "google", "look up"):
        query = re.sub(r".*?\b(?:search|google|look up)\b(?:\s+the\s+web)?(?:\s+for)?\s*",
                       "", t, flags=re.I).strip().strip("\"'")
        if query:
            return ToolCall("web_search", {"query": query})
    bash = re.search(r"(?:run|execute)\s+(?:the\s+)?(?:shell\s+)?command\s+(.+)$", t, re.I)
    if bash:
        return ToolCall("run_bash", {"command": bash.group(1).strip()})
    if low.startswith("$ ") or low.startswith("bash:"):
        return ToolCall("run_bash", {"command": t.split(":", 1)[-1].strip() if ":" in t else t[2:].strip()})
    if has("directory", "folder", "mkdir") and has("create", "make", "add"):
        m = re.search(r"\b(?:directory|folder)\s+(?:named\s+|called\s+)?([\w./-]+)", t, re.I)
        if m:
            return ToolCall("make_dir", {"path": m.group(1)})
    if has("append"):
        path = _first_path(t)
        if path:
            m = re.search(r"\bappend\b\s+(.*?)\s+to\s+", t, re.I)
            content = m.group(1).strip().strip("\"'") if m else (_content_phrase(t) or "")
            return ToolCall("append_file", {"path": path, "content": content})
    if has("move", "rename"):
        m = re.search(r"\b(?:move|rename)\s+([\w./-]+)\s+to\s+([\w./-]+)", t, re.I)
        if m:
            return ToolCall("move_file", {"src": m.group(1), "dst": m.group(2)})
    if has("delete", "remove"):
        path = _first_path(t)
        if path:
            return ToolCall("delete_file", {"path": path})
    if has("list", "ls") and has("file", "director", "folder"):
        return ToolCall("list_dir", {"path": "."})
    if has("create", "write", "save", "make") and _first_path(t):
        path = _first_path(t)
        return ToolCall("write_file", {"path": path, "content": _content_phrase(t) or ""})
    if has("read", "show", "cat", "open", "display", "content", "says", "say"):
        path = _first_path(t)
        if path:
            return ToolCall("read_file", {"path": path})
    return None


# Sequencing connectors that mark a compound task. We split only on explicit
# ordering ("then", "lalu", ...), never on a bare "and", so that phrases like
# "append X and Y to file" stay a single call.
_SEQUENCE_RE = re.compile(
    r"\s*(?:\band then\b|\bafter that\b|\bthen\b|\blalu\b|\bkemudian\b|\bterus\b|;)\s*",
    re.I,
)
_REFER_RE = re.compile(
    r"\b(it|them|that|the file|that file|the folder|the directory|the result)\b", re.I
)


def route_plan(task: str, registry: Optional[ToolRegistry] = None) -> list[ToolCall]:
    """Route a compound task to an ordered list of tool calls.

    Splits on explicit sequencing words (``then`` / ``lalu`` / ``;`` ...) and
    routes each part with :func:`route_task`. A later part that refers back with
    a pronoun ("read it") inherits the path from the previous step, which is what
    makes "create todo.txt with content hi then read it" work end to end.

    Parts that map to no tool (e.g. "summarize it") are simply the model's job
    afterwards, so they are skipped here rather than being forced into a call.
    """
    parts = [p.strip() for p in _SEQUENCE_RE.split(task) if p.strip()]
    if not parts:
        parts = [task.strip()]
    plan: list[ToolCall] = []
    last_path: Optional[str] = None
    for part in parts:
        segment = part
        if last_path and _REFER_RE.search(segment) and not _first_path(segment):
            segment = f"{segment} {last_path}"
        call = route_task(segment, registry)
        if call is None:
            continue
        plan.append(call)
        for key in ("path", "src", "dst"):
            if key in call.args:
                last_path = call.args[key]
    return plan


def _final_answer(runner, task: str, observation: str, temperature: float) -> str:
    prompt = (f"Task: {task}\n"
              f"Observation: {observation}\n"
              "Answer:")
    return runner.generate(prompt, max_tokens=64, temperature=temperature,
                           verbose=False).strip()


def run_agent(runner, registry: ToolRegistry, task: str, max_steps: int = 3,
              step_tokens: int = 48, temperature: float = 0.3,
              top_p: float = 0.9, verbose: bool = True, prime: bool = False,
              constrained: bool = False, constrained_args: bool = False,
              planner: str = "model", reuse_cache: bool = True,
              kv_cache_dir: Optional[str] = None,
              compact: bool = False) -> dict:
    """Drive the model through tool-use steps until it answers or runs out.

    The tiny model rarely emits a perfect call, so the loop is deliberately
    forgiving: every step is parsed with the tolerant parser, and an unparsed
    step is treated as the final answer.

    ``prime`` appends the literal cue ``TOOL:`` to each step's prompt, a
    poor-man's constrained decoding that nudges a weak model into the call
    grammar. ``constrained`` goes further and masks the logits so the tool
    *name* is forced into ``registry.names``.

    ``planner`` selects who chooses the tool:
      * ``model``  - the model (with the loop above);
      * ``route``  - a deterministic router (``route_task``) chooses and runs
        the tool, then the model only phrases the answer;
      * ``hybrid`` - try the router first, fall back to the model loop.

    ``reuse_cache`` reuses the runner's KV cache across steps via
    ``generate_continue`` so growing prompts are not re-prefilled from scratch.

    ``kv_cache_dir`` persists the *static* prompt prefix's KV cache to disk so
    a later run with the same tools skips its prefill entirely (the prefix is
    the tool preamble plus few-shot, which is the expensive, task-independent
    part). The cache is a `.nanow` container carrying the KV-cache header flag,
    so one artifact extension and one reader cover both it and the model.

    ``compact`` builds the short prefix variant (see
    :func:`build_agent_prompt_parts`), trading prompt length for a much cheaper
    one-time prefill.
    """
    transcript, calls = [], []

    # --- router planner: decide deterministically, model only phrases ---
    if planner in {"route", "hybrid"}:
        plan = route_plan(task, registry)
        if plan:
            observations = []
            for planned in plan:
                result = registry.call(planned.name, **planned.args)
                calls.append({"name": planned.name, "args": planned.args, "ok": result.ok})
                transcript.append({"role": "tool", "name": planned.name,
                                   "output": result.output})
                observations.append(f"{planned.name} -> {result.render()}")
                if verbose:
                    print(f"  🧭 router -> {planned.name}("
                          f"{json.dumps(planned.args, ensure_ascii=False)}) "
                          f"-> {result.output[:100]!r}", flush=True)
            answer = _final_answer(runner, task, "\n".join(observations), temperature)
            if verbose:
                print(f"  model answer: {answer or '(empty)'}", flush=True)
            return {"task": task, "answer": answer, "calls": calls,
                    "transcript": transcript, "planner": "route"}
        if planner == "route":
            if verbose:
                print("  🧭 router found no matching tool", flush=True)
            return {"task": task, "answer": "", "calls": [], "transcript": [],
                    "planner": "route"}

    prefix_text, task_text = build_agent_prompt_parts(registry, task, compact=compact)
    context = prefix_text + task_text
    answer = ""
    incremental = (reuse_cache and hasattr(runner, "feed")
                   and hasattr(runner, "generate_next"))
    seen: dict[tuple, str] = {}
    if incremental:
        loaded = False
        cache_path = None
        if kv_cache_dir and hasattr(runner, "save_state") and hasattr(runner, "load_state"):
            prefix_ids = runner._encode(prefix_text)
            seed = f"{getattr(runner, 'model_tag', '')}|{len(prefix_ids)}|"
            key = hashlib.sha1(seed.encode()
                               + ",".join(map(str, prefix_ids)).encode()).hexdigest()[:16]
            os.makedirs(kv_cache_dir, exist_ok=True)
            cache_path = os.path.join(kv_cache_dir, f"prefix_{key}.nanow")
            if runner.load_state(cache_path) and runner.fed_ids == prefix_ids:
                loaded = True
                if verbose:
                    print(f"  ♻️  loaded KV prefix cache ({len(prefix_ids)} tokens)", flush=True)
            else:
                runner.reset()
        if not loaded:
            runner.feed(prefix_text, add_bos=True)
            if cache_path:
                runner.save_state(cache_path)
        runner.feed(task_text)

    for step in range(max_steps):
        step_start = getattr(runner, "pos", 0)
        if verbose:
            print(f"\n  ── step {step + 1}/{max_steps} "
                  f"({len(context)} chars in context) ──", flush=True)

        cue = "\nTOOL: " if (prime or constrained or constrained_args) else "\n"
        if incremental:
            runner.feed(cue)
            if constrained_args and hasattr(runner, "generate_tool_call"):
                raw = runner.generate_tool_call(
                    registry.names, SCHEMAS, max_value_tokens=step_tokens,
                    temperature=temperature, top_p=top_p)
            elif constrained and hasattr(runner, "generate_constrained"):
                raw = runner.generate_constrained(
                    "", registry.names, max_tokens=step_tokens,
                    temperature=temperature, top_p=top_p)
            else:
                raw = runner.generate_next(max_tokens=step_tokens,
                                           temperature=temperature, top_p=top_p)
            out = raw.strip()
            text = ("TOOL: " + out) if (prime or constrained or constrained_args) else out
        elif constrained and hasattr(runner, "generate_constrained"):
            if hasattr(runner, "reset"):
                runner.reset()
            seed = context + "\nTOOL: "
            raw = runner.generate_constrained(
                seed, registry.names, max_tokens=step_tokens,
                temperature=temperature, top_p=top_p)
            out = raw.strip()
            text = ("TOOL: " + out) if out else ""
        else:
            seed = context + cue
            raw = runner.generate(seed, max_tokens=step_tokens,
                                  temperature=temperature, top_p=top_p,
                                  verbose=False)
            out = raw.strip()
            text = ("TOOL: " + out) if prime else out
        transcript.append({"role": "assistant", "text": text})
        if verbose:
            print(f"  model: {text or '(empty)'}", flush=True)

        call = parse_tool_call(text, known=set(registry.names))
        if call is None:
            answer = out
            if verbose:
                print("  (no tool call -> final answer)", flush=True)
            break

        # A greedy small model can latch onto one call and repeat it forever
        # (observed: write_file three times with identical args). Identical
        # repeats add nothing, so stop and answer with the first result.
        signature = (call.name, json.dumps(call.args, sort_keys=True, ensure_ascii=False))
        if signature in seen:
            answer = seen[signature]
            if verbose:
                print("  ↺ identical call repeated -> stopping", flush=True)
            break

        result = registry.call(call.name, **call.args)
        seen[signature] = result.render()
        calls.append({"name": call.name, "args": call.args, "ok": result.ok})
        if verbose:
            print(f"  🔧 {call.name}({json.dumps(call.args, ensure_ascii=False)}) "
                  f"-> {result.output[:120]!r}", flush=True)
        transcript.append({"role": "tool", "name": call.name, "output": result.output})

        block = f"\nObservation: {result.render()}\n"
        if incremental:
            context += block
            runner.feed(block)
        else:
            context = f"{seed}{raw}{block}"
        if verbose and incremental:
            print(f"     cache: +{getattr(runner, 'pos', 0) - step_start} new "
                  f"positions (total {getattr(runner, 'pos', 0)})", flush=True)

    return {"task": task, "answer": answer, "calls": calls,
            "transcript": transcript, "planner": planner}


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0
    verb, rest = argv[0], argv[1:]
    registry = ToolRegistry(root=os.environ.get("NANOW_WORKSPACE", "nanow_workspace"),
                            allow_bash=True)
    if verb == "list":
        result = registry.call("list_dir", path=rest[0] if rest else ".")
    elif verb in {"read", "cat"}:
        result = registry.call("read_file", path=rest[0])
    elif verb == "write":
        result = registry.call("write_file", path=rest[0], content=" ".join(rest[1:]))
    elif verb == "append":
        result = registry.call("append_file", path=rest[0], content=" ".join(rest[1:]))
    elif verb in {"rm", "delete"}:
        result = registry.call("delete_file", path=rest[0])
    elif verb == "bash":
        result = registry.call("run_bash", command=" ".join(rest))
    else:
        print(f"unknown command: {verb}")
        return 2
    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_cli(sys.argv[1:]))
