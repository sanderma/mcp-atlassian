"""Every Jira tool taking an issue key must enforce the configured scope.

The scope boundary lives on the tool decorator, so a new tool that forgets
``@enforce_issue_scope`` would silently bypass JIRA_JQL_FILTER /
JIRA_WRITE_JQL_FILTER. This test walks the registered tools and fails the
build when a tool accepting an issue key is neither decorated nor listed as
a deliberate exemption.
"""

import inspect

import pytest

from mcp_atlassian.servers.jira import jira_mcp
from mcp_atlassian.servers.scope import SCOPE_MARKER

# Parameter names that carry an issue key (or id) into a tool.
ISSUE_KEY_PARAMS = {
    "issue_key",
    "issue_keys",
    "issue_ids_or_keys",
    "inward_issue_key",
    "outward_issue_key",
    "epic_key",
    "parent_issue_key",
    "link_id",
}

# Parameters that read like an issue identifier but are not one, so the
# heuristic below does not flag them.
NOT_ISSUE_IDENTIFIERS = {
    "issue_type",
    "issue_type_id",
    "issue_type_name",
}

# Tools that take an issue-key parameter but cannot be scope-checked on it,
# each with the reason. Keep this list short and justified.
EXEMPT: dict[str, str] = {
    # Reports the scope verdict for keys instead of acting on the issues:
    # it returns only writable/read-only/denied, never issue content, and
    # applies the boundary itself via classify_issue_keys.
    "get_scope": "diagnostic tool; classifies keys rather than accessing them",
}


def _unwrap_params(func) -> set[str]:
    """Parameter names of a tool function, seeing through decorators."""
    target = inspect.unwrap(func)
    return set(inspect.signature(target).parameters)


def _scope_marker(func) -> dict | None:
    """The scope metadata attached by ``enforce_issue_scope``, if any."""
    current = func
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        marker = getattr(current, SCOPE_MARKER, None)
        if marker is not None:
            return marker
        current = getattr(current, "__wrapped__", None)
    return None


async def _jira_tools() -> list:
    return await jira_mcp.list_tools()


@pytest.mark.anyio
async def test_every_issue_key_tool_enforces_scope():
    tools = await _jira_tools()
    missing = []
    for tool in tools:
        func = tool.fn
        name = tool.name
        key_params = _unwrap_params(func) & ISSUE_KEY_PARAMS
        if not key_params:
            continue
        short_name = getattr(func, "__name__", name)
        if short_name in EXEMPT:
            continue
        if _scope_marker(func) is None:
            missing.append(f"{short_name} (params: {sorted(key_params)})")

    assert not missing, (
        "Jira tools accepting an issue key without @enforce_issue_scope:\n  "
        + "\n  ".join(missing)
        + "\nAdd the decorator, or add an entry to EXEMPT with a reason."
    )


@pytest.mark.anyio
async def test_write_tools_require_write_scope():
    """A write tool must check the write boundary, not just the read one."""
    tools = await _jira_tools()
    wrong = []
    for tool in tools:
        func = tool.fn
        name = tool.name
        if "write" not in (tool.tags or set()):
            continue
        if not (_unwrap_params(func) & ISSUE_KEY_PARAMS):
            continue
        short_name = getattr(func, "__name__", name)
        if short_name in EXEMPT:
            continue
        marker = _scope_marker(func)
        if marker is None or marker.get("access") != "write":
            wrong.append(short_name)

    assert not wrong, "Write tools not enforcing the write scope: " + ", ".join(
        sorted(wrong)
    )


@pytest.mark.anyio
async def test_decorator_params_exist_in_signature():
    """A decorator naming a parameter the tool does not have is inert.

    ``@enforce_issue_scope("issue_key")`` on a tool whose parameter is
    ``issue_keys`` collects nothing, enforces nothing, and would still
    look decorated — so the names are checked against the signature.
    """
    tools = await _jira_tools()
    broken = []
    for tool in tools:
        marker = _scope_marker(tool.fn)
        if marker is None:
            continue
        params = _unwrap_params(tool.fn)
        missing = [p for p in marker.get("params", ()) if p not in params]
        if missing:
            short_name = getattr(tool.fn, "__name__", tool.name)
            broken.append(f"{short_name}: {missing} not in {sorted(params)}")

    assert not broken, "Scope decorators naming unknown parameters:\n  " + "\n  ".join(
        broken
    )


@pytest.mark.anyio
async def test_key_param_names_are_current():
    """Catch a tool that names its issue key something new.

    ISSUE_KEY_PARAMS is a hand-maintained list, so a tool introducing
    e.g. ``parent_key`` would silently escape the coverage check. Flag
    any parameter that looks issue-key-shaped but is not in the list.
    """
    tools = await _jira_tools()
    suspicious = []
    for tool in tools:
        for param in _unwrap_params(tool.fn):
            if param in ISSUE_KEY_PARAMS or param in NOT_ISSUE_IDENTIFIERS:
                continue
            lowered = param.lower()
            looks_like_key = (
                ("issue" in lowered and ("key" in lowered or "id" in lowered))
                or lowered.endswith("_issue")
                or lowered in {"key", "keys"}
            )
            if looks_like_key:
                short_name = getattr(tool.fn, "__name__", tool.name)
                suspicious.append(f"{short_name}.{param}")

    assert not suspicious, (
        "Parameters that look like issue keys but are not covered by "
        "ISSUE_KEY_PARAMS:\n  " + "\n  ".join(sorted(set(suspicious)))
    )


@pytest.mark.anyio
async def test_exempt_tools_still_exist():
    """Keep the exemption list from rotting as tools are renamed."""
    tools = await _jira_tools()
    known = {getattr(tool.fn, "__name__", tool.name) for tool in tools}
    stale = [name for name in EXEMPT if name not in known]
    assert not stale, f"EXEMPT lists tools that no longer exist: {stale}"
