"""Preflight check for the Jira read/write scope configuration.

Answers the question an operator has before handing a configuration to an
agent: *given these filters, what exactly can the agent see and change?*

Run via ``mcp-atlassian --jira-scope-check``. It validates that the
configured JQL parses, reports how many issues fall inside each boundary,
and can classify specific issue keys as readable / writable / denied.
"""

import logging
import os
from typing import Any

logger = logging.getLogger("mcp-atlassian.scope-check")

# Issues listed as a sample of what the agent would see.
_SAMPLE_SIZE = 5


def _line(label: str, value: str) -> str:
    return f"  {label:<24} {value}"


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _describe_boundaries(config: Any) -> bool:
    """Print the configured boundaries. Returns True if any is set."""
    _print_header("Configured boundaries")
    settings = [
        ("JIRA_PROJECTS_FILTER", config.projects_filter),
        ("JIRA_JQL_FILTER", config.jql_filter),
        ("JIRA_WRITE_JQL_FILTER", config.write_jql_filter),
    ]
    any_set = False
    for name, value in settings:
        if value:
            any_set = True
            print(_line(name, str(value)))
        else:
            print(_line(name, "(not set)"))

    read_only = os.getenv("READ_ONLY_MODE", "false").lower() in ("true", "1", "yes")
    print(_line("READ_ONLY_MODE", "ENABLED" if read_only else "disabled"))
    if read_only:
        print(
            "\n  Note: READ_ONLY_MODE is on, so every write tool is disabled\n"
            "  regardless of JIRA_WRITE_JQL_FILTER."
        )
    return any_set


def _describe_effective_query(fetcher: Any) -> None:
    """Show what a plain search turns into once boundaries are applied."""
    _print_header("Effective query")
    sample = "status = Open"
    effective = fetcher._apply_read_scope(sample)
    print(f"  A search for:  {sample}")
    print(f"  is sent as:    {effective}")


def _count(fetcher: Any, jql: str) -> int | None:
    """Total issues matching ``jql`` within the read scope, or None on error."""
    try:
        result = fetcher.search_issues(jql, fields=["key"], limit=1)
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"  ERROR: {exc}")
        return None
    total = getattr(result, "total", None)
    if total is None or total < 0:
        # Cloud's v3 search does not return a total; fall back to a page count.
        try:
            page = fetcher.search_issues(jql, fields=["key"], limit=100)
            return len(getattr(page, "issues", []) or [])
        except Exception:  # noqa: BLE001
            return None
    return int(total)


def _validate_boundaries(fetcher: Any, config: Any) -> bool:
    """Run each boundary against Jira. Returns False if any query failed."""
    _print_header("Validating boundaries against Jira")
    ok = True

    visible = _count(fetcher, "")
    if visible is None:
        print(_line("read boundary", "INVALID — the query above was rejected"))
        ok = False
    else:
        print(_line("read boundary", f"OK — {visible} issue(s) visible"))

    if config.write_jql_filter:
        writable = _count(fetcher, config.write_jql_filter)
        if writable is None:
            print(_line("write boundary", "INVALID — JIRA_WRITE_JQL_FILTER rejected"))
            ok = False
        elif visible is not None:
            read_only_count = max(visible - writable, 0)
            print(
                _line(
                    "write boundary",
                    f"OK — {writable} writable, {read_only_count} read-only",
                )
            )
        else:
            print(_line("write boundary", f"OK — {writable} writable"))
    else:
        print(_line("write boundary", "(not set) — anything readable is writable"))
    return ok


def _sample_issues(fetcher: Any, config: Any) -> None:
    """List a few in-scope issues, labelled writable or read-only."""
    from mcp_atlassian.jira.scope import classify_issue_keys

    _print_header(f"Sample of visible issues (up to {_SAMPLE_SIZE})")
    try:
        result = fetcher.search_issues("", fields=["key", "summary"], limit=_SAMPLE_SIZE)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc}")
        return

    issues = getattr(result, "issues", []) or []
    if not issues:
        print("  (none — the read boundary matches no issues)")
        print("  This configuration would leave the agent unable to see anything.")
        return

    keys = [issue.key for issue in issues if getattr(issue, "key", None)]
    verdicts = classify_issue_keys(fetcher, keys)
    for issue in issues:
        key = getattr(issue, "key", "?")
        summary = (getattr(issue, "summary", "") or "")[:48]
        print(f"  {key:<14} {verdicts.get(key, '?'):<10} {summary}")


def _check_issue_keys(fetcher: Any, keys: list[str]) -> None:
    """Classify specific issue keys as writable / read-only / denied."""
    from mcp_atlassian.jira.scope import classify_issue_keys

    _print_header("Issue checks")
    explanation = {
        "writable": "visible and writable",
        "read-only": "visible, but writes are refused",
        "denied": "not visible to the agent (filtered, missing, or no access)",
    }
    for key, verdict in classify_issue_keys(fetcher, keys).items():
        print(f"  {key:<14} {verdict:<10} {explanation[verdict]}")


def run_scope_check(issue_keys: list[str] | None = None) -> int:
    """Report what the configured Jira scope allows.

    Args:
        issue_keys: Optional issue keys to classify individually.

    Returns:
        Process exit code: 0 when the configuration is usable, 1 otherwise.
    """
    from mcp_atlassian.jira import JiraFetcher
    from mcp_atlassian.jira.config import JiraConfig

    try:
        config = JiraConfig.from_env()
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        print(f"Jira is not configured: {exc}")
        return 1

    print("Jira scope preflight")
    print("=" * 20)
    print(_line("URL", config.url))
    print(_line("Auth", config.auth_type))

    any_boundary = _describe_boundaries(config)

    try:
        fetcher = JiraFetcher(config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not connect to Jira: {exc}")
        return 1

    _describe_effective_query(fetcher)

    if not any_boundary:
        print(
            "\nNo boundary is configured: the agent can read and write "
            "everything\nthis account can. Set JIRA_JQL_FILTER and/or "
            "JIRA_WRITE_JQL_FILTER to narrow it."
        )

    ok = _validate_boundaries(fetcher, config)
    _sample_issues(fetcher, config)

    if issue_keys:
        _check_issue_keys(fetcher, issue_keys)

    print()
    if ok:
        print("Result: configuration is valid.")
        return 0
    print(
        "Result: INVALID — a configured filter was rejected by Jira.\n"
        "Fix the JQL above before giving this configuration to an agent."
    )
    return 1
