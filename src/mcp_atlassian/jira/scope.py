"""Read/write scope enforcement for Jira issues.

Two operator-configured JQL boundaries narrow what the agent may touch:

``JIRA_JQL_FILTER``
    The read boundary. ANDed into every JQL query the server issues, and
    required of any issue addressed directly by key.

``JIRA_WRITE_JQL_FILTER``
    The write boundary, applied *within* the read boundary: an issue must
    match it before a write tool may modify it. Everything readable but
    unmatched is therefore read-only — e.g. ``JIRA_JQL_FILTER`` empty and
    ``JIRA_WRITE_JQL_FILTER="team = ours"`` lets the agent read the whole
    instance but only write to its own team's issues.

Both are configuration-only: no tool argument can widen them (a caller may
only narrow further, as with ``JIRA_PROJECTS_FILTER``).

Checks are fail-closed. An issue that does not come back from the scope
query — because it does not exist, is invisible to the credentials, or
does not match the filter — is treated as out of scope.
"""

import logging
import re
from typing import Any, Literal, Protocol

from .utils import quote_jql_identifier_if_needed

logger = logging.getLogger("mcp-jira")

Access = Literal["read", "write"]

_ORDER_BY_RE = re.compile(r"\s+(ORDER\s+BY\s+.*)$", re.IGNORECASE)

# Issue keys look like PROJ-123; anything else (a numeric issue id, for
# instance) is not matched against the project allowlist.
_ISSUE_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)-\d+$")

# Keys per scope query. Small enough to stay well under any configured
# ATLASSIAN_MAX_PAGINATION_LIMIT, so a clamped limit cannot silently
# truncate the answer.
_SCOPE_BATCH_SIZE = 20


class ScopedFetcher(Protocol):
    """The slice of ``JiraFetcher`` the scope checks depend on."""

    config: Any

    def search_issues(self, jql: str, **kwargs: Any) -> Any: ...


def and_jql_clause(jql: str | None, clause: str) -> str:
    """AND ``clause`` into ``jql``, keeping a trailing ORDER BY valid.

    Args:
        jql: The query to constrain. May be empty or ORDER BY-only.
        clause: The clause to AND in (parenthesized unless it stands alone).

    Returns:
        The constrained query.
    """
    if not jql or not jql.strip():
        return clause
    if jql.strip().upper().startswith("ORDER BY"):
        return f"{clause} {jql}"

    grouped = f"({clause})"
    order_match = _ORDER_BY_RE.search(jql)
    if order_match:
        head = jql[: order_match.start()]
        return f"({head}) AND {grouped} {order_match.group(1)}"
    return f"({jql}) AND {grouped}"


def _str_option(config: Any, name: str) -> str | None:
    """Read a string config option, treating anything else as unset.

    Test doubles and partially-built configs can expose non-string values;
    a scope boundary must never be built from one.
    """
    value = getattr(config, name, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def scope_is_configured(config: Any) -> bool:
    """Whether any issue-scope boundary is configured.

    Lets callers skip scope work entirely — including resolving a client —
    on the default configuration, where no boundary applies.
    """
    return any(
        _str_option(config, name)
        for name in ("jql_filter", "write_jql_filter", "projects_filter")
    )


def apply_jql_filter(jql: str, config: Any) -> str:
    """Constrain ``jql`` to the configured read boundary (JIRA_JQL_FILTER)."""
    jql_filter = _str_option(config, "jql_filter")
    if not jql_filter:
        return jql
    constrained = and_jql_clause(jql, jql_filter)
    logger.info("Applied JQL read filter to query: %s", constrained)
    return constrained


def _allowed_projects(config: Any) -> list[str] | None:
    """Project keys from JIRA_PROJECTS_FILTER, upper-cased, or None."""
    projects_filter = _str_option(config, "projects_filter")
    if not projects_filter:
        return None
    projects = [p.strip().upper() for p in projects_filter.split(",") if p.strip()]
    return projects or None


def keys_outside_projects_filter(keys: list[str], config: Any) -> list[str]:
    """Keys whose project prefix is not in JIRA_PROJECTS_FILTER.

    A local check — no API call. Values that are not issue keys (numeric
    issue ids, for instance) are not evaluated here; the JQL scope check
    covers those when a JQL filter is configured.
    """
    projects = _allowed_projects(config)
    if not projects:
        return []
    outside = []
    for key in keys:
        match = _ISSUE_KEY_RE.match(key)
        if match and match.group(1).upper() not in projects:
            outside.append(key)
    return outside


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _issue_in_clause(keys: list[str]) -> str:
    """Build ``issue IN (...)``; accepts issue keys and numeric issue ids."""
    quoted = ", ".join(quote_jql_identifier_if_needed(key) for key in keys)
    return f"issue IN ({quoted})"


def _matching_keys(fetcher: ScopedFetcher, keys: list[str], clause: str | None) -> set[str]:
    """Return the subset of ``keys`` visible in scope, upper-cased.

    ``search_issues`` applies the configured read boundary itself, so the
    result is always within it; ``clause`` narrows further (the write
    boundary).
    """
    found: set[str] = set()
    for chunk in _chunks(keys, _SCOPE_BATCH_SIZE):
        jql = _issue_in_clause(chunk)
        if clause:
            jql = f"{jql} AND ({clause})"
        result = fetcher.search_issues(jql, fields=["key"], limit=len(chunk))
        issues = getattr(result, "issues", []) or []
        seen = {
            issue.key.upper()
            for issue in issues
            if getattr(issue, "key", None)
        }
        # A clamped pagination limit could truncate the answer and make a
        # matching issue look out of scope; re-check the stragglers singly.
        if getattr(result, "total", len(issues)) > len(issues):
            for key in chunk:
                if key.upper() in seen:
                    continue
                single = _issue_in_clause([key])
                if clause:
                    single = f"{single} AND ({clause})"
                one = fetcher.search_issues(single, fields=["key"], limit=1)
                if getattr(one, "issues", None):
                    seen.add(key.upper())
        found |= seen
    return found


def issues_outside_scope(
    fetcher: ScopedFetcher, keys: list[str], access: Access
) -> list[str]:
    """Return the issues in ``keys`` outside the requested scope.

    Args:
        fetcher: Jira fetcher providing ``config`` and ``search_issues``.
        keys: Issue keys (or numeric issue ids) to check.
        access: ``"read"`` checks the read boundary only; ``"write"`` also
            requires the write boundary.

    Returns:
        The offending keys, in input order. Empty when no boundary applies.
    """
    unique: list[str] = []
    for key in keys:
        cleaned = (key or "").strip()
        if cleaned and cleaned.upper() not in {k.upper() for k in unique}:
            unique.append(cleaned)
    if not unique:
        return []

    config = fetcher.config
    # Cheap local check first: a project-allowlist violation needs no API call.
    outside = keys_outside_projects_filter(unique, config)
    remaining = [k for k in unique if k not in outside]
    if not remaining:
        return outside

    write_clause = (
        _str_option(config, "write_jql_filter") if access == "write" else None
    )
    if not _str_option(config, "jql_filter") and not write_clause:
        return outside

    try:
        matching = _matching_keys(fetcher, remaining, write_clause)
    except Exception as exc:  # noqa: BLE001 - fail closed on any lookup failure
        logger.warning(
            "Scope check failed for %s (access=%s); denying: %s",
            ", ".join(remaining),
            access,
            exc,
        )
        return outside + remaining
    return outside + [k for k in remaining if k.upper() not in matching]


def assert_issues_in_scope(
    fetcher: ScopedFetcher,
    keys: list[str],
    access: Access,
    action: str,
) -> None:
    """Raise ValueError if any of ``keys`` is outside the requested scope.

    Args:
        fetcher: Jira fetcher providing ``config`` and ``search_issues``.
        keys: Issue keys (or numeric issue ids) the operation touches.
        access: ``"read"`` or ``"write"``.
        action: Human-readable operation name used in the error message.
    """
    outside = issues_outside_scope(fetcher, keys, access)
    if not outside:
        return

    listed = ", ".join(outside)
    config = fetcher.config
    if access == "write":
        write_jql = _str_option(config, "write_jql_filter")
        detail = (
            f"writable scope (JIRA_WRITE_JQL_FILTER={write_jql!r})"
            if write_jql
            else "configured scope"
        )
        message = (
            f"Cannot {action}: {listed} is outside the {detail}. "
            "The issue may be readable but is not writable, may not exist, "
            "or may be invisible to these credentials."
        )
    else:
        message = (
            f"Cannot {action}: {listed} is outside the readable scope "
            "configured for this server, does not exist, or is invisible "
            "to these credentials."
        )
    logger.warning(message)
    raise ValueError(message)
