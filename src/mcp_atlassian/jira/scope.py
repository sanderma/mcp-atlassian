"""Read/write scope enforcement for Jira issues.

Two operator-configured JQL boundaries narrow what the agent may touch:

``JIRA_JQL_FILTER``
    The read boundary. ANDed into every JQL query the server issues, and
    required of any issue addressed directly by key.

``JIRA_WRITE_JQL_FILTER``
    The write boundary, applied *within* the read boundary: an issue must
    match it before a write tool may modify it. Everything readable but
    unmatched is therefore read-only — e.g. ``JIRA_JQL_FILTER`` empty and
    ``JIRA_WRITE_JQL_FILTER="labels = automation"`` lets the agent read the
    whole instance but only write to issues carrying that label.

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


class ScopeEvaluationError(ValueError):
    """A configured boundary could not be evaluated (usually invalid JQL).

    Kept distinct from a genuine out-of-scope denial so the caller can tell
    the agent "this server is misconfigured" instead of "you may not touch
    that issue" — the remedy is an operator's, not the agent's. Access is
    still refused either way.
    """


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


def _scan_jql(jql: str) -> tuple[bool, int]:
    """Scan JQL outside string literals.

    Returns:
        ``(quotes_balanced, paren_depth_delta)``. A well-formed query has
        balanced quotes and a delta of 0; the index of a top-level
        ``ORDER BY`` is found separately by :func:`_split_order_by`.
    """
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(jql):
        char = jql[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                # Closes a paren the query never opened: enough on its own
                # to break out of a wrapper.
                return quote is None, depth
        index += 1
    return quote is None, depth


def _split_order_by(jql: str) -> tuple[str, str]:
    """Split a trailing top-level ORDER BY off ``jql``.

    Quote-aware, so ``summary ~ "x ORDER BY y"`` is not mistaken for a
    sort clause.

    Returns:
        ``(query, order_by)``; ``order_by`` is "" when there is none.
    """
    for match in reversed(list(_ORDER_BY_RE.finditer(jql))):
        head = jql[: match.start()]
        balanced, depth = _scan_jql(head)
        if balanced and depth == 0:
            return head, match.group(1)
    return jql, ""


def and_jql_clause(jql: str | None, clause: str) -> str:
    """AND ``clause`` into ``jql``, keeping a trailing ORDER BY valid.

    ``jql`` is wrapped in parentheses so ``clause`` constrains all of it.
    That only holds if ``jql`` is itself balanced: a query such as
    ``project = X) OR (project = Y`` would close the wrapper and, because
    JQL binds AND tighter than OR, leave its first branch unconstrained —
    a complete escape from the boundary. Unbalanced input is therefore
    rejected rather than composed.

    Args:
        jql: The query to constrain. May be empty or ORDER BY-only.
        clause: The clause to AND in (parenthesized unless it stands alone).

    Returns:
        The constrained query.

    Raises:
        ValueError: If ``jql`` has unbalanced parentheses or quotes.
    """
    # A boundary is ANDed in as "(clause)", where a sort order is invalid.
    # Configuration strips this too; belt and braces for clauses that reach
    # here from a config not built by JiraConfig.from_env().
    clause = _split_order_by(clause.strip())[0].strip()

    if not jql or not jql.strip():
        return clause
    if jql.strip().upper().startswith("ORDER BY"):
        return f"{clause} {jql}"

    balanced, depth = _scan_jql(jql)
    if not balanced or depth != 0:
        raise ValueError(
            "Malformed JQL: unbalanced "
            f"{'quotes' if not balanced else 'parentheses'}. The query "
            "cannot be safely constrained to this server's configured "
            "scope, so it was rejected."
        )

    grouped = f"({clause})"
    head, order_by = _split_order_by(jql)
    if order_by:
        return f"({head.rstrip()}) AND {grouped} {order_by}"
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
    """Constrain ``jql`` to the configured read boundary (JIRA_JQL_FILTER).

    A sort order on the boundary becomes the default sort: it is appended
    to queries that name none, and never overrides one the caller asked
    for. Operators paste JQL from a saved filter, and the sort they chose
    there is the order they expect to see results in.
    """
    jql_filter = _str_option(config, "jql_filter")
    order_by = _str_option(config, "jql_filter_order_by")
    if not jql_filter and not order_by:
        return jql
    constrained = and_jql_clause(jql, jql_filter) if jql_filter else jql
    if order_by and not _split_order_by(constrained)[1]:
        constrained = f"{constrained.rstrip()} {order_by}".strip()
    logger.info("Applied JQL read filter to query: %s", constrained)
    return constrained


def narrow_boundary(existing: str | None, addition: str | None) -> str | None:
    """Combine a boundary with an additional clause, narrowing only.

    Used for delegation: a caller may add constraints to the boundary a
    server already enforces, but the result is always the intersection,
    so no caller can widen what the operator configured.

    ``addition`` is validated the same way a caller's query is: an
    unbalanced clause could otherwise close the wrapping parentheses and
    ``OR`` its way out of the boundary.

    Args:
        existing: The configured boundary, if any.
        addition: The extra clause to intersect with it, if any.

    Returns:
        The narrowed boundary, or None when neither is set.

    Raises:
        ValueError: If ``addition`` is not a balanced JQL expression.
    """
    extra = addition.strip() if isinstance(addition, str) else None
    if not extra:
        return existing
    balanced, depth = _scan_jql(extra)
    if not balanced or depth != 0:
        raise ValueError(
            "Malformed scope narrowing clause: unbalanced "
            f"{'quotes' if not balanced else 'parentheses'}. It was "
            "rejected rather than applied."
        )
    extra = _split_order_by(extra)[0].strip()
    if not extra:
        return existing
    if not existing:
        return extra
    return f"({existing}) AND ({extra})"


def projects_clause(config: Any) -> str | None:
    """The JIRA_PROJECTS_FILTER allowlist as a JQL clause, or None."""
    projects = allowed_project_keys(config)
    if not projects:
        return None
    quoted = [quote_jql_identifier_if_needed(p) for p in projects]
    if len(quoted) == 1:
        return f"project = {quoted[0]}"
    return f"project IN ({', '.join(quoted)})"


def apply_read_scope(jql: str, config: Any) -> str:
    """Constrain ``jql`` to every configured read boundary.

    Applies the project allowlist and the read JQL filter. Used by paths
    that issue JQL directly instead of going through ``search_issues``.
    """
    clause = projects_clause(config)
    if clause:
        jql = and_jql_clause(jql, clause)
    return apply_jql_filter(jql, config)


def allowed_project_keys(config: Any) -> list[str] | None:
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
    projects = allowed_project_keys(config)
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


def _matching_keys(
    fetcher: ScopedFetcher, keys: list[str], clause: str | None
) -> set[str]:
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
        seen = {issue.key.upper() for issue in issues if getattr(issue, "key", None)}
        # A clamped pagination limit could truncate the answer and make a
        # matching issue look out of scope; re-check the stragglers singly.
        # Driven off the returned keys rather than ``total``, which Jira
        # Cloud's v3 search reports as -1 — denial is the rare path, so the
        # extra calls cost nothing in the common case.
        if len(seen) < len(chunk):
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
    seen_upper: set[str] = set()
    for key in keys:
        cleaned = (key or "").strip()
        upper = cleaned.upper()
        if cleaned and upper not in seen_upper:
            seen_upper.add(upper)
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
    # A numeric issue id carries no project prefix, so the local check above
    # cannot judge it. Verify such identifiers against Jira whenever a
    # project allowlist is configured — otherwise an id would slip past a
    # boundary that a key would have been refused by.
    opaque_ids = [k for k in remaining if not _ISSUE_KEY_RE.match(k)]
    needs_lookup = (
        bool(_str_option(config, "jql_filter"))
        or bool(write_clause)
        or (bool(allowed_project_keys(config)) and bool(opaque_ids))
    )
    if not needs_lookup:
        return outside

    try:
        matching = _matching_keys(fetcher, remaining, write_clause)
    except Exception as exc:  # noqa: BLE001 - fail closed on any lookup failure
        # Jira rejects the query for two very different reasons: the
        # boundary JQL is invalid (an operator's problem), or one of the
        # keys does not exist (an ordinary denial). Probe the boundary on
        # its own to tell them apart rather than blaming the wrong party.
        if not _boundary_is_evaluable(fetcher, write_clause):
            raise ScopeEvaluationError(str(exc)) from exc
        matching = _probe_keys(fetcher, remaining, write_clause)
    return outside + [k for k in remaining if k.upper() not in matching]


def _boundary_is_evaluable(fetcher: ScopedFetcher, clause: str | None) -> bool:
    """Whether the configured boundary itself runs without error."""
    try:
        fetcher.search_issues(clause or "", fields=["key"], limit=1)
    except Exception:  # noqa: BLE001 - the boundary is the broken part
        return False
    return True


def _probe_keys(
    fetcher: ScopedFetcher, keys: list[str], clause: str | None
) -> set[str]:
    """Check keys one at a time, treating an unresolvable key as denied.

    Used when a batch query failed for a key-specific reason (an unknown
    key makes some Jira versions reject the whole query), so that one bad
    key cannot deny the rest.
    """
    matching: set[str] = set()
    for key in keys:
        try:
            matching |= _matching_keys(fetcher, [key], clause)
        except Exception:  # noqa: BLE001 - unknown key: denied
            logger.debug("Scope probe failed for %s; treating as denied", key)
    return matching


def _visible_keys(fetcher: ScopedFetcher, keys: list[str]) -> set[str]:
    """Keys that come back from Jira within the read boundary.

    Unlike the enforcement path this never assumes: a batch query that Jira
    rejects (an unknown key makes some Jira versions fail the whole query)
    is retried key by key, so one bad key cannot hide the others.
    """
    try:
        return _matching_keys(fetcher, keys, None)
    except Exception:  # noqa: BLE001 - fall back to per-key probing
        visible: set[str] = set()
        for key in keys:
            try:
                if _matching_keys(fetcher, [key], None):
                    visible.add(key.upper())
            except Exception:  # noqa: BLE001 - unknown/invisible key
                continue
        return visible


def classify_issue_keys(fetcher: ScopedFetcher, keys: list[str]) -> dict[str, str]:
    """Classify issues as ``writable``, ``read-only`` or ``denied``.

    Reports what an agent would actually experience for each key, for
    preflight/diagnostic use — enforcement uses ``issues_outside_scope``.

    Args:
        fetcher: Jira fetcher providing ``config`` and ``search_issues``.
        keys: Issue keys to classify.

    Returns:
        Mapping of each input key to its classification.
    """
    config = fetcher.config
    result: dict[str, str] = {}

    blocked_by_projects = set(keys_outside_projects_filter(keys, config))
    candidates = [k for k in keys if k not in blocked_by_projects]
    for key in blocked_by_projects:
        result[key] = "denied"

    if not candidates:
        return {key: result.get(key, "denied") for key in keys}

    visible = _visible_keys(fetcher, candidates)  # tolerant of bad keys
    write_clause = _str_option(config, "write_jql_filter")
    writable = visible
    if write_clause:
        in_scope = [k for k in candidates if k.upper() in visible]
        writable = (
            _visible_keys_with_clause(fetcher, in_scope, write_clause)
            if in_scope
            else set()
        )

    for key in candidates:
        upper = key.upper()
        if upper not in visible:
            result[key] = "denied"
        elif upper in writable:
            result[key] = "writable"
        else:
            result[key] = "read-only"
    return {key: result[key] for key in keys}


def _visible_keys_with_clause(
    fetcher: ScopedFetcher, keys: list[str], clause: str
) -> set[str]:
    """``_visible_keys`` narrowed by an extra clause (the write boundary)."""
    try:
        return _matching_keys(fetcher, keys, clause)
    except Exception:  # noqa: BLE001 - fail closed: nothing is writable
        return set()


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
    try:
        outside = issues_outside_scope(fetcher, keys, access)
    except ScopeEvaluationError as exc:
        message = (
            f"Cannot {action}: this server's access scope could not be "
            f"evaluated ({exc}). That is a server misconfiguration, not a "
            "permission problem — the boundary JQL in JIRA_JQL_FILTER / "
            "JIRA_WRITE_JQL_FILTER is probably invalid for this Jira "
            "instance. Report it; retrying will not help."
        )
        logger.error(message)
        raise ScopeEvaluationError(message) from exc

    if not outside:
        return

    listed = ", ".join(outside)
    subject = f"{listed} is" if len(outside) == 1 else f"{listed} are"
    config = fetcher.config
    # Both branches end by telling the agent the boundary is fixed: a
    # denial that reads as transient invites pointless retries and
    # tool-shopping around the same wall.
    final = (
        "This is a fixed boundary in this server's configuration — another "
        "tool, a retry, or a different phrasing will not succeed. Report the "
        "limit instead of working around it."
    )
    if access == "write":
        write_jql = _str_option(config, "write_jql_filter")
        detail = (
            f"writable scope (JIRA_WRITE_JQL_FILTER={write_jql!r})"
            if write_jql
            else "configured scope"
        )
        message = (
            f"Cannot {action}: {subject} outside the {detail}. "
            "Such an issue may be readable but read-only, may not exist, or "
            f"may be invisible to these credentials. {final}"
        )
    else:
        read_note = (
            " A read boundary (JIRA_JQL_FILTER) is configured on this server."
            if _str_option(config, "jql_filter")
            else ""
        )
        message = (
            f"Cannot {action}: {subject} outside the readable scope "
            "configured for this server, does not exist, or is invisible to "
            f"these credentials.{read_note} {final}"
        )
    logger.warning(message)
    raise ValueError(message)
