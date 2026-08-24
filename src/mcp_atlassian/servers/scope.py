"""Tool-layer enforcement of the configured Jira issue scope.

Search results are constrained inside ``search_issues`` itself, but a tool
that addresses an issue directly by key never issues a JQL query, so the
boundary has to be applied where the key enters the server: the tool call.

``enforce_issue_scope`` names the parameters carrying issue keys and the
access the tool needs. Every Jira tool taking an issue key is expected to
carry it — ``tests/unit/servers/test_jira_scope_coverage.py`` fails the
build for any that does not, so a new tool cannot silently skip the check.
"""

import inspect
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from fastmcp import Context
from fastmcp.exceptions import ToolError

from mcp_atlassian.jira.scope import (
    Access,
    assert_issues_in_scope,
    scope_is_configured,
)

from .dependencies import get_jira_fetcher

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

# Marks a tool as scope-checked, and records what it checks, so the
# coverage test can inspect a decorated tool without calling it.
SCOPE_MARKER = "__mcp_atlassian_issue_scope__"


def _lifespan_jira_config(ctx: Context) -> Any | None:
    """The server's Jira config from the lifespan context, if available."""
    try:
        lifespan_ctx = ctx.request_context.lifespan_context
    except Exception:  # noqa: BLE001 - no request context (direct call/tests)
        return None
    app_ctx = (
        lifespan_ctx.get("app_lifespan_context")
        if isinstance(lifespan_ctx, dict)
        else None
    )
    return getattr(app_ctx, "full_jira_config", None)


def _collect_keys(value: Any) -> list[str]:
    """Flatten a parameter value into a list of issue keys."""
    if value is None:
        return []
    if isinstance(value, str):
        # Some tools accept a comma-separated string of keys.
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list | tuple | set):
        keys: list[str] = []
        for item in value:
            keys.extend(_collect_keys(item))
        return keys
    return [str(value)]


def enforce_issue_scope(*params: str, access: Access = "read") -> Callable[[F], F]:
    """Check the named issue-key parameters against the configured scope.

    Args:
        *params: Names of the tool parameters carrying issue keys. A
            parameter may hold a single key, a comma-separated string, or a
            list of keys; missing/None parameters are skipped.
        access: ``"read"`` requires the issue to be within the read
            boundary; ``"write"`` additionally requires the write boundary.

    Returns:
        A decorator that raises ``ValueError`` before the tool body runs
        when any key is out of scope.
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        action = func.__name__.replace("_", " ")

        @wraps(func)
        async def wrapper(ctx: Context, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(ctx, *args, **kwargs)
            keys: list[str] = []
            for param in params:
                keys.extend(_collect_keys(bound.arguments.get(param)))

            # Resolving the fetcher is only worth it when a boundary is
            # actually configured; with none set this decorator is inert.
            if keys and scope_is_configured(_lifespan_jira_config(ctx)):
                fetcher = await get_jira_fetcher(ctx)
                try:
                    assert_issues_in_scope(fetcher, keys, access, action)
                except ValueError as exc:
                    # Surface the boundary to the caller instead of letting
                    # FastMCP mask it as a generic tool failure.
                    raise ToolError(str(exc)) from exc

            return await func(ctx, *args, **kwargs)

        setattr(wrapper, SCOPE_MARKER, {"params": params, "access": access})
        return wrapper  # type: ignore[return-value]

    return decorator


def enforce_link_scope(
    param: str = "link_id", access: Access = "write"
) -> Callable[[F], F]:
    """Scope-check the two issues an issue link connects.

    A link is addressed by id, not by issue key, so the endpoints have to
    be resolved before the boundary can be applied. Removing a link mutates
    both issues, so both must be in scope. Fails closed: if the link cannot
    be resolved, the call is refused.

    Args:
        param: Name of the tool parameter holding the link id.
        access: Boundary to require of both endpoints.
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        action = func.__name__.replace("_", " ")

        @wraps(func)
        async def wrapper(ctx: Context, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind_partial(ctx, *args, **kwargs)
            link_id = bound.arguments.get(param)

            if link_id and scope_is_configured(_lifespan_jira_config(ctx)):
                fetcher = await get_jira_fetcher(ctx)
                try:
                    keys = fetcher.issue_keys_for_link(str(link_id))
                except Exception as exc:  # noqa: BLE001 - fail closed
                    logger.warning(
                        "Could not resolve issue link %s for scope check: %s",
                        link_id,
                        exc,
                    )
                    raise ToolError(
                        f"Cannot {action}: issue link {link_id} could not be "
                        "resolved, so its issues cannot be checked against the "
                        "configured scope."
                    ) from exc
                try:
                    assert_issues_in_scope(fetcher, keys, access, action)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc

            return await func(ctx, *args, **kwargs)

        setattr(wrapper, SCOPE_MARKER, {"params": (param,), "access": access})
        return wrapper  # type: ignore[return-value]

    return decorator
