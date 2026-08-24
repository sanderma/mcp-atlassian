"""Per-request scope narrowing can restrict, never widen.

This is the delegation guarantee: a caller proxying through this server
(a subagent, say) may add constraints to the boundary the operator
configured, but cannot escape or replace it. The server's configuration
is always an upper bound on what any caller gets.
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from mcp_atlassian.jira.config import JiraConfig
from mcp_atlassian.jira.scope import narrow_boundary
from mcp_atlassian.servers.dependencies import (
    JIRA_SCOPE_NARROW_HEADER,
    JIRA_WRITE_SCOPE_NARROW_HEADER,
    _narrowed_for_request,
)


@dataclass
class FakeRequest:
    headers: dict


def make_config(**overrides) -> JiraConfig:
    config = JiraConfig(
        url="https://jira.example.com",
        auth_type="pat",
        personal_token="t",  # noqa: S106 - test fixture
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestNarrowBoundary:
    def test_addition_alone_becomes_the_boundary(self):
        assert narrow_boundary(None, "project = PROJ") == "project = PROJ"

    def test_both_are_intersected(self):
        assert (
            narrow_boundary("labels = automation", "project = PROJ")
            == "(labels = automation) AND (project = PROJ)"
        )

    def test_empty_addition_keeps_the_boundary(self):
        assert narrow_boundary("labels = automation", "") == "labels = automation"
        assert narrow_boundary("labels = automation", None) == "labels = automation"

    def test_an_or_cannot_widen_the_boundary(self):
        """The addition is parenthesized, so OR only applies within it."""
        result = narrow_boundary("labels = automation", "a = 1 OR b = 2")
        assert result == "(labels = automation) AND (a = 1 OR b = 2)"

    @pytest.mark.parametrize(
        "escape",
        [
            "1 = 1) OR (1 = 1",
            "x = 1)",
            'summary ~ "unterminated',
        ],
    )
    def test_unbalanced_addition_is_rejected(self, escape):
        """An unbalanced clause could close the AND and OR its way out."""
        with pytest.raises(ValueError, match="Malformed scope narrowing"):
            narrow_boundary("labels = automation", escape)

    def test_order_by_in_the_addition_is_stripped(self):
        assert (
            narrow_boundary("labels = a", "project = PROJ ORDER BY created")
            == "(labels = a) AND (project = PROJ)"
        )


class TestRequestNarrowing:
    def _with_headers(self, headers: dict, config: JiraConfig) -> JiraConfig:
        with patch(
            "mcp_atlassian.servers.dependencies.get_http_request",
            return_value=FakeRequest(headers=headers),
        ):
            return _narrowed_for_request(config)

    def test_no_headers_is_a_passthrough(self):
        config = make_config(jql_filter="project = PROJ")
        assert self._with_headers({}, config) is config

    def test_header_narrows_the_read_boundary(self):
        config = make_config(jql_filter="project = PROJ")
        result = self._with_headers(
            {JIRA_SCOPE_NARROW_HEADER: "labels = automation"}, config
        )
        assert result.jql_filter == "(project = PROJ) AND (labels = automation)"

    def test_header_narrows_the_write_boundary(self):
        config = make_config(write_jql_filter="labels = automation")
        result = self._with_headers(
            {JIRA_WRITE_SCOPE_NARROW_HEADER: "assignee = currentUser()"}, config
        )
        assert result.write_jql_filter == (
            "(labels = automation) AND (assignee = currentUser())"
        )

    def test_header_cannot_replace_the_configured_boundary(self):
        """The operator's boundary survives whatever the caller sends."""
        config = make_config(jql_filter="project = PROJ")
        result = self._with_headers(
            {JIRA_SCOPE_NARROW_HEADER: "project = OTHER"}, config
        )
        assert "project = PROJ" in result.jql_filter
        assert result.jql_filter.startswith("(project = PROJ) AND")

    def test_header_sets_a_boundary_where_none_was_configured(self):
        config = make_config()
        result = self._with_headers(
            {JIRA_SCOPE_NARROW_HEADER: "project = PROJ"}, config
        )
        assert result.jql_filter == "project = PROJ"

    def test_malformed_header_is_refused(self):
        config = make_config(jql_filter="project = PROJ")
        with pytest.raises(ValueError, match=JIRA_SCOPE_NARROW_HEADER):
            self._with_headers(
                {JIRA_SCOPE_NARROW_HEADER: "1 = 1) OR (1 = 1"}, config
            )

    def test_the_shared_config_is_never_mutated(self):
        """One caller's narrowing must not leak into another's request."""
        config = make_config(jql_filter="project = PROJ")
        result = self._with_headers(
            {JIRA_SCOPE_NARROW_HEADER: "labels = automation"}, config
        )
        assert config.jql_filter == "project = PROJ"
        assert result is not config

    def test_no_http_request_is_a_passthrough(self):
        """stdio transport has no request; the config is returned as-is."""
        config = make_config(jql_filter="project = PROJ")
        with patch(
            "mcp_atlassian.servers.dependencies.get_http_request",
            side_effect=RuntimeError("no request"),
        ):
            assert _narrowed_for_request(config) is config
