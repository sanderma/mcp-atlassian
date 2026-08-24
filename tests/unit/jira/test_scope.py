"""Tests for the Jira read/write scope boundaries.

Covers JIRA_JQL_FILTER (read) and JIRA_WRITE_JQL_FILTER (write): how they
are ANDed into queries, how issues addressed by key are checked, and that
the checks fail closed.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_atlassian.jira.scope import (
    and_jql_clause,
    apply_jql_filter,
    assert_issues_in_scope,
    issues_outside_scope,
    keys_outside_projects_filter,
    scope_is_configured,
)


@dataclass
class FakeConfig:
    projects_filter: str | None = None
    jql_filter: str | None = None
    write_jql_filter: str | None = None


class FakeFetcher:
    """Minimal fetcher: returns the keys it is told are in scope."""

    def __init__(self, config: FakeConfig, in_scope: set[str] | None = None):
        self.config = config
        self.in_scope = {k.upper() for k in (in_scope or set())}
        self.queries: list[str] = []
        self.raises: Exception | None = None
        self.truncate = False

    def search_issues(self, jql: str, **kwargs):
        self.queries.append(jql)
        if self.raises:
            raise self.raises
        keys = {
            key.strip().strip('"')
            for key in jql.split("issue IN (", 1)[1].split(")", 1)[0].split(",")
        }
        matched = [k for k in keys if k.upper() in self.in_scope]
        issues = [SimpleNamespace(key=k) for k in matched]
        if self.truncate and len(issues) > 1:
            # Simulate a clamped pagination limit: report more than returned
            return SimpleNamespace(issues=issues[:1], total=len(issues))
        return SimpleNamespace(issues=issues, total=len(issues))


class TestAndJqlClause:
    def test_empty_jql_returns_clause(self):
        assert and_jql_clause("", "team = ours") == "team = ours"
        assert and_jql_clause(None, "team = ours") == "team = ours"

    def test_plain_jql_is_grouped(self):
        assert (
            and_jql_clause("project = FOO", "team = ours")
            == "(project = FOO) AND (team = ours)"
        )

    def test_order_by_stays_at_the_end(self):
        assert (
            and_jql_clause("project = FOO ORDER BY created DESC", "team = ours")
            == "(project = FOO) AND (team = ours) ORDER BY created DESC"
        )

    def test_order_by_only_query(self):
        assert (
            and_jql_clause("ORDER BY created DESC", "team = ours")
            == "team = ours ORDER BY created DESC"
        )

    def test_clause_cannot_be_escaped_by_or(self):
        """A caller's OR must not widen the boundary."""
        result = and_jql_clause("a = 1 OR b = 2", "team = ours")
        assert result == "(a = 1 OR b = 2) AND (team = ours)"


class TestApplyJqlFilter:
    def test_no_filter_is_passthrough(self):
        assert apply_jql_filter("project = FOO", FakeConfig()) == "project = FOO"

    def test_filter_is_anded(self):
        config = FakeConfig(jql_filter="team = ours")
        assert (
            apply_jql_filter("project = FOO", config)
            == "(project = FOO) AND (team = ours)"
        )

    def test_non_string_config_value_is_ignored(self):
        """A mock/partial config must never become a boundary clause."""
        config = MagicMock()
        assert apply_jql_filter("project = FOO", config) == "project = FOO"


class TestScopeIsConfigured:
    @pytest.mark.parametrize(
        "config, expected",
        [
            (FakeConfig(), False),
            (FakeConfig(jql_filter="team = ours"), True),
            (FakeConfig(write_jql_filter="team = ours"), True),
            (FakeConfig(projects_filter="FOO"), True),
            (FakeConfig(jql_filter="   "), False),
        ],
    )
    def test_detection(self, config, expected):
        assert scope_is_configured(config) is expected


class TestProjectsFilterKeyCheck:
    def test_no_filter_allows_everything(self):
        assert keys_outside_projects_filter(["FOO-1"], FakeConfig()) == []

    def test_key_outside_allowlist_is_flagged(self):
        config = FakeConfig(projects_filter="FOO, BAR")
        assert keys_outside_projects_filter(["FOO-1", "BAZ-9"], config) == ["BAZ-9"]

    def test_case_insensitive(self):
        config = FakeConfig(projects_filter="foo")
        assert keys_outside_projects_filter(["FOO-1"], config) == []

    def test_numeric_ids_are_left_to_the_jql_check(self):
        config = FakeConfig(projects_filter="FOO")
        assert keys_outside_projects_filter(["10001"], config) == []


class TestIssuesOutsideScope:
    def test_no_boundary_means_no_api_call(self):
        fetcher = FakeFetcher(FakeConfig())
        assert issues_outside_scope(fetcher, ["FOO-1"], "read") == []
        assert fetcher.queries == []

    def test_read_boundary_allows_matching_issue(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), {"FOO-1"})
        assert issues_outside_scope(fetcher, ["FOO-1"], "read") == []

    def test_read_boundary_blocks_unmatched_issue(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), set())
        assert issues_outside_scope(fetcher, ["FOO-1"], "read") == ["FOO-1"]

    def test_write_boundary_is_anded_into_the_query(self):
        config = FakeConfig(write_jql_filter="team = ours")
        fetcher = FakeFetcher(config, {"FOO-1"})
        issues_outside_scope(fetcher, ["FOO-1"], "write")
        assert fetcher.queries == ['issue IN (FOO-1) AND (team = ours)']

    def test_read_access_ignores_the_write_boundary(self):
        """A read-only issue stays readable."""
        config = FakeConfig(write_jql_filter="team = ours")
        fetcher = FakeFetcher(config, set())
        # No read boundary configured, so reads need no check at all
        assert issues_outside_scope(fetcher, ["FOO-1"], "read") == []
        assert fetcher.queries == []

    def test_readable_but_not_writable(self):
        config = FakeConfig(jql_filter="project = FOO", write_jql_filter="team = ours")
        readable = FakeFetcher(config, {"FOO-1"})
        assert issues_outside_scope(readable, ["FOO-1"], "read") == []
        not_writable = FakeFetcher(config, set())
        assert issues_outside_scope(not_writable, ["FOO-1"], "write") == ["FOO-1"]

    def test_projects_filter_short_circuits_without_api_call(self):
        config = FakeConfig(projects_filter="FOO", jql_filter="team = ours")
        fetcher = FakeFetcher(config, set())
        assert issues_outside_scope(fetcher, ["BAR-1"], "read") == ["BAR-1"]
        assert fetcher.queries == []

    def test_duplicate_and_blank_keys_are_normalized(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="x = 1"), {"FOO-1"})
        assert issues_outside_scope(fetcher, ["FOO-1", " FOO-1 ", ""], "read") == []
        assert len(fetcher.queries) == 1

    def test_lookup_failure_fails_closed(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), {"FOO-1"})
        fetcher.raises = RuntimeError("Jira exploded")
        assert issues_outside_scope(fetcher, ["FOO-1"], "read") == ["FOO-1"]

    def test_truncated_result_rechecks_individually(self):
        """A clamped pagination limit must not fabricate a violation."""
        fetcher = FakeFetcher(
            FakeConfig(jql_filter="team = ours"), {"FOO-1", "FOO-2"}
        )
        fetcher.truncate = True
        assert issues_outside_scope(fetcher, ["FOO-1", "FOO-2"], "read") == []


class TestAssertIssuesInScope:
    def test_in_scope_is_silent(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), {"FOO-1"})
        assert_issues_in_scope(fetcher, ["FOO-1"], "read", "get issue")

    def test_write_violation_names_the_setting(self):
        config = FakeConfig(write_jql_filter="team = ours")
        fetcher = FakeFetcher(config, set())
        with pytest.raises(ValueError) as excinfo:
            assert_issues_in_scope(fetcher, ["FOO-1"], "write", "add comment")
        message = str(excinfo.value)
        assert "FOO-1" in message
        assert "add comment" in message
        assert "JIRA_WRITE_JQL_FILTER" in message

    def test_read_violation_message(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), set())
        with pytest.raises(ValueError, match="outside the readable scope"):
            assert_issues_in_scope(fetcher, ["FOO-1"], "read", "get issue")
