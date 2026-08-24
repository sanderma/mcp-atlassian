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
    ScopeEvaluationError,
    and_jql_clause,
    apply_jql_filter,
    assert_issues_in_scope,
    classify_issue_keys,
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
        # Keys matching the write clause; defaults to everything in scope.
        self.write_scope: set[str] | None = None
        self.fail_batches_larger_than: int | None = None
        # Keys whose presence in a query makes Jira reject it.
        self.fail_keys: set[str] = set()

    def search_issues(self, jql: str, **kwargs):
        self.queries.append(jql)
        if self.raises:
            raise self.raises
        if "issue IN (" not in jql:
            # Boundary-only probe (no keys): succeeds unless raises is set.
            return SimpleNamespace(issues=[], total=0)
        keys = {
            key.strip().strip('"')
            for key in jql.split("issue IN (", 1)[1].split(")", 1)[0].split(",")
        }
        if (
            self.fail_batches_larger_than is not None
            and len(keys) > self.fail_batches_larger_than
        ):
            raise RuntimeError("Jira rejected the batch query")
        if keys & self.fail_keys:
            raise RuntimeError("An issue with key does not exist for field 'issue'")
        pool = self.in_scope
        if " AND (" in jql and self.write_scope is not None:
            pool = {k.upper() for k in self.write_scope}
        matched = [k for k in keys if k.upper() in pool]
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
        assert fetcher.queries == ["issue IN (FOO-1) AND (team = ours)"]

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

    def test_broken_boundary_raises_evaluation_error(self):
        """An unevaluable boundary must deny, and say it is a misconfig."""
        fetcher = FakeFetcher(FakeConfig(jql_filter="nosuchfield = x"), {"FOO-1"})
        fetcher.raises = RuntimeError("Error in JQL Query")
        with pytest.raises(ScopeEvaluationError):
            issues_outside_scope(fetcher, ["FOO-1"], "read")

    def test_unknown_key_is_denied_not_reported_as_misconfig(self):
        """One bad key must deny only itself, and not blame the operator."""
        fetcher = FakeFetcher(FakeConfig(jql_filter="project = FOO"), {"FOO-1"})
        # The boundary alone evaluates fine; only queries naming the unknown
        # key fail, exactly as Jira behaves for a nonexistent issue.
        fetcher.fail_keys = {"NOPE-1"}
        assert issues_outside_scope(fetcher, ["FOO-1", "NOPE-1"], "read") == ["NOPE-1"]

    def test_truncated_result_rechecks_individually(self):
        """A clamped pagination limit must not fabricate a violation."""
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), {"FOO-1", "FOO-2"})
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

    def test_unevaluable_boundary_reports_misconfiguration(self):
        """The agent must be told this is the operator's problem."""
        fetcher = FakeFetcher(FakeConfig(jql_filter="nosuchfield = x"), set())
        fetcher.raises = RuntimeError("Error in JQL Query")
        with pytest.raises(ScopeEvaluationError) as excinfo:
            assert_issues_in_scope(fetcher, ["FOO-1"], "read", "get issue")
        message = str(excinfo.value)
        assert "misconfiguration" in message
        assert "retrying will not help" in message.lower()

    def test_denial_tells_the_agent_to_stop(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="project = FOO"), set())
        with pytest.raises(ValueError) as excinfo:
            assert_issues_in_scope(fetcher, ["FOO-1"], "read", "get issue")
        assert "fixed boundary" in str(excinfo.value)

    def test_plural_subject_grammar(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="project = FOO"), set())
        with pytest.raises(ValueError, match="FOO-1, FOO-2 are outside"):
            assert_issues_in_scope(fetcher, ["FOO-1", "FOO-2"], "read", "get issue")

    def test_read_violation_message(self):
        fetcher = FakeFetcher(FakeConfig(jql_filter="team = ours"), set())
        with pytest.raises(ValueError, match="outside the readable scope"):
            assert_issues_in_scope(fetcher, ["FOO-1"], "read", "get issue")


class TestClassifyIssueKeys:
    """Preflight classification: writable / read-only / denied."""

    def test_no_boundary_everything_visible_is_writable(self):
        fetcher = FakeFetcher(FakeConfig(), {"FOO-1"})
        assert classify_issue_keys(fetcher, ["FOO-1"]) == {"FOO-1": "writable"}

    def test_missing_issue_is_denied_not_read_only(self):
        """A nonexistent key must not be reported as merely read-only."""
        fetcher = FakeFetcher(FakeConfig(write_jql_filter="team = ours"), set())
        assert classify_issue_keys(fetcher, ["NOPE-1"]) == {"NOPE-1": "denied"}

    def test_visible_but_unmatched_is_read_only(self):
        config = FakeConfig(write_jql_filter="team = ours")
        fetcher = FakeFetcher(config, {"FOO-1"})
        # visible to the plain query, but the write clause query returns none
        fetcher.write_scope = set()
        assert classify_issue_keys(fetcher, ["FOO-1"]) == {"FOO-1": "read-only"}

    def test_mixed_verdicts_preserve_input_order(self):
        config = FakeConfig(write_jql_filter="team = ours")
        fetcher = FakeFetcher(config, {"FOO-1", "FOO-2"})
        fetcher.write_scope = {"FOO-1"}
        assert list(classify_issue_keys(fetcher, ["FOO-2", "FOO-1", "NOPE-1"])) == [
            "FOO-2",
            "FOO-1",
            "NOPE-1",
        ]

    def test_projects_filter_violation_is_denied_without_api_call(self):
        fetcher = FakeFetcher(FakeConfig(projects_filter="FOO"), {"BAR-1"})
        assert classify_issue_keys(fetcher, ["BAR-1"]) == {"BAR-1": "denied"}
        assert fetcher.queries == []

    def test_batch_failure_falls_back_to_per_key_probing(self):
        """One unknown key must not hide the valid ones."""
        fetcher = FakeFetcher(FakeConfig(), {"FOO-1"})
        fetcher.fail_batches_larger_than = 1
        verdicts = classify_issue_keys(fetcher, ["FOO-1", "NOPE-1"])
        assert verdicts == {"FOO-1": "writable", "NOPE-1": "denied"}


class TestBoundaryCannotBeEscaped:
    """Regressions for ways a caller could break out of the boundary."""

    @pytest.mark.parametrize(
        "malicious",
        [
            # Closes the wrapper and reopens it: because JQL binds AND
            # tighter than OR, the first branch would be unconstrained.
            "project = SECRET) OR (project = SECRET",
            "project = SECRET) OR (project = SECRET ORDER BY created DESC",
            "a = 1)) OR ((b = 2",
            "x = 1) OR (y = 2",
            # Stray closer alone
            "project = X)",
            # Unterminated quote
            'summary ~ "unterminated',
        ],
    )
    def test_unbalanced_jql_is_rejected(self, malicious):
        with pytest.raises(ValueError, match="Malformed JQL"):
            and_jql_clause(malicious, "labels = automation")

    @pytest.mark.parametrize(
        "jql",
        [
            "project = FOO",
            "(a = 1 OR b = 2) AND c = 3",
            'summary ~ "close ) paren"',
            'summary ~ "x ORDER BY y"',
            "project = FOO ORDER BY created DESC",
            'assignee in membersOf("group (x)")',
            "text ~ 'it\\'s'",
        ],
    )
    def test_balanced_jql_is_constrained(self, jql):
        result = and_jql_clause(jql, "labels = automation")
        assert "(labels = automation)" in result

    def test_order_by_inside_a_string_is_not_split(self):
        result = and_jql_clause('summary ~ "x ORDER BY y"', "labels = a")
        assert result == '(summary ~ "x ORDER BY y") AND (labels = a)'

    def test_order_by_in_the_boundary_is_stripped(self):
        """A boundary pasted from a saved filter usually ends in ORDER BY."""
        result = and_jql_clause("project = FOO", "labels = a ORDER BY created")
        assert result == "(project = FOO) AND (labels = a)"


class TestOpaqueIdentifiers:
    """A numeric issue id must not slip past a project allowlist."""

    def test_numeric_id_is_verified_against_jira(self):
        config = FakeConfig(projects_filter="ALLOWED")
        fetcher = FakeFetcher(config, set())
        assert issues_outside_scope(fetcher, ["10042"], "read") == ["10042"]
        assert fetcher.queries, "a numeric id must be checked server-side"

    def test_numeric_id_in_scope_is_allowed(self):
        config = FakeConfig(projects_filter="ALLOWED")
        fetcher = FakeFetcher(config, {"10042"})
        assert issues_outside_scope(fetcher, ["10042"], "read") == []

    def test_plain_keys_still_need_no_api_call(self):
        config = FakeConfig(projects_filter="ALLOWED")
        fetcher = FakeFetcher(config, set())
        assert issues_outside_scope(fetcher, ["ALLOWED-1"], "read") == []
        assert fetcher.queries == []

    def test_large_key_list_is_deduplicated_efficiently(self):
        config = FakeConfig(jql_filter="project = FOO")
        keys = [f"FOO-{i % 50}" for i in range(5000)]
        fetcher = FakeFetcher(config, {f"FOO-{i}" for i in range(50)})
        assert issues_outside_scope(fetcher, keys, "read") == []
