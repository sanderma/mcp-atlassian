"""Read/write scope boundaries against a live Jira DC.

Unit tests drive the scope logic with a fake searcher; these prove the JQL
it builds means what we think it means to a real Jira — that a write-scope
violation is refused while the same issue stays readable, that searches are
constrained, and that unknown issues fail closed.

Run with a Jira DC instance up (see tests/e2e/docker/README.md):

    uv run pytest tests/e2e/test_scope_dc.py --dc-e2e -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
import requests

from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.scope import issues_outside_scope
from tests.e2e.conftest import DCInstanceInfo, _check_dc_health

pytestmark = pytest.mark.dc_e2e

PROJECT_KEY = "E2E"


@pytest.fixture(scope="module")
def dc_jira() -> DCInstanceInfo:
    info = DCInstanceInfo()
    if not _check_dc_health(info.jira_url):
        pytest.skip(f"Jira DC not reachable at {info.jira_url}")
    # Same reason as conftest's dc_instance: the SSRF pinning adapter
    # only trusts hosts named in the environment, so without this the
    # file passes when run beside the rest of the suite and fails with
    # "localhost resolves to non-global address" when run on its own.
    os.environ.setdefault("JIRA_URL", info.jira_url)
    return info


@pytest.fixture(scope="module")
def scope_issues(dc_jira: DCInstanceInfo) -> Iterator[dict[str, str]]:
    """Two issues distinguishable by summary: one 'ours', one 'theirs'."""
    session = requests.Session()
    session.trust_env = False
    session.auth = (dc_jira.admin_username, dc_jira.admin_password)
    marker = uuid.uuid4().hex[:8]
    created: dict[str, str] = {}
    try:
        for side in ("ours", "theirs"):
            response = session.post(
                f"{dc_jira.jira_url}/rest/api/2/issue",
                json={
                    "fields": {
                        "project": {"key": PROJECT_KEY},
                        "summary": f"scope-{marker} team-{side}",
                        "issuetype": {"name": "Task"},
                    }
                },
                timeout=30,
            )
            assert response.status_code == 201, response.text[:300]
            created[side] = response.json()["key"]
        created["marker"] = marker
        yield created
    finally:
        for side in ("ours", "theirs"):
            if side in created:
                session.delete(
                    f"{dc_jira.jira_url}/rest/api/2/issue/{created[side]}",
                    timeout=30,
                )


def make_fetcher(dc_jira: DCInstanceInfo, **overrides) -> JiraFetcher:
    """A fetcher against the DC instance with scope overrides applied."""
    from mcp_atlassian.jira.config import JiraConfig

    config = JiraConfig(
        url=dc_jira.jira_url,
        auth_type="basic",
        username=dc_jira.admin_username,
        api_token=dc_jira.admin_password,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return JiraFetcher(config=config)


def test_no_boundary_allows_everything(dc_jira, scope_issues):
    fetcher = make_fetcher(dc_jira)
    keys = [scope_issues["ours"], scope_issues["theirs"]]
    assert issues_outside_scope(fetcher, keys, "read") == []
    assert issues_outside_scope(fetcher, keys, "write") == []


def test_write_boundary_makes_unmatched_issues_read_only(dc_jira, scope_issues):
    """The headline case: read everything, write only 'our' issues."""
    marker = scope_issues["marker"]
    fetcher = make_fetcher(
        dc_jira, write_jql_filter=f'summary ~ "scope-{marker} team-ours"'
    )
    ours, theirs = scope_issues["ours"], scope_issues["theirs"]

    # Both remain readable
    assert issues_outside_scope(fetcher, [ours, theirs], "read") == []
    # Only 'ours' is writable
    assert issues_outside_scope(fetcher, [ours], "write") == []
    assert issues_outside_scope(fetcher, [theirs], "write") == [theirs]
    # A mixed batch flags exactly the offender
    assert issues_outside_scope(fetcher, [ours, theirs], "write") == [theirs]


def test_read_boundary_hides_unmatched_issues(dc_jira, scope_issues):
    marker = scope_issues["marker"]
    fetcher = make_fetcher(dc_jira, jql_filter=f'summary ~ "scope-{marker} team-ours"')
    ours, theirs = scope_issues["ours"], scope_issues["theirs"]

    assert issues_outside_scope(fetcher, [ours], "read") == []
    assert issues_outside_scope(fetcher, [theirs], "read") == [theirs]


def test_read_boundary_constrains_search(dc_jira, scope_issues):
    """A search cannot return issues outside the read boundary."""
    marker = scope_issues["marker"]
    fetcher = make_fetcher(dc_jira, jql_filter=f'summary ~ "scope-{marker} team-ours"')
    result = fetcher.search_issues(f"project = {PROJECT_KEY}", limit=50)
    keys = {issue.key for issue in result.issues}
    assert scope_issues["ours"] in keys
    assert scope_issues["theirs"] not in keys


def test_search_cannot_be_widened_by_caller_jql(dc_jira, scope_issues):
    """An OR in the caller's query must not escape the boundary."""
    marker = scope_issues["marker"]
    fetcher = make_fetcher(dc_jira, jql_filter=f'summary ~ "scope-{marker} team-ours"')
    theirs = scope_issues["theirs"]
    result = fetcher.search_issues(
        f"project = {PROJECT_KEY} OR issue = {theirs}", limit=50
    )
    assert theirs not in {issue.key for issue in result.issues}


def test_caller_cannot_escape_boundary_with_parens(dc_jira, scope_issues):
    """The boundary must survive a query that closes its own wrapper.

    JQL binds AND tighter than OR, so `X) OR (Y` would otherwise leave the
    first branch unconstrained and return the whole instance.
    """
    marker = scope_issues["marker"]
    fetcher = make_fetcher(dc_jira, jql_filter=f'summary ~ "scope-{marker} team-ours"')
    with pytest.raises(Exception) as excinfo:
        fetcher.search_issues(
            f"project = {PROJECT_KEY}) OR (project = {PROJECT_KEY}", limit=50
        )
    assert "unbalanced" in str(excinfo.value).lower()


def test_numeric_issue_id_is_scope_checked(dc_jira, scope_issues):
    """A numeric id has no project prefix, so it must be verified in Jira."""
    marker = scope_issues["marker"]
    fetcher = make_fetcher(dc_jira, jql_filter=f'summary ~ "scope-{marker} team-ours"')
    session = requests.Session()
    session.trust_env = False
    session.auth = (dc_jira.admin_username, dc_jira.admin_password)
    theirs_id = session.get(
        f"{dc_jira.jira_url}/rest/api/2/issue/{scope_issues['theirs']}?fields=key",
        timeout=30,
    ).json()["id"]
    assert issues_outside_scope(fetcher, [str(theirs_id)], "read") == [str(theirs_id)]


def test_unknown_issue_fails_closed(dc_jira):
    fetcher = make_fetcher(dc_jira, jql_filter=f"project = {PROJECT_KEY}")
    missing = f"{PROJECT_KEY}-999999"
    assert issues_outside_scope(fetcher, [missing], "read") == [missing]


def test_projects_filter_is_checked_without_api_call(dc_jira, scope_issues):
    fetcher = make_fetcher(dc_jira, projects_filter="SOMEOTHERPROJECT")
    ours = scope_issues["ours"]
    assert issues_outside_scope(fetcher, [ours], "read") == [ours]


def test_boundary_sort_reaches_jira_and_orders_results(dc_jira, scope_issues):
    """A sort on the read boundary has to be valid JQL where it lands.

    It cannot live inside the "(clause)" the boundary is ANDed in as, so
    it is carried separately and appended. This proves the composed query
    is one Jira accepts, and that the order actually takes effect.
    """
    fetcher = make_fetcher(
        dc_jira,
        jql_filter=f"project = {PROJECT_KEY}",
        jql_filter_order_by="ORDER BY key DESC",
    )
    descending = [issue.key for issue in fetcher.search_issues("", limit=50).issues]
    assert descending, "the boundary should match the seeded issues"
    assert descending == sorted(descending, reverse=True)

    # The agent's own sort wins: the boundary bounds what is visible, not
    # the order it is presented in.
    ascending = [
        issue.key
        for issue in fetcher.search_issues("ORDER BY key ASC", limit=50).issues
    ]
    assert ascending == sorted(ascending)


def test_boundary_sort_survives_a_caller_query(dc_jira, scope_issues):
    fetcher = make_fetcher(
        dc_jira,
        jql_filter=f"project = {PROJECT_KEY}",
        jql_filter_order_by="ORDER BY created DESC",
    )
    result = fetcher.search_issues(
        f"summary ~ {scope_issues['ours'].split('-')[0]}", limit=50
    )
    assert result is not None  # the composed query is valid JQL to Jira


def test_a_sort_only_boundary_restricts_nothing(dc_jira, scope_issues):
    """ "JIRA_JQL_FILTER=ORDER BY ..." is a house default, not a limit."""
    fetcher = make_fetcher(dc_jira, jql_filter=None, jql_filter_order_by="ORDER BY key")
    keys = [scope_issues["ours"], scope_issues["theirs"]]
    assert issues_outside_scope(fetcher, keys, "read") == []
