"""Real-renderer validation of markdown -> Jira wiki markup conversion.

The unit suite asserts what the converter *emits*; these tests assert
what Jira actually *renders* from it, using the wiki renderer of a live
Jira DC instance as the oracle (``/rest/api/1.0/render`` — the endpoint
the issue-preview UI uses).  This is what catches wrong assumptions
about Jira's notoriously weird parsing: fork issue #1's proposed fix
(space-padding braces in ``{{...}}``) passed every unit test and still
rendered garbage; the entity encoding shipped instead was chosen by
probing this renderer.

Run with a Jira DC instance up (see tests/e2e/docker/README.md):

    uv run pytest tests/e2e/test_markup_rendering_dc.py --dc-e2e -v

Every conversion-affecting change should extend BEHAVIORS with the
Markdown it touches plus the expected/forbidden HTML fragments.
"""

from __future__ import annotations

import pytest
import requests

from mcp_atlassian.preprocessing.jira import JiraPreprocessor
from tests.e2e.conftest import DCInstanceInfo, _check_dc_health

pytestmark = pytest.mark.dc_e2e


@pytest.fixture(scope="module")
def jira_render_session() -> tuple[requests.Session, str]:
    """Session against a live Jira DC; skips when unreachable.

    Deliberately independent of the ``dc_instance`` fixture: rendering
    tests need only Jira, not Confluence.
    """
    info = DCInstanceInfo()
    if not _check_dc_health(info.jira_url):
        pytest.skip(f"Jira DC not reachable at {info.jira_url}")
    session = requests.Session()
    session.trust_env = False
    session.auth = (info.admin_username, info.admin_password)
    return session, info.jira_url


@pytest.fixture(scope="module")
def preprocessor() -> JiraPreprocessor:
    return JiraPreprocessor(base_url="http://localhost:8080")


def render_markdown(
    jira_render_session: tuple[requests.Session, str],
    preprocessor: JiraPreprocessor,
    markdown: str,
) -> str:
    """Convert Markdown and return the HTML Jira renders for it."""
    session, base_url = jira_render_session
    markup = preprocessor.markdown_to_jira(markdown)
    response = session.post(
        f"{base_url}/rest/api/1.0/render",
        json={
            "rendererType": "atlassian-wiki-renderer",
            "unrenderedMarkup": markup,
        },
        timeout=30,
    )
    assert response.status_code == 200, response.text[:500]
    return response.text


# (id, markdown, fragments that MUST appear, fragments that MUST NOT)
BEHAVIORS = [
    (
        "bold-italic",
        "some **bold** and *italic* text",
        ["<b>bold</b>", "<em>italic</em>"],
        [],
    ),
    (
        "inline-code-plain",
        "call `getStatusById` now",
        ["<tt>getStatusById</tt>"],
        [],
    ),
    (
        # Issue #1: macro names in inline code must render as literal
        # monospace text, not execute the macro or show stray braces
        "inline-code-macro-braces",
        "Jira macro namen zoals `{panel}` en `{code:go}`",
        ["<tt>&#123;panel&#125;</tt>", "<tt>&#123;code:go&#125;</tt>"],
        ['class="panel"', "<pre", "{{"],
    ),
    (
        # Emphasis must not fire inside inline code either
        "inline-code-specials",
        "run `my_var --dry-run` then `2*3*4`",
        ["<tt>my&#95;var &#45;&#45;dry&#45;run</tt>"],
        ["<em>", "<del>", "<b>"],
    ),
    (
        # Issue #2: pipe in a table cell's inline code keeps the row
        # at two cells and stays monospace
        "table-cell-code-pipe",
        "| Status | Criteria |\n|---|---|\n| x | `[text|url]` test |",
        ["<tt>&#91;text&#124;url&#93;</tt>", "confluenceTh"],
        ["<em>", "`"],
    ),
    (
        "table-cell-link",
        "| a | b |\n|---|---|\n| [doc](https://x.test/p) | y |",
        ['href="https://x.test/p"'],
        [],
    ),
    (
        "table-cell-hard-break",
        "| a | b |\n|---|---|\n| line1<br>line2 | y |",
        ["atl-forced-newline"],
        [],
    ),
    (
        "snake-case-prose",
        "the foo_bar_baz identifier stays plain",
        [],
        ["<em>"],
    ),
    (
        "hyphens-and-dates",
        "well-known values from 2024-01-15",
        ["well-known", "2024-01-15"],
        ["<del>"],
    ),
    (
        # Docs *about* Jira markup: fenced code containing {code} must
        # render as one literal block, not nested/broken code panels
        "code-block-about-jira",
        "```\nuse {code:java} blocks {code}\n```",
        ["use {code:java} blocks {code}"],
        ["code-java"],
    ),
    (
        "blockquote-multiline",
        "> line one\n> line two",
        ["<blockquote>"],
        [],
    ),
    (
        "nested-lists",
        "1. first\n  1. nested\n2. second\n    - mixed",
        ["<ol>", "<ul>"],
        [],
    ),
    (
        "horizontal-rule",
        "before\n\n---\n\nafter",
        ["<hr />"],
        ["h2"],
    ),
    (
        "link-with-pipe-text",
        "[a|b](https://x.test)",
        ['href="https://x.test"', "a&#124;b"],
        [],
    ),
    (
        "image-with-alt",
        "![diagram](https://x.test/i.png)",
        ['alt="diagram"'],
        [],
    ),
]


@pytest.mark.parametrize(
    "markdown, expected, forbidden",
    [pytest.param(m, e, f, id=i) for i, m, e, f in BEHAVIORS],
)
def test_rendered_html_matches_expectations(
    jira_render_session: tuple[requests.Session, str],
    preprocessor: JiraPreprocessor,
    markdown: str,
    expected: list[str],
    forbidden: list[str],
) -> None:
    html = render_markdown(jira_render_session, preprocessor, markdown)
    for fragment in expected:
        assert fragment in html, f"missing {fragment!r} in rendered HTML: {html}"
    for fragment in forbidden:
        assert fragment not in html, f"forbidden {fragment!r} in rendered HTML: {html}"


def test_rendered_issue_description_round_trip(
    jira_render_session: tuple[requests.Session, str],
    preprocessor: JiraPreprocessor,
) -> None:
    """Full write path: converted markup survives the issue REST API.

    Creates a real issue with a converted description and checks the
    HTML Jira serves via ``expand=renderedFields`` — the same rendering
    users see — then cleans up.
    """
    session, base_url = jira_render_session
    markdown = (
        "## Steps\n\n"
        "1. run `--dry-run`\n"
        "2. check `{panel}` docs\n\n"
        "| key | value |\n|---|---|\n| `a|b` | [doc](https://x.test/p) |"
    )
    markup = preprocessor.markdown_to_jira(markdown)

    create = session.post(
        f"{base_url}/rest/api/2/issue",
        json={
            "fields": {
                "project": {"key": "E2E"},
                "summary": "markup rendering e2e",
                "description": markup,
                "issuetype": {"name": "Task"},
            }
        },
        timeout=30,
    )
    assert create.status_code == 201, create.text[:500]
    key = create.json()["key"]
    try:
        rendered = session.get(
            f"{base_url}/rest/api/2/issue/{key}?expand=renderedFields",
            timeout=30,
        )
        assert rendered.status_code == 200
        html = rendered.json()["renderedFields"]["description"]
        assert "<tt>&#123;panel&#125;</tt>" in html
        assert "<tt>a&#124;b</tt>" in html
        assert 'href="https://x.test/p"' in html
        assert "Steps" in html
    finally:
        session.delete(f"{base_url}/rest/api/2/issue/{key}", timeout=30)
