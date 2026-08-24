"""Real-renderer validation of markdown -> Jira wiki markup conversion.

The unit suite asserts what the converter *emits*; these tests assert
what Jira actually *renders* from it, using the wiki renderer of a live
Jira DC instance as the oracle (``/rest/api/1.0/render`` — the endpoint
the issue-preview UI uses).  This is what catches wrong assumptions
about Jira's notoriously weird parsing: an earlier fix for macro names
in inline code (space-padding braces in ``{{...}}``) passed every unit
test and still rendered garbage; the entity encoding shipped instead
was chosen by probing this renderer.

Run with a Jira DC instance up (see tests/e2e/docker/README.md):

    uv run pytest tests/e2e/test_markup_rendering_dc.py --dc-e2e -v

Every conversion-affecting change should extend CORPUS with the
Markdown it touches.  Each case checks browser-visible text (entity
decoding included, exactly what a user sees), raw-HTML fragments, and
generic invariants (no double-encoded entities, no Jira error spans,
no leaked sentinels, no stray backslash escapes).
"""

from __future__ import annotations

import os
import re

import pytest
import requests
from bs4 import BeautifulSoup

from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.config import JiraConfig
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
) -> tuple[str, str]:
    """Convert Markdown; return (converted markup, HTML Jira renders)."""
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
    return markup, response.text


# Case tuple: (id, markdown, visible-must, visible-must-not,
#              html-must, html-must-not[, flags])
# flags: "allow_backtick" — backticks are expected visible content
#        "allow_entity"   — "&#..." is expected visible content
Case = tuple  # noqa: N816 - readability alias

CORPUS: list[Case] = [
    # --- headings ---
    ("h1-h6", "# H1\n\n###### H6", ["H1", "H6"], [], ["<h1", "<h6"], []),
    ("setext", "Big\n===\n\nSmall\n---", ["Big", "Small"], [], ["<h1", "<h2"], []),
    ("heading-with-code", "## Use `run_all` now", ["run_all"], [], ["<h2", "<tt>"], []),
    (
        "heading-with-link",
        "## See [docs](https://x.test)",
        ["docs"],
        [],
        ["<h2", "href"],
        [],
    ),
    # --- emphasis ---
    ("bold", "some **bold** text", ["bold"], [], ["<b>bold</b>"], []),
    ("italic", "some *it* text", ["it"], [], ["<em>it</em>"], []),
    ("underscore-italic", "some _it_ text", ["it"], [], ["<em>it</em>"], []),
    ("bold-italic", "***both*** here", ["both"], [], [], []),
    (
        "nested-emph",
        "**bold *ital* inside**",
        ["bold", "ital", "inside"],
        [],
        ["<b>", "<em>"],
        [],
    ),
    (
        "emph-punct",
        "(**b**) and (_i_)!",
        ["b", "i"],
        [],
        ["<b>b</b>", "<em>i</em>"],
        [],
    ),
    (
        "midword-underscore",
        "snake_case_name stays",
        ["snake_case_name"],
        [],
        [],
        ["<em>"],
    ),
    ("midword-star", "2*3*4 = 24", ["2*3*4"], [], [], ["<b>", "<em>"]),
    ("intraword-strong", "2**3**4 = x", ["2**3**4"], [], [], ["<b>"]),
    (
        "escaped-emph",
        r"\*not bold\* and \_not it\_",
        ["*not bold*", "_not it_"],
        [],
        [],
        ["<b>", "<em>"],
    ),
    ("strike", "is ~~gone~~ now", ["gone"], [], ["<del>gone</del>"], []),
    (
        "hyphens",
        "well-known 2024-01-15 co-op",
        ["well-known", "2024-01-15", "co-op"],
        [],
        [],
        ["<del>"],
    ),
    (
        "plus-caret-tilde",
        "C++ and x^2 and ~5ms",
        ["C++", "x^2", "~5ms"],
        [],
        [],
        ["<sup>", "<sub>", "<ins>"],
    ),
    # --- inline code ---
    ("code-simple", "run `ls -la` now", ["ls -la"], [], ["<tt>"], []),
    (
        "code-braces",
        "names `{panel}` and `{code:go}`",
        ["{panel}", "{code:go}"],
        [],
        ["<tt>"],
        ['class="panel"', "<pre"],
    ),
    (
        "code-emph-chars",
        "`*a* _b_ -c- +d+ ^e^ ~f~`",
        ["*a* _b_ -c- +d+ ^e^ ~f~"],
        [],
        ["<tt>"],
        ["<b>", "<em>", "<del>", "<ins>", "<sup>", "<sub>"],
    ),
    (
        "code-link-image",
        "`[a|b]` and `!x.png!`",
        ["[a|b]", "!x.png!"],
        [],
        ["<tt>"],
        ["<img"],
    ),
    (
        "code-amp",
        "`a && b` and `&#42;`",
        ["a && b", "&#42;"],
        [],
        ["<tt>"],
        [],
        "allow_entity",
    ),
    ("code-question", "`x ?? y` here", ["x ?? y"], [], ["<tt>"], ["<cite>"]),
    ("code-unicode", "`naïve_café` ok", ["naïve_café"], [], ["<tt>"], []),
    # --- code blocks ---
    ("fence-lang", "```python\nprint('hi')\n```", ["print"], [], ["code-python"], []),
    ("fence-nolang", "```\nplain text\n```", ["plain text"], [], ["<pre"], []),
    (
        "fence-mapped-lang",
        "```typescript\nlet x = 1;\n```",
        ["let x = 1;"],
        [],
        ["code-javascript"],
        [],
    ),
    ("fence-unknown-lang", "```zig\nvar x = 1;\n```", ["var x = 1;"], [], [], []),
    ("fence-hash", "```bash\n# comment\necho hi\n```", ["# comment"], [], [], ["<h1"]),
    (
        "fence-markdown-inside",
        "```\n# heading\n**bold** [link](url)\n```",
        ["# heading", "**bold** [link](url)"],
        [],
        [],
        ["<h1", "<b>"],
        "allow_backtick",
    ),
    (
        "fence-about-jira",
        "```\n{code:java} and {panel} and {{mono}}\n```",
        ["{code:java} and {panel} and {{mono}}"],
        [],
        [],
        ["code-java", 'class="panel"'],
    ),
    (
        "indented-code",
        "para:\n\n    indented code line\n\nafter",
        ["indented code line"],
        [],
        ["<pre"],
        [],
    ),
    # --- lists ---
    (
        "ul-markers",
        "- a\n- b\n\ntext\n\n* c\n* d\n\ntext\n\n+ e",
        ["a", "b", "c", "d", "e"],
        [],
        ["<ul"],
        [],
    ),
    ("ol", "1. one\n2. two\n3. three", ["one", "two", "three"], [], ["<ol"], []),
    ("ol-paren", "1) one\n2) two", ["one", "two"], [], ["<ol"], []),
    ("nested-2sp", "1. a\n  1. b\n2. c", ["a", "b", "c"], [], ["<ol"], []),
    ("nested-4sp", "- a\n    - b\n        - c", ["a", "b", "c"], [], ["<ul"], []),
    (
        "mixed-nest",
        "1. num\n   - bullet\n2. num2",
        ["num", "bullet", "num2"],
        [],
        ["<ol", "<ul"],
        [],
    ),
    (
        "deep-nest",
        "- 1\n  - 2\n    - 3\n      - 4\n        - 5",
        ["1", "2", "3", "4", "5"],
        [],
        [],
        [],
    ),
    (
        "task-list",
        "- [ ] open item\n- [x] done item",
        ["open item", "done item", "[x]"],
        [],
        [],
        ['class="error"'],
    ),
    (
        "list-inline-fmt",
        "- **bold** item with `code`\n- [link](https://x.test)",
        ["bold", "code", "link"],
        [],
        ["<b>", "<tt>", "href"],
        [],
    ),
    (
        "list-item-code-block",
        "1. step:\n   ```\n   cmd --run\n   ```\n2. done",
        ["cmd --run", "step:", "done"],
        [],
        ["<pre"],
        [],
    ),
    (
        "list-multiline-item",
        "- first line\n  continued line\n- second",
        ["first line", "continued line", "second"],
        [],
        [],
        [],
    ),
    # --- blockquotes ---
    ("bq-single", "> one line", ["one line"], [], ["<blockquote>"], []),
    (
        "bq-multi",
        "> line one\n> line two",
        ["line one", "line two"],
        [],
        ["<blockquote>"],
        [],
    ),
    (
        "bq-multipara",
        "> para one\n>\n> para two",
        ["para one", "para two"],
        [],
        ["<blockquote>"],
        [],
    ),
    (
        "bq-with-code",
        "> note:\n> ```\n> x = 1\n> ```",
        ["note:", "x = 1"],
        [],
        ["<blockquote>"],
        [],
    ),
    ("bq-with-list", "> - a\n> - b", ["a", "b"], [], ["<blockquote>"], []),
    (
        "bq-formatted",
        "> **important** and `code`",
        ["important", "code"],
        [],
        ["<blockquote>", "<b>", "<tt>"],
        [],
    ),
    # --- links ---
    (
        "link-basic",
        "[text](https://example.com)",
        ["text"],
        [],
        ['href="https://example.com"'],
        [],
    ),
    ("link-title", '[t](https://x.test "the title")', ["t"], [], ["href"], []),
    (
        "link-underscore-url",
        "[doc](https://x.test/a_b_c)",
        ["doc"],
        [],
        ["a_b_c"],
        ["<em>"],
    ),
    (
        "link-parens-url",
        "[wiki](https://en.wikipedia.org/wiki/A_(b))",
        ["wiki"],
        [],
        ["href"],
        ['class="error"'],
    ),
    (
        "autolink",
        "<https://example.com/x>",
        ["https://example.com/x"],
        [],
        ["href"],
        [],
    ),
    ("mailto", "<mailto:a@b.test>", ["a@b.test"], [], [], ['class="error"']),
    (
        "reference-link",
        "[ref text][1]\n\n[1]: https://x.test/ref",
        ["ref text"],
        [],
        ['href="https://x.test/ref"'],
        [],
    ),
    (
        "link-formatted-text",
        "[**bold** link](https://x.test)",
        ["bold link"],
        [],
        ["href"],
        [],
    ),
    ("link-pipe", "[a|b](https://x.test)", ["a|b"], [], ["href"], []),
    (
        "emph-in-link-text",
        "[a *b* c](https://x.test/u_v)",
        ["a", "c"],
        [],
        ['href="https://x.test/u_v"'],
        [],
    ),
    (
        "bare-url",
        "see https://example.com/path today",
        ["https://example.com/path"],
        [],
        [],
        [],
    ),
    (
        "bare-url-underscore",
        "see https://x.test/a_b_c today",
        ["https://x.test/a_b_c"],
        [],
        ['href="https://x.test/a_b_c"'],
        [],
    ),
    (
        "bare-url-braces",
        "see https://x.test/a{c} today",
        ["https://x.test/a%7Bc%7D"],
        [],
        [],
        [],
    ),
    # --- images ---
    (
        "image-alt",
        "![diagram](https://x.test/i.png)",
        [],
        [],
        ['alt="diagram"', "<img"],
        [],
    ),
    ("image-noalt", "![](https://x.test/i.png)", [], [], ["<img"], []),
    (
        "image-in-link",
        "[![alt](https://x.test/i.png)](https://x.test)",
        [],
        [],
        ["<img"],
        [],
    ),
    # --- tables ---
    (
        "table-basic",
        "| a | b |\n|---|---|\n| 1 | 2 |",
        ["a", "b", "1", "2"],
        [],
        ["confluenceTh", "confluenceTd"],
        [],
    ),
    (
        "table-align",
        "| l | c | r |\n|:--|:-:|--:|\n| 1 | 2 | 3 |",
        ["l", "c", "r"],
        [":-"],
        [],
        [],
    ),
    ("table-empty-cell", "| a | b |\n|---|---|\n|  | 2 |", ["2"], [], [], []),
    (
        "table-fmt-cells",
        "| h |\n|---|\n| **b** and `c` and [l](https://x.t) |",
        ["b", "c", "l"],
        [],
        ["<b>", "<tt>", "href"],
        [],
    ),
    ("table-pipe-code", "| h |\n|---|\n| `a|b` |", ["a|b"], [], ["<tt>"], []),
    ("table-escaped-pipe", "| h |\n|---|\n| a\\|b |", ["a|b"], [], [], []),
    (
        "table-many-cols",
        "|a|b|c|d|e|f|\n|-|-|-|-|-|-|\n|1|2|3|4|5|6|",
        ["1", "6"],
        [],
        [],
        [],
    ),
    (
        "table-br-cell",
        "| h |\n|---|\n| x<br>y |",
        ["x", "y"],
        [],
        ["atl-forced-newline"],
        [],
    ),
    # --- html passthrough ---
    # Jira cannot attach ^sup^/~sub~ to a word; caret notation is the
    # accepted degradation for the attached form
    (
        "html-supsub-attached",
        "E=mc<sup>2</sup>, H<sub>2</sub>O",
        ["mc^2^", "H~2~O"],
        [],
        [],
        ['class="error"'],
    ),
    ("html-supsub-spaced", "result <sup>2</sup> here", ["2"], [], ["<sup>2</sup>"], []),
    (
        "html-insdel",
        "<ins>new</ins> <del>old</del>",
        ["new", "old"],
        [],
        ["<ins>new</ins>", "<del>old</del>"],
        [],
    ),
    (
        "html-color",
        '<span style="color:red">alert</span>',
        ["alert"],
        [],
        ['<font color="red"'],
        ["{color"],
    ),
    ("html-br", "one<br>two", ["one", "two"], [], ["<br"], []),
    # --- breaks & rules ---
    ("hr-dash", "a\n\n---\n\nb", ["a", "b"], [], ["<hr"], ["<h2"]),
    ("hr-star", "a\n\n***\n\nb", ["a", "b"], [], ["<hr"], []),
    ("hard-break", "one  \ntwo", ["one", "two"], [], ["<br"], []),
    ("soft-break", "one\ntwo", ["one", "two"], [], [], []),
    # --- special characters in prose ---
    (
        "braces-prose",
        "config {json} and {{tpl}} values",
        ["config {json} and {{tpl}} values"],
        [],
        [],
        ["<tt>", 'class="error"'],
    ),
    (
        "brackets-prose",
        "array[0] and [note] here",
        ["array[0]", "[note]"],
        [],
        [],
        ['class="error"'],
    ),
    ("amp-prose", "AT&T and a && b", ["AT&T", "a && b"], [], [], []),
    ("angle-prose", "if a < b and c > d", ["a < b", "c > d"], [], [], []),
    ("backslash-prose", r"path C:\temp\new here", ["C:", "temp", "new"], [], [], []),
    (
        "unicode-prose",
        "café emoji 🚀 CJK 日本語 done",
        ["café", "🚀", "日本語"],
        [],
        [],
        [],
    ),
    ("exclaim-prose", "wow! really!? yes!", ["wow! really!? yes!"], [], [], []),
    # Emoticons: Jira eats the text and shows icons; the author's
    # literal characters must survive
    (
        "emoticons-smilies",
        "smile :) frown :( wink ;) tongue :P grin :D",
        ["smile :) frown :( wink ;) tongue :P grin :D"],
        [],
        [],
        ['class="emoticon"'],
    ),
    (
        "emoticons-symbols",
        "thumbs (y) down (n) info (i) check (/) cross (x) warn (!) q (?)",
        ["thumbs (y) down (n) info (i) check (/) cross (x) warn (!) q (?)"],
        [],
        [],
        ['class="emoticon"'],
    ),
    (
        "emoticons-toggles",
        "toggle (on) and (off) and flag (flag)",
        ["toggle (on) and (off) and flag (flag)"],
        [],
        [],
        ['class="emoticon"'],
    ),
    (
        "math-parens",
        "f(x) = y and item (i)",
        ["f(x) = y and item (i)"],
        [],
        [],
        ['class="emoticon"'],
    ),
    # Typographic dash conversion must not rewrite author text
    ("dash-runs", "range 1 -- 2 and a --- b", ["range 1 -- 2 and a --- b"], [], [], []),
    # Jira line-start tokens appearing as prose
    (
        "prose-h2-token",
        "h2. is the heading syntax",
        ["h2. is the heading syntax"],
        [],
        [],
        ["<h2"],
    ),
    (
        "prose-bq-token",
        "bq. means blockquote",
        ["bq. means blockquote"],
        [],
        [],
        ["<blockquote"],
    ),
    (
        "prose-continuation-tokens",
        "note this:\nh3. not a heading\nbq. not a quote",
        ["h3. not a heading", "bq. not a quote"],
        [],
        [],
        ["<h3", "<blockquote"],
    ),
    # --- jira passthrough ---
    ("mention", "ping [~admin] now", ["Admin"], [], ["user-hover"], []),
    ("issue-key", "see E2E-1 there", ["E2E-1"], [], [], []),
    (
        "jira-nested-list",
        "* top\n** deeper\n*# mixed",
        ["top", "deeper", "mixed"],
        [],
        [],
        [],
    ),
    # --- Markdown block openers Jira reads as prose ---
    (
        "escaped-ordered-line",
        "1\\. not a list",
        ["1. not a list"],
        [],
        [],
        ["<ol"],
    ),
    (
        "escaped-quote-line",
        "\\> not a quote",
        ["> not a quote"],
        [],
        [],
        ["<blockquote"],
    ),
    ("escaped-plus-line", "\\+ plus lead", ["+ plus lead"], [], [], ["<ul"]),
    # --- Jira-native macros ---
    (
        "macro-anchor",
        "{anchor:rel-2}\n\nRelease notes",
        ["Release notes"],
        [],
        ["<a name="],
        [],
    ),
    (
        "confluence-macro-is-literal",
        "write {note} to make a note",
        ["{note}"],
        [],
        [],
        [],
    ),
    # --- composition ---
    (
        "kitchen-sink",
        "## Rel 2024-01\n\nShip `svc_a --fast`. See [runbook](https://x.test/r_1).\n\n"
        "| K | V |\n|---|---|\n| `a|b` | **ok** |\n\n> check `{code}` docs\n\n"
        '1. do X\n  1. sub\n2. done\n\n```go\nfmt.Println("hi")\n```\n',
        ["svc_a --fast", "runbook", "a|b", "ok", "{code}", "do X", "sub", "done"],
        [],
        ["<h2", "confluenceTd", "<blockquote>", "<ol", "<pre"],
        [],
    ),
]

_LEFTOVER_ESCAPE_RE = re.compile(r"\\[*_{}\[\]|#+~^-]")


@pytest.mark.parametrize(
    "markdown, vis_yes, vis_no, html_yes, html_no, flags",
    [
        pytest.param(c[1], c[2], c[3], c[4], c[5], c[6] if len(c) > 6 else "", id=c[0])
        for c in CORPUS
    ],
)
def test_rendered_output(
    jira_render_session: tuple[requests.Session, str],
    preprocessor: JiraPreprocessor,
    markdown: str,
    vis_yes: list[str],
    vis_no: list[str],
    html_yes: list[str],
    html_no: list[str],
    flags: str,
) -> None:
    markup, html_out = render_markdown(jira_render_session, preprocessor, markdown)
    text = BeautifulSoup(html_out, "html.parser").get_text()
    detail = f"\nmarkup={markup!r}\nvisible={text!r}"

    for snippet in vis_yes:
        assert snippet in text, f"missing visible {snippet!r}{detail}"
    for snippet in vis_no:
        assert snippet not in text, f"forbidden visible {snippet!r}{detail}"
    for fragment in html_yes:
        assert fragment in html_out, f"missing html {fragment!r}{detail}"
    for fragment in html_no:
        assert fragment not in html_out, f"forbidden html {fragment!r}{detail}"

    # Generic invariants
    assert "\x00" not in markup and "\x01" not in markup, "sentinel leaked"
    if "allow_entity" not in flags and not any("&#" in s for s in vis_yes):
        assert "&#" not in text, f"double-encoded entity visible{detail}"
    if "allow_backtick" not in flags and not any("`" in s for s in vis_yes):
        assert "`" not in text, f"backtick visible{detail}"
    if 'class="error"' not in html_no:
        assert 'class="error"' not in html_out, f"Jira error span{detail}"
    assert not _LEFTOVER_ESCAPE_RE.search(text), f"visible escape leftover{detail}"


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
        html_out = rendered.json()["renderedFields"]["description"]
        text = BeautifulSoup(html_out, "html.parser").get_text()
        assert "{panel}" in text
        assert "a|b" in text
        assert 'href="https://x.test/p"' in html_out
        assert "Steps" in text
        assert "&#" not in text
    finally:
        session.delete(f"{base_url}/rest/api/2/issue/{key}", timeout=30)


# Markdown an agent plausibly writes, exercised through a full
# read-modify-write cycle rather than a single conversion.
ROUND_TRIP_CASES = [
    ("heading-text", "# Release\n\nShip **now** with *care*."),
    ("table", "| k | v |\n|---|---|\n| `a|b` | **ok** |"),
    ("list", "1. one\n2. two\n\n- a\n- b"),
    ("code", "```python\nprint('{hi}')\n```"),
    ("quote", "> check the `{code}` docs"),
    ("prose-specials", "config `{json}`, 2*3*4, f(x), a|b, AT&T"),
    ("links", "see [docs](https://x.test/a_b) and https://x.test/c_d"),
    ("mention", "ping [~admin] about PROJ-12"),
    ("image", "![the diagram](https://x.test/i.png)"),
    ("escaped-openers", "1\\. not a list\n\n\\> not a quote"),
    ("anchor", "{anchor:rel-2}\n\nRelease notes"),
]


@pytest.mark.parametrize(
    "markdown", [c[1] for c in ROUND_TRIP_CASES], ids=[c[0] for c in ROUND_TRIP_CASES]
)
def test_round_trip_renders_identically(
    jira_render_session: tuple[requests.Session, str],
    preprocessor: JiraPreprocessor,
    markdown: str,
) -> None:
    """An agent that reads, edits and writes back must not change the page.

    Converting is lossy - Jira has no Markdown - so the guarantee is
    stability: the markup after one cycle is what Jira keeps forever,
    and what a reader sees does not shift underneath them.
    """
    first = preprocessor.markdown_to_jira(markdown)
    second = preprocessor.markdown_to_jira(preprocessor.jira_to_markdown(first))
    third = preprocessor.markdown_to_jira(preprocessor.jira_to_markdown(second))
    assert second == third, f"markup drifts\n1={first!r}\n2={second!r}\n3={third!r}"

    session, base_url = jira_render_session

    def visible(markup: str) -> str:
        response = session.post(
            f"{base_url}/rest/api/1.0/render",
            json={
                "rendererType": "atlassian-wiki-renderer",
                "unrenderedMarkup": markup,
            },
            headers={"X-Atlassian-Token": "no-check"},
            timeout=30,
        )
        assert response.status_code == 200, response.text[:300]
        return BeautifulSoup(response.text, "html.parser").get_text()

    assert visible(first) == visible(second), (
        f"rendering shifts after one edit cycle\n1={first!r}\n2={second!r}"
    )


@pytest.fixture(scope="module")
def dc_fetcher() -> JiraFetcher:
    """A fetcher against the live DC, for the API-level round trip."""
    info = DCInstanceInfo()
    if not _check_dc_health(info.jira_url):
        pytest.skip(f"Jira DC not reachable at {info.jira_url}")
    os.environ.setdefault("JIRA_URL", info.jira_url)
    return JiraFetcher(
        config=JiraConfig(
            url=info.jira_url,
            auth_type="basic",
            username=info.admin_username,
            api_token=info.admin_password,
            ssl_verify=False,
        )
    )


AGENT_MARKDOWN = (
    "## Head\n\n"
    "See `cfg` and **bold** and [~admin] and 2*3*4\n\n"
    "1. run `--dry-run`\n2. check the output\n\n"
    "| key | value |\n|---|---|\n| `a|b` | [doc](https://x.test/p) |"
)


def test_every_read_path_returns_markdown(dc_fetcher: JiraFetcher) -> None:
    """get_issue, search and the post-write reads must agree.

    They used to disagree: only ``get_issue`` translated the description,
    so an agent that found an issue through ``jira_search`` got wiki
    markup and fed its own markup back through the converter on the next
    write.
    """
    created = dc_fetcher.create_issue(
        project_key="E2E",
        summary="read path parity",
        issue_type="Task",
        description=AGENT_MARKDOWN,
    )
    key = created.key
    try:
        from_get = dc_fetcher.get_issue(key, fields="description").description
        found = dc_fetcher.search_issues(
            f"key = {key}", fields=["description"], limit=1
        )
        from_search = found.issues[0].description

        assert created.description == from_get
        assert from_search == from_get
        assert "h2." not in from_get and "{{" not in from_get
        assert from_get.startswith("## Head")
    finally:
        dc_fetcher.jira.delete_issue(key)


def test_search_edit_write_leaves_the_issue_unchanged(dc_fetcher: JiraFetcher) -> None:
    """The full agent journey: find it, edit a line, write it back."""
    created = dc_fetcher.create_issue(
        project_key="E2E",
        summary="round trip through the API",
        issue_type="Task",
        description=AGENT_MARKDOWN,
    )
    key = created.key
    try:
        found = dc_fetcher.search_issues(
            f"key = {key}", fields=["description"], limit=1
        )
        description = found.issues[0].description
        edited = description.replace("check the output", "check the log")

        dc_fetcher.update_issue(key, fields={"description": edited})
        after_first = dc_fetcher.get_issue(key, fields="description").description
        assert after_first == edited

        # A second cycle changes nothing at all - not even the stored markup.
        dc_fetcher.update_issue(key, fields={"description": after_first})
        assert dc_fetcher.get_issue(key, fields="description").description == edited
    finally:
        dc_fetcher.jira.delete_issue(key)


# YAML's own nested-mapping shape: the case Jira's highlighter breaks on.
_NESTED_MAPPING = "resources:\n  limits:\n    memory: 2Gi\n  requests:\n    cpu: 500m"


@pytest.mark.parametrize(
    "language", sorted(JiraPreprocessor.VALID_JIRA_LANGUAGES | {"yaml", "yml"})
)
def test_code_block_languages_keep_their_line_breaks(
    jira_render_session: tuple[requests.Session, str],
    preprocessor: JiraPreprocessor,
    language: str,
) -> None:
    """A highlighter that eats newlines destroys the block.

    Jira's YAML highlighter swallows the newline after a line ending in
    a bare "key:", collapsing a nested mapping onto one line - and
    indentation *is* YAML's syntax. The converter drops those languages
    to plain {code}; this sweep is what keeps that list honest as Jira
    versions change.
    """
    markdown = f"```{language}\n{_NESTED_MAPPING}\n```"
    markup, html_out = render_markdown(jira_render_session, preprocessor, markdown)

    if language in JiraPreprocessor.BROKEN_JIRA_HIGHLIGHTERS:
        assert ":" not in markup.split("\n", 1)[0], (
            f"{language} still asks Jira to highlight: {markup!r}"
        )

    block = re.search(r"<pre[^>]*>([\s\S]*?)</pre>", html_out)
    assert block is not None, f"no code block rendered\nmarkup={markup!r}"
    rendered = re.sub(r"<[^>]+>", "", block.group(1))
    assert rendered.count("\n") >= _NESTED_MAPPING.count("\n") + 1, (
        f"{language} lost line breaks: {rendered!r}"
    )
