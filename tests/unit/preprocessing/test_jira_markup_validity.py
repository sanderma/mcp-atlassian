"""Structural validity harness for markdown_to_jira output.

Unit tests assert exact conversions for known inputs; this harness
attacks a recurring class of bug — content colliding with the
wiki-markup delimiters it gets wrapped in (macro names inside inline
code, pipes inside table-cell code spans, and the like).  Every
corpus document is converted and the *output* is linted for structural
damage: leftover Markdown syntax, merged brace delimiters, unbalanced
block macros, stray placeholder sentinels, or broken table rows.

The corpus should grow with every formatting bug report: add the
reproducing Markdown here in addition to the exact-output regression
test, so the whole harness re-checks it against every invariant.
"""

import re

import pytest

from mcp_atlassian.preprocessing.jira import JiraPreprocessor

# A Markdown link that survived conversion: brackets with no Jira alias
# pipe in them, followed by a parenthesised target on the same line.
_MARKDOWN_LINK_RE = re.compile(r"(?<!\\)\[[^\]\n|]*\]\([^)\n]*\)")

_CODE_BLOCK_RE = re.compile(
    r"\{code[^}]*\}[\s\S]*?\{code\}|\{noformat[^}]*\}[\s\S]*?\{noformat\}"
)


def strip_code_blocks(markup: str) -> str:
    """Remove {code}/{noformat} blocks — their content is free-form."""
    return _CODE_BLOCK_RE.sub("", markup)


def assert_valid_jira_markup(markup: str) -> None:
    """Assert the converted output contains no structural damage."""
    # Placeholder sentinels must never leak into output
    assert "\x00" not in markup, "extraction placeholder leaked"
    assert "\x01" not in markup, "pipe sentinel leaked"

    text = strip_code_blocks(markup)

    # Merged monospace/macro delimiters are unparseable for Jira.
    # Checked before stripping monospace spans, which would hide them.
    assert "{{{" not in text, f"merged brace delimiters: {markup!r}"
    assert "}}}" not in text, f"merged brace delimiters: {markup!r}"

    # {{...}} content is rendered literally by Jira, so anything inside
    # is fair game — drop the spans before the remaining checks.
    text = re.sub(r"\{\{.*\}\}", "", text)

    # A backtick *pair* around content outside code is an unconverted
    # code span. A lone backtick is not: CommonMark leaves it as prose
    # (an empty "``" is literal text), and Jira shows the character.
    assert not re.search(r"`[^`\n]+`", text), (
        f"Markdown code span survived conversion: {markup!r}"
    )

    # Block macros must pair up (an odd count means an unterminated
    # block that swallows the rest of the document)
    for macro in ("code", "noformat", "quote"):
        count = len(re.findall(r"\{" + macro + r"(?::[^}]*)?\}", text))
        assert count % 2 == 0, f"unbalanced {{{macro}}} markers: {markup!r}"

    for line in text.split("\n"):
        # Markdown table separator rows must not survive
        assert not re.fullmatch(r"\|[\s:|-]*-[\s:|-]*\|?", line.strip()), (
            f"Markdown table separator survived: {line!r}"
        )
        # Markdown links/images must be converted. The pattern is
        # deliberately whole-link: a bare "](" also occurs where a Jira
        # link "[url]" is followed by author text starting with "(", and
        # an escaped "\\]" is prose CommonMark left alone.
        assert not _MARKDOWN_LINK_RE.search(line), f"Markdown link survived: {line!r}"
        # A newline inside a table row splits it; every table line must
        # start and end with a pipe
        if line.startswith("|"):
            assert line.rstrip().endswith("|"), f"broken table row: {line!r}"


CORPUS = [
    pytest.param(
        "# Release 2024-01\n\n"
        "Deploy `my_service` with `--dry-run`. See [docs](https://x.test/a_b).\n\n"
        "| Component | Status |\n|:---|---:|\n| api-gateway | done |\n\n"
        "> quoted line one\n> quoted line two\n\n"
        "1. First\n  1. Nested\n2. Second\n    - Mixed\n\n"
        "```python\nprint('x')\n```\n\n---\n\n**bold** *it* ~~del~~",
        id="kitchen-sink",
    ),
    pytest.param(
        "Jira macros: `{panel}`, `{code:go}`, `{{.Values.x}}` en `fn() {ok} end`",
        id="issue-1-braces-in-inline-code",
    ),
    pytest.param(
        "| Status | Criteria |\n|---|---|\n"
        "| (?) | `[text|url]` wordt letterlijk weergegeven (niet als link) |",
        id="issue-2-pipe-in-table-code-span",
    ),
    pytest.param(
        "Writing about Jira itself:\n\n"
        "```\n{code:java}\nSystem.out.println();\n{code}\n```\n\n"
        "and `{quote}` and `{color:red}` inline.",
        id="jira-markup-as-content",
    ),
    pytest.param(
        "| a | b |\n|---|---|\n"
        "| `x|y` | [doc](https://x.test/p?q=1&r=2) |\n"
        "| line1<br>line2 | a\\|b |",
        id="table-cell-hazards",
    ),
    pytest.param(
        "![see this! now, ok](https://x.test/i.png) and [a|b](https://x.test) "
        "and <https://x.test/auto>",
        id="image-and-link-hazards",
    ),
    pytest.param(
        "snake_case_name well-known 2024-01-15 2*3*4 C++ x^2 ~5ms a|b {json}",
        id="special-chars-in-prose",
    ),
    pytest.param(
        "* First level\n** Second level\n*# Mixed\n\n[~jsmith] and PROJ-123-45",
        id="jira-syntax-passthrough",
    ),
    pytest.param(
        "## Heading with `code` and **bold**\n\n"
        "Para with [link](https://x.test) then:\n\n"
        "- [ ] task one\n- [x] task two\n\n"
        "Setext\n======\n\nUnder\n-----",
        id="headings-and-tasks",
    ),
    pytest.param(
        "````\nouter fence with ``` inside\n````\n\nText `` code with ` tick `` end",
        id="nested-backticks",
    ),
]


class TestMarkdownToJiraStructuralValidity:
    @pytest.fixture
    def preprocessor(self):
        return JiraPreprocessor(base_url="https://example.atlassian.net")

    @pytest.mark.parametrize("markdown", CORPUS)
    def test_output_is_structurally_valid(self, preprocessor, markdown):
        assert_valid_jira_markup(preprocessor.markdown_to_jira(markdown))

    @pytest.mark.parametrize("markdown", CORPUS)
    def test_round_trip_output_is_structurally_valid(self, preprocessor, markdown):
        """jira -> markdown -> jira must also stay structurally valid."""
        jira = preprocessor.markdown_to_jira(markdown)
        back = preprocessor.jira_to_markdown(jira)
        assert_valid_jira_markup(preprocessor.markdown_to_jira(back))
