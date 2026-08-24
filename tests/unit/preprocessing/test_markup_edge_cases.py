"""Systematic edge-case coverage for the Jira DC markup conversion.

The other suites in this package assert *known* conversions. This one
sweeps a matrix instead: every hazardous token placed in every structural
context, and every shape a code block can take. It exists because the
bugs that reach production are never in the cases someone thought to
write down — they are in the cell that nobody filled in.

Three layers of assertion, cheapest first:

1. :class:`TestHazardMatrix` — token x context, checking the invariants
   that must hold everywhere (no crash, no leaked sentinel, structurally
   valid output, stable under a read-modify-write cycle).
2. :class:`TestCodeBlockShapes` and :class:`TestEscapingRules` — exact
   output for the cases where "valid" is not enough and the precise
   markup matters.
3. ``tests/e2e/test_markup_rendering_dc.py`` renders the same matrix
   through a live Jira and asserts on what a reader actually sees.
"""

import pytest

from mcp_atlassian.preprocessing.jira import JiraPreprocessor

from .test_jira_markup_validity import assert_valid_jira_markup

# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------

# Tokens that collide with Jira's markup, Markdown's markup, or both.
HAZARD_TOKENS = {
    "star": "*",
    "underscore": "_",
    "hyphen": "-",
    "plus": "+",
    "caret": "^",
    "tilde": "~",
    "bracket-open": "[",
    "bracket-close": "]",
    "brace-open": "{",
    "brace-close": "}",
    "pipe": "|",
    "bang": "!",
    "question": "?",
    "paren-open": "(",
    "paren-close": ")",
    "ampersand": "&",
    "hash": "#",
    "colon": ":",
    "semicolon": ";",
    "dot": ".",
    "backslash": "\\",
    "percent": "%",
    "at": "@",
    "lt": "<",
    "gt": ">",
    "quote-double": '"',
    "quote-single": "'",
    "double-brace": "{{",
    "double-brace-pair": "{{}}",
    "double-star": "**",
    "double-underscore": "__",
    "double-hyphen": "--",
    "triple-hyphen": "---",
    "double-tilde": "~~",
    "double-question": "??",
    "heading-token": "h1.",
    "heading-token-2": "h2.",
    "quote-token": "bq.",
    "emoticon-smile": ":)",
    "emoticon-tick": "(y)",
    "emoticon-cross": "(x)",
    "macro-code": "{code}",
    "macro-panel": "{panel}",
    "macro-color": "{color:red}",
    "entity": "&#42;",
    "entity-amp": "&amp;",
    "html-tag": "<b>",
    "image-markup": "!img.png!",
    "pipe-word": "a|b",
    "star-word": "a*b",
    "underscore-word": "a_b",
    "brace-word": "x{y}z",
    "call": "f(x)",
    "math": "2*3*4",
    "cpp": "C++",
    "power": "x^2",
    "windows-path": "C:\\logs\\",
    "regex": "\\d{2,3}",
    "jira-link": "[text|url]",
}

# Every structural position a token can land in.
HAZARD_CONTEXTS = {
    "prose-mid": "before {t} after",
    "prose-start": "{t} after",
    "prose-end": "before {t}",
    "prose-alone": "{t}",
    "prose-continuation": "first line\n{t} after",
    "heading": "## Head {t} end",
    "heading-setext": "Head {t} end\n===",
    "bullet": "- item {t} end",
    "bullet-nested": "- a\n  - b {t} end",
    "ordered": "1. item {t} end",
    "task": "- [ ] item {t} end",
    "quote": "> quoted {t} end",
    "quote-nested": "> outer\n>\n> > inner {t} end",
    "table-cell": "| a | b |\n|---|---|\n| {t} | y |",
    "table-header": "| h {t} | b |\n|---|---|\n| x | y |",
    "bold": "**bold {t} end**",
    "italic": "*it {t} end*",
    "strike": "~~gone {t} end~~",
    "link-text": "[text {t} end](https://x.test/p)",
    "image-alt": "![alt {t} end](https://x.test/i.png)",
    "inline-code": "code `{t}` end",
    "inline-code-in-table": "| a |\n|---|\n| `{t}` |",
    "inline-code-in-heading": "## Head `{t}` end",
    "inline-code-in-bold": "**bold `{t}` end**",
    "fenced-code": "```\n{t}\n```",
    "fenced-code-lang": "```python\n{t}\n```",
    "fenced-code-in-list": "- a\n\n  ```\n  {t}\n  ```\n",
    "fenced-code-in-quote": "> a\n>\n> ```\n> {t}\n> ```\n",
    "indented-code": "para\n\n    {t}\n",
}

MATRIX = [
    pytest.param(context, token, id=f"{cname}-{tname}")
    for cname, context in HAZARD_CONTEXTS.items()
    for tname, token in HAZARD_TOKENS.items()
]


@pytest.fixture
def preprocessor() -> JiraPreprocessor:
    return JiraPreprocessor(base_url="https://jira.example.com")


class TestHazardMatrix:
    """Invariants that must hold for every token in every context.

    Exact output is asserted elsewhere; what is checked here is that no
    combination can crash the converter, leak an internal placeholder,
    produce structurally broken markup, or drift on re-edit. A single
    "> " line used to raise, and because ``markdown_to_jira`` swallows
    the error and returns its input, one stray character sent an entire
    document to Jira as unconverted Markdown.

    All four invariants share one parametrization so each of the ~1900
    combinations is converted once.
    """

    @pytest.mark.parametrize("context, token", MATRIX)
    def test_conversion_invariants(self, preprocessor, context, token):
        markdown = context.format(t=token)

        # 1. Never raises. The private entry point is used deliberately:
        #    the public one catches everything and returns its input,
        #    which would hide exactly the failure being tested for.
        markup = preprocessor._markdown_to_jira(markdown)

        # 2. No internal placeholder survives, no broken structure.
        assert_valid_jira_markup(markup)

        # 3. Reading it back never raises either.
        back = preprocessor.jira_to_markdown(markup)

        # 4. Read-modify-write converges after the first write.
        second = preprocessor.markdown_to_jira(back)
        third = preprocessor.markdown_to_jira(preprocessor.jira_to_markdown(second))
        assert second == third, f"drifts: {markup!r} -> {second!r} -> {third!r}"

    @pytest.mark.parametrize("context, token", MATRIX)
    def test_round_trip_is_stable(self, preprocessor, context, token):
        """Read-modify-write must converge after the first write.

        Kept separate from the invariants above because it is the
        expensive half: two more passes through the wiki parser.
        """
        first = preprocessor.markdown_to_jira(context.format(t=token))
        second = preprocessor.markdown_to_jira(preprocessor.jira_to_markdown(first))
        third = preprocessor.markdown_to_jira(preprocessor.jira_to_markdown(second))
        assert second == third, f"drifts: {first!r} -> {second!r} -> {third!r}"


class TestCodeBlockShapes:
    """Every shape a code block takes, and what it must convert to.

    Code is the hardest construct to get right: its content is arbitrary
    text that must reach Jira untouched, but the delimiters Jira uses
    ({code}, {noformat}, {{...}}) are themselves things people write
    *about* inside code blocks.
    """

    def test_plain_fence(self, preprocessor):
        assert preprocessor.markdown_to_jira("```\nx = 1\n```") == (
            "{code}\nx = 1\n{code}"
        )

    def test_fence_with_language(self, preprocessor):
        assert preprocessor.markdown_to_jira("```python\nx = 1\n```") == (
            "{code:python}\nx = 1\n{code}"
        )

    def test_tilde_fence(self, preprocessor):
        assert preprocessor.markdown_to_jira("~~~\nx = 1\n~~~") == (
            "{code}\nx = 1\n{code}"
        )

    def test_indented_code_block(self, preprocessor):
        assert preprocessor.markdown_to_jira("para\n\n    x = 1\n") == (
            "para\n\n{code}\nx = 1\n{code}"
        )

    def test_empty_fence(self, preprocessor):
        assert preprocessor.markdown_to_jira("```\n```") == "{code}\n{code}"

    def test_unknown_language_drops_to_plain(self, preprocessor):
        assert preprocessor.markdown_to_jira("```brainfuck\n+++\n```") == (
            "{code}\n+++\n{code}"
        )

    def test_language_alias_is_mapped(self, preprocessor):
        assert preprocessor.markdown_to_jira("```ts\nlet x = 1\n```") == (
            "{code:javascript}\nlet x = 1\n{code}"
        )

    def test_content_mentioning_code_macro_uses_noformat(self, preprocessor):
        """{code} inside a {code} block ends it early - Jira offers no
        escape, so the block has to switch delimiters."""
        assert preprocessor.markdown_to_jira("```\n{code:java}\nx\n{code}\n```") == (
            "{noformat}\n{code:java}\nx\n{code}\n{noformat}"
        )

    def test_content_mentioning_both_delimiters_keeps_code(self, preprocessor):
        """Neither delimiter is safe; {code} is the least-bad choice and
        the degradation is documented rather than silently mangled."""
        result = preprocessor.markdown_to_jira(
            "```\n{code} and {noformat}\n```"
        )
        assert result.startswith("{code}")
        assert "{code} and {noformat}" in result

    def test_content_is_never_escaped(self, preprocessor):
        """Everything inside a block is literal to Jira, so nothing in
        it may be rewritten - an escape there is visible corruption."""
        body = "a*b _c_ [d] {e} |f| !g! ~h~ ^i^ &j; \\k \\* 2*3*4"
        result = preprocessor.markdown_to_jira(f"```\n{body}\n```")
        assert result == "{code}\n" + body + "\n{code}"

    def test_fence_longer_than_inner_backticks(self, preprocessor):
        assert preprocessor.markdown_to_jira("````\nhas ``` inside\n````") == (
            "{code}\nhas ``` inside\n{code}"
        )

    def test_fence_inside_list_item(self, preprocessor):
        result = preprocessor.markdown_to_jira("- step\n\n  ```\n  make\n  ```\n")
        assert "{code}\nmake\n{code}" in result
        assert result.startswith("* step")

    def test_fence_inside_quote(self, preprocessor):
        result = preprocessor.markdown_to_jira("> note\n>\n> ```\n> make\n> ```\n")
        assert "{code}\nmake\n{code}" in result

    def test_consecutive_fences_stay_separate(self, preprocessor):
        result = preprocessor.markdown_to_jira("```\na\n```\n\n```\nb\n```")
        assert result.count("{code}") == 4

    def test_info_string_with_attributes(self, preprocessor):
        """```js {highlight=1} - only the first word is the language,
        and the rest must not leak into the macro parameters."""
        assert preprocessor.markdown_to_jira("```js {highlight=1}\nx\n```") == (
            "{code:js}\nx\n{code}"
        )

    def test_read_back_gives_a_fence(self, preprocessor):
        assert preprocessor.jira_to_markdown("{code:python}\nx = 1\n{code}") == (
            "```python\nx = 1\n```"
        )

    def test_read_back_sizes_the_fence_past_inner_backticks(self, preprocessor):
        result = preprocessor.jira_to_markdown("{code}\nhas ``` inside\n{code}")
        assert result.startswith("````")
        assert "has ``` inside" in result


class TestInlineCodeShapes:
    """Inline code is a *text effect* in Jira, not a literal span.

    Macros, emphasis, links, images and citations all still execute
    inside ``{{...}}``, and backslash escapes render as backslashes, so
    the content has to be entity-encoded rather than escaped.
    """

    def test_plain_span(self, preprocessor):
        assert preprocessor.markdown_to_jira("use `run()` now") == "use {{run()}} now"

    def test_macro_name_is_encoded(self, preprocessor):
        assert preprocessor.markdown_to_jira("`{panel}`") == "{{&#123;panel&#125;}}"

    def test_doubled_braces_do_not_merge_delimiters(self, preprocessor):
        """`{{x}}` naively becomes {{{{x}}}}, which Jira cannot parse."""
        result = preprocessor.markdown_to_jira("`{{x}}`")
        assert result == "{{&#123;&#123;x&#125;&#125;}}"
        assert "{{{" not in result

    def test_pipe_is_encoded(self, preprocessor):
        assert preprocessor.markdown_to_jira("`a|b`") == "{{a&#124;b}}"

    def test_span_containing_backticks(self, preprocessor):
        """A backtick is ordinary text to Jira; only Markdown needs the
        wider delimiter, and it must not survive into the markup."""
        assert preprocessor.markdown_to_jira("`` a ` b ``") == "{{a ` b}}"

    def test_span_in_table_cell_keeps_the_row_intact(self, preprocessor):
        result = preprocessor.markdown_to_jira("| a | `x|y` |\n|---|---|\n| 1 | 2 |")
        assert result.splitlines()[0] == "||a||{{x&#124;y}}||"

    def test_span_in_heading(self, preprocessor):
        assert preprocessor.markdown_to_jira("## Use `run()`") == "h2. Use {{run()}}"

    def test_span_in_link_text(self, preprocessor):
        assert preprocessor.markdown_to_jira("[see `x`](https://e.test)") == (
            "[see {{x}}|https://e.test]"
        )

    def test_read_back_decodes_the_entities(self, preprocessor):
        assert preprocessor.jira_to_markdown("{{&#123;panel&#125;}}") == "`{panel}`"


class TestEscapingRules:
    """Exact escaping, per context, as verified against the renderer.

    The mechanism differs by context because Jira's does: backslashes
    work in prose, entities are required inside monospace and at line
    starts, and percent-encoding is the only option inside a URL.
    """

    def test_braces_in_prose_use_entities(self, preprocessor):
        assert preprocessor.markdown_to_jira("config {json} here") == (
            "config &#123;json&#125; here"
        )

    def test_emphasis_chars_use_backslashes(self, preprocessor):
        assert preprocessor.markdown_to_jira("snake_case_name") == "snake\\_case\\_name"

    def test_boundary_effect_chars_stay_readable(self, preprocessor):
        """Jira only applies -, +, ^, ~ at word boundaries, so escaping
        them inside a word would add visible noise for nothing."""
        assert preprocessor.markdown_to_jira("well-known 2024-01-15") == (
            "well-known 2024-01-15"
        )

    def test_line_start_tokens_are_neutralized(self, preprocessor):
        assert preprocessor.markdown_to_jira("h2. is the heading syntax") == (
            "h2&#46; is the heading syntax"
        )

    def test_line_start_tokens_in_a_table_cell(self, preprocessor):
        """A cell opens a line: "h1." there renders as a heading and
        swallows the cell's content."""
        result = preprocessor.markdown_to_jira("| a |\n|---|\n| h1. text |")
        assert result.endswith("|h1&#46; text|")

    def test_literal_backslash_is_an_entity(self, preprocessor):
        """Jira eats a backslash in front of !#%()*+-?@[]^_{|}~, and a
        trailing one escapes a table cell's closing delimiter. Which
        applies depends on the rendered neighbour, so every literal
        backslash is encoded; &#92; displays as "\\" everywhere."""
        assert preprocessor.markdown_to_jira("regex \\\\* here") == "regex &#92;* here"
        assert preprocessor.markdown_to_jira("C:\\\\Users\\\\test") == (
            "C:&#92;Users&#92;test"
        )

    def test_trailing_backslash_in_a_cell_does_not_merge_cells(self, preprocessor):
        result = preprocessor.markdown_to_jira(
            "| a | b |\n|---|---|\n| C:\\\\logs\\\\ | y |"
        )
        assert result.endswith("|C:&#92;logs&#92;|y|")

    def test_bare_urls_are_not_escaped(self, preprocessor):
        result = preprocessor.markdown_to_jira("see https://x.test/a_b_c now")
        assert "https://x.test/a_b_c" in result

    def test_braces_in_a_bare_url_are_percent_encoded(self, preprocessor):
        """Jira macro-parses braces even inside a URL, and does not
        decode entities there."""
        assert "%7Bc%7D" in preprocessor.markdown_to_jira("see https://x.test/a{c} now")

    def test_pipe_in_link_text_uses_an_entity(self, preprocessor):
        assert preprocessor.markdown_to_jira("[a|b](https://x.test)") == (
            "[a&#124;b|https://x.test]"
        )

    def test_image_markup_in_link_text_is_neutralized(self, preprocessor):
        """Jira runs image markup inside a link alias, replacing the
        author's words with a broken-image icon."""
        assert preprocessor.markdown_to_jira("[see !x.png! here](https://x.test)") == (
            "[see &#33;x.png&#33; here|https://x.test]"
        )

    def test_macros_in_image_alt_are_encoded(self, preprocessor):
        """A macro in the alt terminates the image markup entirely."""
        assert preprocessor.markdown_to_jira("![a {code} b](x.png)") == (
            "!x.png|alt=a &#123;code&#125; b!"
        )

    def test_emoticons_keep_the_author_text(self, preprocessor):
        assert preprocessor.markdown_to_jira("smile :) and f(x)") == (
            "smile &#58;) and f&#40;x)"
        )


class TestDegenerateInput:
    """Inputs that are almost nothing, and must still not break.

    ``markdown_to_jira`` catches every exception and returns its input
    unchanged, so a crash here does not surface as an error - it
    silently sends raw Markdown to Jira. These use the private entry
    point so a regression actually fails.
    """

    @pytest.mark.parametrize(
        "markdown",
        [
            "",
            " ",
            "\n",
            "\n\n\n",
            ">",
            "> ",
            ">\n>\n>",
            "-",
            "- ",
            "1.",
            "#",
            "###### ",
            "|",
            "||",
            "|  |\n|---|",
            "```",
            "```\n",
            "~~~",
            "`",
            "``",
            "[",
            "[]",
            "[]()",
            "![]()",
            "*",
            "**",
            "***",
            "_",
            "---",
            "\\",
            "{",
            "{}",
            "{{}}",
            "\x00",
            "\x01",
        ],
        ids=repr,
    )
    def test_degenerate_input_converts(self, preprocessor, markdown):
        markup = preprocessor._markdown_to_jira(markdown)
        assert "\x00" not in markup or "\x00" in markdown
        preprocessor.jira_to_markdown(markup)

    def test_a_bare_quote_line_does_not_abort_the_document(self, preprocessor):
        """The regression this class exists for: an empty blockquote
        raised, and the caught exception sent the whole document to
        Jira as unconverted Markdown."""
        markdown = "# Title\n\n>\n\n| a |\n|---|\n| 1 |"
        result = preprocessor.markdown_to_jira(markdown)
        assert result.startswith("h1. Title")
        assert "||a||" in result
        assert "#" not in result.split("\n")[0]


class TestEmptyListItems:
    """An item with no content still has to keep its neighbours apart.

    The newline after a list item comes from the block *inside* it, so
    an empty item produced none: "- a\n- \n- c" collapsed to
    "* a\n* * c", which Jira reads as two items, the second nested.
    """

    def test_empty_item_between_two_others(self, preprocessor):
        assert preprocessor.markdown_to_jira("- a\n- \n- c") == "* a\n* \n* c"

    def test_all_items_empty(self, preprocessor):
        assert preprocessor.markdown_to_jira("1. \n2. \n3. ") == "# \n# \n# "

    def test_adjacent_lists_are_joined(self, preprocessor):
        """A blank line ends a list in Jira, so two adjacent lists have
        no representation - and CommonMark splits "- \\n- x" into exactly
        that. Rendering the split faithfully put a blank line inside what
        Jira reads as one list, and the markup then flipped between the
        two forms on every edit."""
        assert preprocessor.markdown_to_jira("- \n- |\n- ") == (
            "* \n* &#124;\n* "
        )
        assert preprocessor.markdown_to_jira("- a\n\n- b") == "* a\n* b"

    def test_a_list_separated_by_real_content_stays_separate(self, preprocessor):
        assert preprocessor.markdown_to_jira("- a\n\npara\n\n- b") == (
            "* a\n\npara\n\n* b"
        )

    def test_empty_parent_with_a_nested_list(self, preprocessor):
        """The nested list starts its own lines; without the break the
        markers run together and read as one deeper level."""
        assert preprocessor.markdown_to_jira("- \n  - x") == "*\n** x"

    def test_a_list_of_empty_items_round_trips(self, preprocessor):
        markup = "# \n# x"
        for _ in range(3):
            markup = preprocessor.markdown_to_jira(
                preprocessor.jira_to_markdown(markup)
            )
        assert markup == "# \n# x"

    @pytest.mark.parametrize(
        "markdown", ["> 1.", "- )", "- .", "- >\n  - 1", ">\n\n- a"]
    )
    def test_degenerate_empty_blocks_converge_without_growing(
        self, preprocessor, markdown
    ):
        """The shapes that need a second cycle to settle.

        Jira writes a single empty item as a bare "#", which the wiki
        parser has no list to attach to and hands back as a heading; and
        CommonMark reads a lone "." or ")" as an ordered-list marker, so
        "- )" parses as a list nested in a list; and an empty blockquote
        needs the "{quote}" block form, which cannot sit inside a list
        item. None of them is worth contorting the converter for. What matters is that they *settle*
        - the failure mode this pins against is unbounded growth.
        """
        cycles = []
        markup = preprocessor.markdown_to_jira(markdown)
        for _ in range(5):
            cycles.append(markup)
            markup = preprocessor.markdown_to_jira(
                preprocessor.jira_to_markdown(markup)
            )
        assert cycles[-1] == cycles[-2] == markup, f"never settles: {cycles}"
        assert len(markup) <= len(cycles[0]) + 8, f"grows: {cycles}"


class TestLinkAndImageShapes:
    """Links and images carry two payloads that are markup themselves."""

    def test_link(self, preprocessor):
        assert preprocessor.markdown_to_jira("[t](https://x.test/p)") == (
            "[t|https://x.test/p]"
        )

    def test_autolink(self, preprocessor):
        assert preprocessor.markdown_to_jira("<https://x.test/p>") == (
            "[https://x.test/p]"
        )

    def test_brackets_in_the_target_are_percent_encoded(self, preprocessor):
        """The stock renderer backslash-escapes them; Jira strips the
        backslash, the read path keeps it, and the next write encodes it
        - so the URL grew a "%5C" on every edit cycle."""
        assert preprocessor.markdown_to_jira("<https://x.test/a[b]c>") == (
            "[https://x.test/a%5Bb%5Dc]"
        )

    def test_empty_target_is_not_a_link(self, preprocessor):
        """"[text|]" is not link markup, and the stray pipe splits a
        table cell."""
        assert preprocessor.markdown_to_jira("[text]()") == "text"
        assert preprocessor.markdown_to_jira("| [1]() |\n|---|\n|  |") == (
            "||1||\n| |"
        )

    def test_empty_image_source_is_not_an_image(self, preprocessor):
        assert preprocessor.markdown_to_jira("![alt]()") == "alt"

    def test_nested_image_keeps_its_own_delimiters(self, preprocessor):
        """The "|" and "!" neutralization applies to author text, not to
        markup this renderer emitted."""
        assert preprocessor.markdown_to_jira(
            "[![alt](https://x.test/i.png)](https://x.test)"
        ) == "[!https://x.test/i.png|alt=alt!|https://x.test]"

    def test_image_source_braces_are_percent_encoded(self, preprocessor):
        assert preprocessor.markdown_to_jira("![a](https://x.test/a{b}.png)") == (
            "!https://x.test/a%7Bb%7D.png|alt=a!"
        )

    def test_image_with_a_bang_in_the_url_falls_back_to_the_alt(self, preprocessor):
        """"!" ends image markup and Jira accepts no encoding for it
        there; the alternative is markup that eats the paragraph."""
        assert preprocessor.markdown_to_jira("![the chart](we!rd.png)") == "the chart"

    def test_image_parameters_are_dropped_on_read(self, preprocessor):
        """Markdown has nowhere to put them, and jira2markdown's Pandoc
        suffix comes back as visible text beside the image."""
        assert preprocessor.jira_to_markdown("!x.png|width=200,alt=a b!") == (
            "![a b](x.png)"
        )
        assert preprocessor.jira_to_markdown("!x.png|thumbnail!") == "![](x.png)"

    def test_caret_opening_a_link_alias_is_escaped(self, preprocessor):
        """"[^name]" is Jira's attachment-link syntax, so an alias that
        opens with a caret becomes a broken attachment reference."""
        assert preprocessor.markdown_to_jira("[^1](https://x.test/p)") == (
            "[\\^1|https://x.test/p]"
        )
        assert preprocessor.markdown_to_jira("[a^b](https://x.test/p)") == (
            "[a^b|https://x.test/p]"
        )

    def test_mention_pattern_does_not_span_a_cell_boundary(self, preprocessor):
        """A looser "[~ anything ]" matched across cells, and passing
        that through verbatim handed Jira a link that ate the row."""
        result = preprocessor.markdown_to_jira("|  |  |\n|---|---|\n| [~ | ]( |")
        assert "[~" not in result
        assert result.count("\n") == 1


class TestTableCellHazards:
    """A cell is the tightest context: its delimiter is one character,
    it opens a line, and it cannot contain a newline."""

    def test_literal_pipe_in_a_cell(self, preprocessor):
        assert preprocessor.markdown_to_jira("| p |\n|---|\n| a\\|b |") == (
            "||p||\n|a&#124;b|"
        )

    def test_link_pipes_are_structural_and_survive(self, preprocessor):
        result = preprocessor.markdown_to_jira(
            "| p |\n|---|\n| [t](https://x.test) |"
        )
        assert result == "||p||\n|[t|https://x.test]|"

    def test_line_start_token_in_a_cell(self, preprocessor):
        assert preprocessor.markdown_to_jira("| p |\n|---|\n| h1. x |") == (
            "||p||\n|h1&#46; x|"
        )

    def test_cell_ending_in_a_backslash_round_trips(self, preprocessor):
        """Two delimiters at once: the backslash escapes Jira's closing
        "|" on the way out, and GFM's on the way back - GFM splits on any
        "|" not directly preceded by a backslash, without counting them,
        so the cell swallowed the rest of the row."""
        markup = preprocessor.markdown_to_jira("| p |\n|---|\n| C:\\\\logs\\\\ |")
        assert markup == "||p||\n|C:&#92;logs&#92;|"
        for _ in range(3):
            markup = preprocessor.markdown_to_jira(
                preprocessor.jira_to_markdown(markup)
            )
        assert markup == "||p||\n|C:&#92;logs&#92;|"

    def test_hard_break_in_a_cell_keeps_the_row_intact(self, preprocessor):
        """A raw newline ends the row; Jira's in-cell break has none."""
        result = preprocessor.markdown_to_jira("| p |\n|---|\n| a<br>b |")
        assert result.count("\n") == 1
        assert result.endswith("|a \\\\ b|")

    def test_empty_cell_is_not_collapsed(self, preprocessor):
        assert preprocessor.markdown_to_jira("| a | b |\n|---|---|\n|  | y |") == (
            "||a||b||\n| |y|"
        )
