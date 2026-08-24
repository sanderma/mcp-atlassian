"""Property-based tests for the Jira DC markup conversion.

The matrix in ``test_markup_edge_cases`` covers the hazards someone
thought of. This module covers the ones nobody did: Hypothesis builds
Markdown documents out of the characters that collide with Jira's markup
and checks the properties that must hold for *every* document, not for a
list of them.

The properties are deliberately few and strong:

``never_raises``
    ``markdown_to_jira`` swallows exceptions and returns its input, so a
    crash does not surface as an error - it silently sends raw Markdown
    to Jira, where a heading renders as "# Title" and a table as pipes.
    The private entry point is used so a regression actually fails.

``structurally_valid``
    The output parses as Jira markup: no unbalanced block macro (which
    swallows the rest of the description), no merged brace delimiters,
    no leaked internal placeholder.

``stable``
    After the first write, another read-modify-write cycle produces
    byte-identical markup. This is what makes it safe for an agent to
    edit one line of an issue without disturbing the rest.

``code_is_verbatim``
    Everything inside a fenced block reaches Jira unchanged. Escaping
    code content is not a cosmetic bug: it changes what the command or
    the snippet does.
"""

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mcp_atlassian.preprocessing.jira import JiraPreprocessor

from .test_jira_markup_validity import assert_valid_jira_markup

# Characters that mean something to Jira, to Markdown, or to both. Plain
# letters and digits are in there so the generated text has anchors the
# escaping rules can key off (many are position-sensitive).
HAZARD_ALPHABET = "ab12 \t*_-+^~[]{}|!?()&#:;.\\<>\"'%@/"

fragment = st.text(alphabet=HAZARD_ALPHABET, min_size=0, max_size=24)
line = st.text(alphabet=HAZARD_ALPHABET.replace("\t", ""), min_size=0, max_size=24).map(
    lambda s: s.replace("\n", " ")
)


@st.composite
def paragraph(draw: st.DrawFn) -> str:
    parts = draw(st.lists(line, min_size=1, max_size=3))
    return "\n".join(parts)


@st.composite
def heading(draw: st.DrawFn) -> str:
    level = draw(st.integers(min_value=1, max_value=6))
    return "#" * level + " " + draw(line)


@st.composite
def bullet_list(draw: st.DrawFn) -> str:
    items = draw(st.lists(line, min_size=1, max_size=3))
    marker = draw(st.sampled_from(["-", "*", "+"]))
    indent = draw(st.sampled_from(["", "  "]))
    return "\n".join(f"{indent * i}{marker} {item}" for i, item in enumerate(items))


@st.composite
def ordered_list(draw: st.DrawFn) -> str:
    items = draw(st.lists(line, min_size=1, max_size=3))
    delim = draw(st.sampled_from([".", ")"]))
    return "\n".join(f"{n}{delim} {item}" for n, item in enumerate(items, 1))


@st.composite
def quote(draw: st.DrawFn) -> str:
    items = draw(st.lists(line, min_size=1, max_size=3))
    return "\n".join(f"> {item}" for item in items)


@st.composite
def fenced_code(draw: st.DrawFn) -> str:
    language = draw(st.sampled_from(["", "python", "js", "yaml", "nonesuch"]))
    body = draw(st.lists(line, min_size=0, max_size=3))
    fence = draw(st.sampled_from(["```", "~~~"]))
    return f"{fence}{language}\n" + "\n".join(body) + f"\n{fence}"


@st.composite
def table(draw: st.DrawFn) -> str:
    columns = draw(st.integers(min_value=1, max_value=3))
    # A literal pipe in a GFM cell is a column separator, not content.
    cell = st.text(
        alphabet=HAZARD_ALPHABET.replace("|", "").replace("\t", ""),
        min_size=0,
        max_size=12,
    )
    rows = draw(
        st.lists(
            st.lists(cell, min_size=columns, max_size=columns), min_size=1, max_size=3
        )
    )
    header = (
        "| "
        + " | ".join(draw(st.lists(cell, min_size=columns, max_size=columns)))
        + " |"
    )
    divider = "|" + "---|" * columns
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


@st.composite
def inline_rich(draw: st.DrawFn) -> str:
    """A paragraph carrying the inline constructs, not just raw text."""
    template = draw(
        st.sampled_from(
            [
                "text `{f}` end",
                "text **{f}** end",
                "text *{f}* end",
                "text ~~{f}~~ end",
                "text [{f}](https://x.test/p) end",
                "text ![{f}](https://x.test/i.png) end",
                "text <https://x.test/{f}> end",
                "text https://x.test/{f} end",
            ]
        )
    )
    return template.format(f=draw(fragment).replace("\n", " "))


BLOCK = st.one_of(
    paragraph(),
    heading(),
    bullet_list(),
    ordered_list(),
    quote(),
    fenced_code(),
    table(),
    inline_rich(),
)

document = st.lists(BLOCK, min_size=1, max_size=4).map("\n\n".join)

# Conversion is CPU-bound and the wiki parser is not fast; the default
# deadline flags that as a failure rather than the bug it is not.
SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@pytest.fixture(scope="module")
def preprocessor() -> JiraPreprocessor:
    return JiraPreprocessor(base_url="https://jira.example.com")


class TestConversionProperties:
    @SETTINGS
    @given(markdown=document)
    def test_conversion_never_raises(self, preprocessor, markdown):
        preprocessor._markdown_to_jira(markdown)

    @SETTINGS
    @given(markdown=document)
    def test_output_is_structurally_valid(self, preprocessor, markdown):
        assert_valid_jira_markup(preprocessor.markdown_to_jira(markdown))

    @SETTINGS
    @given(markdown=document)
    def test_reading_back_never_raises(self, preprocessor, markdown):
        markup = preprocessor.markdown_to_jira(markdown)
        preprocessor.jira_to_markdown(markup)
        preprocessor.clean_jira_text(markup)

    @SETTINGS
    @given(markdown=document)
    def test_round_trip_settles_and_does_not_grow(self, preprocessor, markdown):
        """Repeated editing must converge, and never accumulate.

        For real content the markup settles after the *first* write -
        that stricter guarantee is asserted over the curated corpora in
        ``TestRoundTripStability`` and the hazard matrix, where it can be
        stated exactly. Here the generator also produces empty blocks in
        odd nestings (a bullet whose whole content is ">", a quote whose
        whole content is ")"), which take one more cycle because a
        CommonMark parser reads the marker as a block of its own. What
        must hold for *every* document is that the cycle reaches a fixed
        point: once it does, nothing can accumulate. The failure this
        guards against is a literal "*" gaining a backslash per edit
        until "\\\\" injects a line break.
        """
        cycles = [preprocessor.markdown_to_jira(markdown)]
        for _ in range(4):
            cycles.append(
                preprocessor.markdown_to_jira(preprocessor.jira_to_markdown(cycles[-1]))
            )
        assert cycles[-1] == cycles[-2], f"never settles: {cycles}"

    @SETTINGS
    @given(markdown=document)
    def test_nothing_is_silently_dropped(self, preprocessor, markdown):
        """A non-empty document must not convert to nothing."""
        if markdown.strip(" \t\n#>-*+`~|"):
            assert preprocessor.markdown_to_jira(markdown).strip()


class TestCodeContentProperties:
    """Code content is data, not prose: it must arrive byte for byte."""

    @SETTINGS
    @given(
        body=st.lists(
            st.text(alphabet=HAZARD_ALPHABET.replace("\t", ""), max_size=20).map(
                lambda s: s.replace("\n", " ")
            ),
            min_size=1,
            max_size=4,
        ),
        language=st.sampled_from(["", "python", "js", "nonesuch"]),
    )
    def test_fenced_code_content_is_verbatim(self, preprocessor, body, language):
        # A line that is itself a fence would close the block early -
        # that is Markdown's rule, not a conversion bug.
        content = "\n".join(body)
        if re.search(r"^\s*(```|~~~)", content, re.MULTILINE):
            return
        markup = preprocessor.markdown_to_jira(f"```{language}\n{content}\n```")
        opener, _, rest = markup.partition("\n")
        assert opener.startswith("{code") or opener.startswith("{noformat")
        closer = "{noformat}" if opener.startswith("{noformat") else "{code}"
        assert rest.endswith(closer)
        assert rest[: -len(closer)].rstrip("\n") == content.rstrip("\n")

    @SETTINGS
    @given(
        content=st.text(
            alphabet=HAZARD_ALPHABET.replace("`", "").replace("\t", ""), max_size=20
        ).map(lambda s: s.replace("\n", " "))
    )
    def test_inline_code_content_survives_the_round_trip(self, preprocessor, content):
        """Entities are how content survives {{...}}; reading has to
        decode them or the next write encodes them again."""
        if not content.strip():
            return
        markup = preprocessor.markdown_to_jira(f"a `{content}` b")
        back = preprocessor.jira_to_markdown(markup)
        assert preprocessor.markdown_to_jira(back) == markup
