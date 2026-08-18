"""Jira-specific text preprocessing module.

Converts between Jira wiki markup (Server/DC and Cloud API v2) and
Markdown using real parsers instead of regex chains:

- Jira wiki markup -> Markdown: ``jira2markdown`` (pyparsing grammar)
- Markdown -> Jira wiki markup: ``mistletoe`` with a customized
  :class:`~mistletoe.contrib.jira_renderer.JiraRenderer`

Both directions fall back to returning the input text unchanged if the
underlying parser raises, so a conversion bug never destroys content.
"""

import logging
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from jira2markdown import convert as _jira2markdown_convert
from jira2markdown.elements import MarkupElements
from jira2markdown.markup.advanced import Code
from mistletoe.block_token import Document
from mistletoe.contrib import jira_renderer as jira_renderer_module
from mistletoe.contrib.jira_renderer import JiraRenderer
from pyparsing import ParseResults

from .base import BasePreprocessor, _extract_blocks, _restore_blocks

logger = logging.getLogger("mcp-atlassian")

# jira2markdown's pyparsing grammar costs roughly 0.25 ms per character;
# above this size the simpler regex fallback keeps latency bounded.
_WIKI_PARSER_MAX_CHARS = 20_000


class _CodeBlock(Code):
    """{code} conversion without jira2markdown's "Java" default language."""

    def action(self, tokens: ParseResults) -> str:
        lang = (tokens.lang or "").lower()
        text = tokens.text.strip("\n")
        return f"```{lang}\n{text}\n```"


_WIKI_ELEMENTS = MarkupElements()
_WIKI_ELEMENTS.replace(Code, _CodeBlock)


@lru_cache(maxsize=128)
def _convert_wiki_cached(text: str) -> str:
    """Convert Jira wiki markup to Markdown, memoized.

    Issue descriptions are converted repeatedly across get/search
    calls; the conversion is pure, so caching is safe.
    """
    return _jira2markdown_convert(text, elements=_WIKI_ELEMENTS)


# Lines using Jira's own nested-list syntax (e.g. "** item", "*# item").
# Agents sometimes send Jira markup directly; these lines are not valid
# Markdown constructs, so they are preserved verbatim (issue #786).
# Pure "#"-runs are excluded: "## text" is a Markdown heading.
_JIRA_NESTED_LIST_RE = r"^(?!#+ )[*#]{2,} .*$"

# Jira user mentions like [~username] or [~accountid:...] must survive
# Markdown parsing untouched.
_JIRA_MENTION_RE = r"\[~[^\]\n]+\]"

# Issue keys may carry numeric segments (e.g. PROJ-123-45) on some
# Server/DC setups (issue #1476).
_ISSUE_KEY_PATTERN = r"[A-Z][A-Z0-9_]+-\d+(?:-\d+)*"

_LIST_ITEM_RE = re.compile(r"^(\s*)((?:[-+*]|\d{1,9}[.)])\s+)(\S.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Inline HTML formatting tags mapped to Jira text-effect markers.
# The same marker opens and closes the effect.
_HTML_TAG_MARKERS = {
    "cite": "??",
    "q": "??",
    "del": "-",
    "s": "-",
    "strike": "-",
    "ins": "+",
    "u": "+",
    "sup": "^",
    "sub": "~",
    "b": "*",
    "strong": "*",
    "i": "_",
    "em": "_",
}

_HTML_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s[^>]*?)?)/?>")
_COLOR_ATTR_RE = re.compile(r"color\s*[:=]\s*[\"']?(#?\w+)")


def _convert_panel(params: str | None, content: str) -> str:
    """Convert a Jira {panel} block to markdown (regex fallback path)."""
    title = ""
    if params:
        title_match = re.search(r"title=([^|}]+)", params)
        if title_match:
            title = title_match.group(1).strip()
    content = content.strip()
    if title:
        return f"\n**{title}**\n{content}\n"
    return f"\n{content}\n"


def _convert_jira_list_line(match: re.Match[str]) -> str:
    """Convert one Jira list line to Markdown (regex fallback path)."""
    jira_bullets = match.group(1)
    content = match.group(2)
    indent = " " * ((len(jira_bullets) - 1) * 2)
    prefix = "1." if jira_bullets[-1] == "#" else "-"
    return f"{indent}{prefix} {content}"


def _regex_jira_to_markdown(input_text: str) -> str:
    """Convert Jira wiki markup to Markdown with regex heuristics.

    Fallback used when the input is too large for the pyparsing-based
    converter or when that converter raises.  Less accurate than
    ``jira2markdown`` but fast and linear.
    """
    output = input_text

    # Protect code/noformat/inline-code blocks from downstream
    # transformations by replacing them with placeholders.
    code_blocks: list[str] = []
    inline_codes: list[str] = []

    def _jira_code_to_md(match: re.Match[str]) -> str:
        lang = match.group(1) or ""
        content = match.group(2)
        return f"```{lang}\n{content}\n```"

    output = _extract_blocks(
        output,
        r"\{code(?::([a-z]+))?\}([\s\S]*?)\{code\}",
        _jira_code_to_md,
        code_blocks,
        "CODEBLOCK",
        flags=re.MULTILINE,
    )
    output = _extract_blocks(
        output,
        r"\{noformat\}([\s\S]*?)\{noformat\}",
        lambda m: f"```\n{m.group(1)}\n```",
        code_blocks,
        "CODEBLOCK",
    )
    output = _extract_blocks(
        output,
        r"\{\{([^}]+)\}\}",
        lambda m: f"`{m.group(1)}`",
        inline_codes,
        "INLINECODE",
    )

    # Block quotes
    output = re.sub(r"^bq\.(.*?)$", r"> \1\n", output, flags=re.MULTILINE)

    # Text formatting (bold, italic)
    output = re.sub(
        r"([*_])(.*?)\1",
        lambda match: ("**" if match.group(1) == "*" else "*")
        + match.group(2)
        + ("**" if match.group(1) == "*" else "*"),
        output,
    )

    # Multi-level lists
    output = re.sub(
        r"^((?:#|-|\+|\*)+) (.*)$",
        _convert_jira_list_line,
        output,
        flags=re.MULTILINE,
    )

    # Headers
    output = re.sub(
        r"^h([0-6])\.(.*)$",
        lambda match: "#" * int(match.group(1)) + match.group(2),
        output,
        flags=re.MULTILINE,
    )

    # Citation (non-overlapping alternation to avoid catastrophic backtracking)
    output = re.sub(r"\?\?([^?]+(?:\?[^?]+)*)\?\?", r"<cite>\1</cite>", output)

    # Inserted text
    output = re.sub(r"\+([^+]*)\+", r"<ins>\1</ins>", output)

    # Superscript
    output = re.sub(r"\^([^^]*)\^", r"<sup>\1</sup>", output)

    # Subscript
    output = re.sub(r"~([^~]*)~", r"<sub>\1</sub>", output)

    # Quote blocks
    output = re.sub(
        r"\{quote\}([\s\S]*)\{quote\}",
        lambda match: "\n".join([f"> {line}" for line in match.group(1).split("\n")]),
        output,
        flags=re.MULTILINE,
    )

    # Panel blocks - extract content, optionally show title as bold
    output = re.sub(
        r"\{panel(?::([^}]*))?\}([\s\S]*?)\{panel\}",
        lambda match: _convert_panel(match.group(1), match.group(2)),
        output,
        flags=re.MULTILINE,
    )

    # Images with alt text
    output = re.sub(
        r"!([^|\n\s]+)\|([^\n!]*)alt=([^\n!\,]+?)(,([^\n!]*))?!",
        r"![\3](\1)",
        output,
    )

    # Images with other parameters (ignore them)
    output = re.sub(r"!([^|\n\s]+)\|([^\n!]*)!", r"![](\1)", output)

    # Images without parameters
    output = re.sub(r"!([^\n\s!]+)!", r"![](\1)", output)

    # Links
    output = re.sub(r"\[([^|]+)\|(.+?)\]", r"[\1](\2)", output)
    output = re.sub(r"\[(.+?)\]([^\(])", r"\1\2", output)

    # Colored text
    output = re.sub(
        r"\{color:([^}]+)\}([\s\S]*?)\{color\}",
        r"<span style=\"color:\1\">\2</span>",
        output,
        flags=re.MULTILINE,
    )

    # Convert Jira table headers (||) to markdown table format
    lines = output.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if "||" in line:
            lines[i] = line.replace("||", "|")
            header_cells = lines[i].count("|") - 1
            if header_cells > 0:
                separator_line = "|" + "---|" * header_cells
                lines.insert(i + 1, separator_line)
                i += 1
        i += 1
    output = "\n".join(lines)

    # Restore code/noformat blocks and inline code
    output = _restore_blocks(output, code_blocks, "CODEBLOCK")
    output = _restore_blocks(output, inline_codes, "INLINECODE")

    return output


def _normalize_list_indentation(text: str) -> str:
    """Normalize list indentation to CommonMark-required columns.

    Agents commonly indent nested lists with two spaces per level
    (e.g. ``1. a\\n  1. b``).  CommonMark requires a nested item to be
    indented to the parent's content column (three columns for
    ``1. ``), so a strict parser flattens such lists.  This pass
    re-indents list-item lines so each intended level starts at its
    parent's content column, leaving fenced code untouched.
    """
    lines = text.split("\n")
    result: list[str] = []
    # Stack of (original_indent, normalized_indent, content_column)
    stack: list[tuple[int, int, int]] = []
    in_fence = False

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue

        match = _LIST_ITEM_RE.match(line)
        if not match:
            # Blank lines and indented continuations keep list context;
            # flush it on any other unindented text.
            if line.strip() and not line.startswith(" "):
                stack = []
            result.append(line)
            continue

        indent = len(match.group(1).expandtabs(4))
        marker = match.group(2)
        content = match.group(3)

        while stack and indent < stack[-1][0]:
            stack.pop()

        if stack and indent == stack[-1][0]:
            norm = stack[-1][1]
            stack[-1] = (indent, norm, norm + len(marker))
        elif stack and indent > stack[-1][0]:
            norm = stack[-1][2]
            stack.append((indent, norm, norm + len(marker)))
        else:
            norm = indent
            stack.append((indent, norm, norm + len(marker)))

        result.append(" " * norm + marker + content)

    return "\n".join(result)


class JiraMarkupRenderer(JiraRenderer):
    """Markdown -> Jira wiki markup renderer with MCP-specific fixes.

    Extends mistletoe's stock ``JiraRenderer`` with:

    - no backslash-escaping inside ``{{inline code}}``
    - code-block language normalization to Jira's supported set
    - image alt text (``!src|alt=text!``)
    - newline-preserving soft line breaks
    - inline HTML formatting tags translated to Jira text effects
    """

    def __init__(
        self,
        *extras: Any,
        normalize_language: Callable[[str | None], str | None] | None = None,
    ) -> None:
        super().__init__(*extras)
        self.normalize_language = normalize_language
        self._span_color_stack: list[bool] = []

    # Jira's emphasis engine triggers *bold* and _italic_ even inside
    # words (the classic "snake_case turns italic" problem), so those
    # two are escaped whenever they touch non-whitespace.  The other
    # text effects (-strike-, +ins+, ^sup^, ~sub~) only matter at word
    # boundaries, and macros/links ({...}, [...]) trigger anywhere.
    _EMPHASIS_CHARS = frozenset("*_")
    _BOUNDARY_EFFECT_CHARS = frozenset("+^~-")
    _MACRO_CHARS = frozenset("{}[]")

    def render_raw_text(self, token: Any, escape: bool = True) -> str:
        """Escape Jira markup characters where Jira would interpret them.

        ``*`` and ``_`` are escaped aggressively because Jira applies
        emphasis intraword (``foo_bar_baz`` renders with italic "bar"
        otherwise; ``\\_`` renders as a plain underscore).  The
        boundary-only effect characters are left alone between word
        characters, so ``Sub-item`` and ``2024-01-01`` stay readable.
        """
        if not escape:
            return str(token.content)

        text = token.content
        length = len(text)
        if length == 1 and text in (
            self._EMPHASIS_CHARS | self._BOUNDARY_EFFECT_CHARS | self._MACRO_CHARS
        ):
            return "\\" + text

        result: list[str] = []
        for i, char in enumerate(text):
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i < length - 1 else ""
            prev_solid = bool(prev) and not prev.isspace()
            next_solid = bool(nxt) and not nxt.isspace()
            if char in self._MACRO_CHARS or char in self._EMPHASIS_CHARS:
                if prev_solid or next_solid:
                    result.append("\\" + char)
                    continue
            elif char in self._BOUNDARY_EFFECT_CHARS:
                prev_word = bool(prev) and (prev.isalnum() or prev == "_")
                next_word = bool(nxt) and (nxt.isalnum() or nxt == "_")
                could_open = not prev_word and next_solid
                could_close = prev_solid and not next_word
                if could_open or could_close:
                    result.append("\\" + char)
                    continue
            result.append(char)
        return "".join(result)

    def render_inline_code(self, token: Any) -> str:
        # Jira renders {{...}} content literally; backslash escapes
        # inserted by render_raw_text would show up verbatim.
        content = token.children[0].content
        # Content starting or ending with a brace would merge with the
        # {{ }} delimiters into {{{...}}}, which Jira's renderer cannot
        # disambiguate; pad with spaces to keep the delimiters intact
        # (issue #1 on this fork).
        if content and (content[0] in "{}" or content[-1] in "{}"):
            content = f" {content} "
        return "{{" + content + "}}"

    def render_block_code(self, token: Any) -> str:
        inner = token.children[0].content
        # Jira terminates a {code} block at the first inner {code...}
        # token and offers no escaping, so content that mentions {code}
        # (e.g. docs about Jira markup) must use {noformat} instead.
        # Content containing both delimiters is not representable;
        # {code} is kept as the least-bad option.
        if "{code" in inner and "{noformat" not in inner:
            return "{noformat}\n" + inner + "{noformat}" + self._block_eol(token)
        lang = token.language or ""
        if self.normalize_language is not None:
            lang = self.normalize_language(lang) or ""
        attr = f":{lang}" if lang else ""
        return "{code" + attr + "}\n" + inner + "{code}" + self._block_eol(token)

    def render_image(self, token: Any) -> str:
        alt = "".join(
            child.content for child in token.children if hasattr(child, "content")
        )
        # "!" ends the image markup, "|" and "," separate its
        # parameters; none of them are escapable inside it.
        alt = " ".join(alt.replace("!", "").replace("|", " ").replace(",", " ").split())
        if alt:
            return f"!{token.src}|alt={alt}!"
        return f"!{token.src}!"

    def render_link(self, token: Any) -> str:
        # "|" separates alias from URL in [text|url] and cannot be
        # escaped, so pipes in the visible text become HTML entities.
        inner = self.render_inner(token).replace("|", "&#124;")
        target = jira_renderer_module.escape_url(token.target)
        title = (
            "|" + jira_renderer_module.escape_link_chars(token.title)
            if token.title
            else ""
        )
        return f"[{inner}|{target}{title}]"

    def render_table_cell(self, token: Any, in_header: bool = False) -> str:
        template = "||{inner}" if in_header else "|{inner}"
        inner = self.render_inner(token).replace("|", "\\|")
        # A raw newline (from a hard break or <br>) ends the table row;
        # Jira's in-cell line break is "\\" without a newline.
        inner = re.sub(r"(?:\\\\)?\n", r" \\\\ ", inner).strip()
        return template.format(inner=inner or " ")

    def render_line_break(self, token: Any) -> str:
        # Jira preserves single newlines, so keeping soft breaks as
        # newlines matches the author's visual intent.
        if token.soft:
            return "\n"
        return "\\\\\n"

    def render_thematic_break(self, token: Any) -> str:
        # Keep a blank line after the rule so following text starts a
        # fresh paragraph instead of hugging the ruler line.
        return "----" + self._block_eol(token)

    def render_quote(self, token: Any) -> str:
        # "bq. " only quotes a single line; any quote whose content
        # spans multiple lines needs a {quote} block.
        self.lastChildOfQuotes.append(token.children[-1])
        inner = self.render_inner(token)
        del self.lastChildOfQuotes[-1]
        if len(token.children) == 1 and "\n" not in inner.rstrip("\n"):
            return "bq. " + inner + self._block_eol(token)[0:-1]
        return "{quote}\n" + inner + "{quote}" + self._block_eol(token)

    def render_html_span(self, token: Any) -> str:
        return self._convert_html_tag(token.content)

    def render_html_block(self, token: Any) -> str:
        content = _HTML_TAG_RE.sub(
            lambda m: self._convert_html_tag(m.group(0)), token.content
        )
        return content + self._block_eol(token)

    def _convert_html_tag(self, content: str) -> str:
        """Translate a single inline HTML tag to Jira markup."""
        match = _HTML_TAG_RE.fullmatch(content.strip())
        if not match:
            return content
        closing, tag, attrs = match.group(1), match.group(2).lower(), match.group(3)
        if tag == "br":
            return "\n"
        marker = _HTML_TAG_MARKERS.get(tag)
        if marker is not None:
            return marker
        if tag in ("span", "font"):
            if closing:
                had_color = (
                    self._span_color_stack.pop() if self._span_color_stack else False
                )
                return "{color}" if had_color else ""
            color_match = _COLOR_ATTR_RE.search(attrs)
            if color_match:
                self._span_color_stack.append(True)
                return "{color:" + color_match.group(1) + "}"
            self._span_color_stack.append(False)
            return ""
        return content


class JiraPreprocessor(BasePreprocessor):
    """Handles text preprocessing for Jira content."""

    # Step 1: Valid JIRA languages (official list)
    # Source: https://jira.atlassian.com/browse/JRASERVER-21067 (JIRA 7.5.0+)
    # and JIRA v9.12.12 release notes
    # Official documentation: https://jira.atlassian.com/secure/WikiRendererHelpAction.jspa
    VALID_JIRA_LANGUAGES = {
        # Core languages from JIRA 7.5.0+
        "actionscript",
        "actionscript3",
        "ada",
        "applescript",
        "bash",
        "sh",  # alias for bash
        "c",
        "c#",
        "csharp",  # alias for c#
        "cs",  # alias for c#
        "c++",
        "cpp",  # alias for c++
        "css",
        "sass",  # CSS preprocessor
        "less",  # CSS preprocessor
        "coldfusion",
        "delphi",
        "diff",
        "patch",  # alias for diff
        "erlang",
        "erl",  # alias for erlang
        "go",
        "groovy",
        "haskell",
        "html",
        "xml",
        "java",
        "javafx",
        "javascript",
        "js",  # alias for javascript
        "json",
        "lua",
        "nyan",
        "objc",
        "objective-c",  # alias for objc
        "perl",
        "php",
        "powershell",
        "ps1",  # alias for powershell
        "python",
        "py",  # alias for python
        "r",
        "rainbow",
        "ruby",
        "rb",  # alias for ruby
        "scala",
        "sql",
        "swift",
        "visualbasic",
        "vb",  # alias for visualbasic
        "yaml",
        "yml",  # alias for yaml
        "none",  # plain text, no highlighting
    }

    # Step 2: Mapping for unsupported languages to closest valid JIRA alternative
    # Only map to actual JIRA languages; unmapped languages will return None → {code}
    LANGUAGE_MAPPING = {
        # Dockerfile → bash (similar shell syntax)
        "dockerfile": "bash",
        "docker": "bash",
        # TypeScript → javascript
        "typescript": "javascript",
        "ts": "javascript",
        "tsx": "javascript",
        # JSX/React → javascript
        "jsx": "javascript",
        # Kotlin → java (JVM-based language)
        "kotlin": "java",
        "kt": "java",
        # Build files → bash
        "makefile": "bash",
        "make": "bash",
        "cmake": "bash",
    }

    def __init__(
        self, base_url: str = "", disable_translation: bool = False, **kwargs: Any
    ) -> None:
        """
        Initialize the Jira text preprocessor.

        Args:
            base_url: Base URL for Jira API
            disable_translation: If True, disable markup translation between formats
            **kwargs: Additional arguments for the base class
        """
        super().__init__(base_url=base_url, **kwargs)
        self.disable_translation = disable_translation

    def clean_jira_text(self, text: str) -> str:
        """
        Clean Jira text content by:
        1. Processing user mentions and links
        2. Converting Jira markup to markdown (if translation enabled)
        3. Converting HTML/wiki markup to markdown (if translation enabled)
        """
        if not text:
            return ""

        # Process user mentions
        mention_pattern = r"\[~accountid:(.*?)\]"
        text = self._process_mentions(text, mention_pattern)

        # Process Jira smart links
        text = self._process_smart_links(text)

        # Convert markup only if translation is enabled
        if not self.disable_translation:
            # Smart-link processing above already produced Markdown
            # links; protect them from the wiki-markup parser.
            md_links: list[str] = []
            text = _extract_blocks(
                text,
                r"\[[^\]\n]*\]\([^)\n]*\)",
                lambda m: m.group(0),
                md_links,
                "MDLINK",
            )

            # First convert any Jira markup to Markdown
            text = self.jira_to_markdown(text)

            # Protect Markdown autolinks (<https://...>) and the inline
            # formatting tags this pipeline emits itself (<ins>, <cite>,
            # <span style="color:...">, ...) from the HTML-to-Markdown
            # pass: markdownify collapses newlines and escapes Markdown
            # in every text node it touches, so it must only run when
            # genuine HTML content remains.
            autolinks: list[str] = []
            text = _extract_blocks(
                text,
                r"<[a-zA-Z][a-zA-Z0-9+.-]*://[^>\s]+>"
                r"|</?(?:span|ins|cite|sup|sub|del|u|q|font)\b[^<>]*>",
                lambda m: m.group(0),
                autolinks,
                "AUTOLINK",
            )

            # Then convert any remaining HTML to markdown
            text = self._convert_html_to_markdown(text)

            text = _restore_blocks(text, autolinks, "AUTOLINK")
            text = _restore_blocks(text, md_links, "MDLINK")

        return text.strip()

    def _process_mentions(self, text: str, pattern: str) -> str:
        """
        Process user mentions in text.

        Args:
            text: The text containing mentions
            pattern: Regular expression pattern to match mentions

        Returns:
            Text with mentions replaced with display names
        """
        mentions = re.findall(pattern, text)
        for account_id in mentions:
            try:
                # Note: This is a placeholder - actual user fetching should be injected
                display_name = f"User:{account_id}"
                text = text.replace(f"[~accountid:{account_id}]", display_name)
            except Exception as e:
                logger.error(f"Error processing mention for {account_id}: {str(e)}")
        return text

    def _process_smart_links(self, text: str) -> str:
        """Process Jira/Confluence smart links."""
        # Pattern matches: [text|url|smart-link]
        link_pattern = r"\[(.*?)\|(.*?)\|smart-link\]"
        matches = re.finditer(link_pattern, text)

        for match in matches:
            full_match = match.group(0)
            link_text = match.group(1)
            link_url = match.group(2)

            # Extract issue key if it's a Jira issue link
            issue_key_match = re.search(
                rf"browse/({_ISSUE_KEY_PATTERN})(?=$|[/?#])", link_url
            )
            # Check if it's a Confluence wiki link
            confluence_match = re.search(
                r"wiki/spaces/.+?/pages/\d+/(.+?)(?:\?|$)", link_url
            )

            if issue_key_match:
                issue_key = issue_key_match.group(1)
                clean_url = f"{self.base_url}/browse/{issue_key}"
                text = text.replace(full_match, f"[{issue_key}]({clean_url})")
            elif confluence_match:
                url_title = confluence_match.group(1)
                readable_title = url_title.replace("+", " ")
                readable_title = re.sub(
                    rf"^{_ISSUE_KEY_PATTERN}\s+", "", readable_title
                )
                text = text.replace(full_match, f"[{readable_title}]({link_url})")
            else:
                clean_url = link_url.split("?")[0]
                text = text.replace(full_match, f"[{link_text}]({clean_url})")

        return text

    def jira_to_markdown(self, input_text: str) -> str:
        """
        Convert Jira wiki markup to Markdown format.

        Uses the ``jira2markdown`` parser, which handles the full wiki
        syntax (headings, text effects, lists, tables, panels, code
        blocks, links, mentions, colors) far more reliably than regex
        chains.  Returns the input unchanged if parsing fails.

        Args:
            input_text: Text in Jira markup format

        Returns:
            Text in Markdown format (or original text if translation disabled)
        """
        if not input_text:
            return ""

        if self.disable_translation:
            return input_text

        if len(input_text) > _WIKI_PARSER_MAX_CHARS:
            logger.debug(
                "Input exceeds %d chars; using regex wiki-markup fallback",
                _WIKI_PARSER_MAX_CHARS,
            )
            return _regex_jira_to_markdown(input_text).rstrip("\n")

        try:
            output = _convert_wiki_cached(input_text)
        except Exception as e:
            logger.warning(f"Error parsing Jira markup, using regex fallback: {e}")
            try:
                return _regex_jira_to_markdown(input_text).rstrip("\n")
            except Exception:
                return input_text

        # Normalize HTML produced by jira2markdown to the tags this
        # pipeline has historically used, protecting code content.
        code_blocks: list[str] = []
        inline_codes: list[str] = []
        output = _extract_blocks(
            output,
            r"```[^\n]*\n[\s\S]*?\n```",
            lambda m: m.group(0),
            code_blocks,
            "J2MCODE",
        )
        output = _extract_blocks(
            output,
            r"`[^`\n]+`",
            lambda m: m.group(0),
            inline_codes,
            "J2MINLINE",
        )

        output = re.sub(r"<(/?)u>", r"<\1ins>", output)
        output = re.sub(r"<(/?)q>", r"<\1cite>", output)
        output = re.sub(
            r"<font color=[\"']?(#?\w+)[\"']?>",
            r'<span style="color:\1">',
            output,
        )
        output = output.replace("</font>", "</span>")

        output = _restore_blocks(output, inline_codes, "J2MINLINE")
        output = _restore_blocks(output, code_blocks, "J2MCODE")

        return output.rstrip("\n")

    def _normalize_code_language(self, lang: str | None) -> str | None:
        """
        Normalize and map markdown code language to JIRA-supported language.

        Step 3: Default handling - unmapped languages return None for plain {code}

        Args:
            lang: Language identifier from markdown code block

        Returns:
            Valid JIRA language string, or None for plain {code} block
        """
        if not lang:
            return None

        lang_lower = lang.lower()

        # Step 1: Check if already valid JIRA language
        if lang_lower in self.VALID_JIRA_LANGUAGES:
            return lang_lower

        # Step 2: Check language mapping
        if lang_lower in self.LANGUAGE_MAPPING:
            return self.LANGUAGE_MAPPING[lang_lower]

        # Step 3: Default - unmapped language returns None for plain {code}
        return None

    def markdown_to_jira(self, input_text: str) -> str:
        """
        Convert Markdown syntax to Jira wiki markup syntax.

        Parses the input as CommonMark (plus tables and strikethrough)
        with ``mistletoe`` and renders Jira wiki markup, so nested
        lists, tables with alignment rows, blockquotes, and code blocks
        all survive the conversion.  Jira-specific syntax that agents
        may emit directly — ``[~mentions]`` and nested-list lines like
        ``** item`` — is preserved verbatim.  Returns the input
        unchanged if parsing fails.

        Args:
            input_text: Text in Markdown format

        Returns:
            Text in Jira markup format (or original text if translation disabled)
        """
        if not input_text:
            return ""

        if self.disable_translation:
            return input_text

        try:
            return self._markdown_to_jira(input_text)
        except Exception as e:
            logger.warning(f"Error converting Markdown to Jira markup: {e}")
            return input_text

    def _markdown_to_jira(self, input_text: str) -> str:
        """Run the actual Markdown -> Jira wiki markup conversion."""
        # Preserve Jira syntax the Markdown parser would mangle.
        mentions: list[str] = []
        output = _extract_blocks(
            input_text,
            _JIRA_MENTION_RE,
            lambda m: m.group(0),
            mentions,
            "JIRAMENTION",
        )
        jira_lists: list[str] = []
        output = _extract_blocks(
            output,
            _JIRA_NESTED_LIST_RE,
            lambda m: m.group(0),
            jira_lists,
            "JIRALIST",
            flags=re.MULTILINE,
        )

        output = _normalize_list_indentation(output)

        with JiraMarkupRenderer(
            normalize_language=self._normalize_code_language
        ) as renderer:
            output = renderer.render(Document(output))

        output = output.rstrip("\n")
        output = _restore_blocks(output, jira_lists, "JIRALIST")
        output = _restore_blocks(output, mentions, "JIRAMENTION")
        return output
