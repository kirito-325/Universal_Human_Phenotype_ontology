"""`sources/pmc.py`: JATS flattening, which is where this module's wrong answers live.

Three failures are pinned here by name, all of them silent:

* `"".join(itertext())` over a body fuses consecutive paragraphs — every sentence
  boundary at a paragraph edge disappears, which a model reads straight past.
* A body deposited as loose `<p>` children plus one `Supplementary Material` `<sec>` has
  one `<sec>`, so an `if not secs` fallback never fires. PMC3030664 and PMC7164637 both
  reported `body_chars: 0` against 12,791 and 10,339 real characters, with `fell_back`
  still False.
* `Supplementary Material` contains the substring `material`, so the alias table bucketed
  it as `methods` — on a Science article the only section, which made `sections=['methods']`
  return an empty back-matter stub.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from hpo_agent.upstream.sources.pmc import (
    SANDBOX_READ_CAP,
    PMCError,
    _block_text,
    _caption_text,
    _inline_text,
    _parse_sections,
    _render_table,
    _resolve_asset,
    canonical_section,
    parse_jats,
)


class TestInlineText:
    def test_keeps_text_after_inline_markup(self):
        node = ET.fromstring("<title>Effect of <italic>TP53</italic> loss</title>")
        assert _inline_text(node) == "Effect of TP53 loss"

    def test_collapses_wrapping_whitespace_to_single_spaces(self):
        node = ET.fromstring("<title>A\n   long\t title</title>")
        assert _inline_text(node) == "A long title"

    def test_an_empty_node_is_an_empty_string(self):
        assert _inline_text(ET.fromstring("<title/>")) == ""


class TestBlockText:
    def test_consecutive_paragraphs_keep_their_boundary(self):
        """itertext fuses these into '...ferroptosis.The P47S...' with no break."""
        node = ET.fromstring(
            "<sec><p>MDM2 regulates ferroptosis.</p><p>The P47S polymorphism.</p></sec>"
        )
        text = _block_text(node)
        assert "ferroptosis.The" not in text
        assert text == "MDM2 regulates ferroptosis.\n\nThe P47S polymorphism."

    def test_inline_markup_inside_a_paragraph_is_kept(self):
        node = ET.fromstring("<sec><p>IC<sub>50</sub> of 4 nM</p></sec>")
        assert _block_text(node) == "IC50 of 4 nM"

    def test_a_tail_after_inline_markup_is_not_dropped(self):
        node = ET.fromstring("<sec><p>See <xref>Fig 1</xref> for details.</p></sec>")
        assert "for details." in _block_text(node)

    def test_a_lifted_table_is_left_out_of_the_running_text(self):
        """Inline, a table-wrap becomes an unreadable run of undelimited cells."""
        node = ET.fromstring(
            "<sec><p>Before.</p><table-wrap><table><tr><td>a</td><td>b</td></tr></table>"
            "</table-wrap><p>After.</p></sec>"
        )
        text = _block_text(node)
        assert "Before." in text and "After." in text
        assert "a" not in text.replace("Before.", "").replace("After.", "")

    def test_a_lifted_figure_caption_is_not_duplicated_into_the_body(self):
        node = ET.fromstring(
            "<sec><p>Body.</p><fig><caption><p>Figure caption.</p></caption></fig></sec>"
        )
        assert "Figure caption" not in _block_text(node)

    def test_skip_first_title_drops_the_heading_the_caller_already_emitted(self):
        node = ET.fromstring("<sec><title>Introduction</title><p>The tumor.</p></sec>")
        assert _block_text(node, skip_first_title=True) == "The tumor."

    def test_without_the_flag_the_title_stays(self):
        node = ET.fromstring("<sec><title>Introduction</title><p>The tumor.</p></sec>")
        assert "Introduction" in _block_text(node)

    def test_runs_of_blank_lines_are_collapsed(self):
        node = ET.fromstring("<sec><p>a</p><sec><p>b</p></sec></sec>")
        assert "\n\n\n" not in _block_text(node)


class TestCanonicalSection:
    @pytest.mark.parametrize(
        ("title", "bucket"),
        [
            ("Methods", "methods"),
            ("Materials and methods", "methods"),
            ("MATERIALS AND METHODS", "methods"),
            ("2. Experimental section", "methods"),
            ("Results", "results"),
            ("Results and Discussion", "results"),
            ("Discussion", "discussion"),
            ("Conclusions", "conclusion"),
            ("Summary and outlook", "conclusion"),
            ("Introduction", "intro"),
            ("Background", "intro"),
        ],
    )
    def test_maps_the_spellings_journals_actually_use(self, title: str, bucket: str):
        assert canonical_section(title) == bucket

    @pytest.mark.parametrize("title", ["1. Introduction", "3.2) Results", "IV. Discussion"])
    def test_strips_section_numbering_before_matching(self, title: str):
        assert canonical_section(title) is not None

    def test_the_roman_numeral_branch_does_not_eat_the_i_of_introduction(self):
        """The separator is required, or 'Introduction' becomes 'ntroduction'."""
        assert canonical_section("Introduction") == "intro"

    @pytest.mark.parametrize(
        "title", ["Supplementary Material", "Supplementary Materials", "Supporting Information"]
    )
    def test_back_matter_is_excluded_despite_containing_material(self, title: str):
        assert canonical_section(title) is None

    def test_an_unrecognised_heading_is_not_forced_into_a_bucket(self):
        assert canonical_section("Acknowledgements") is None

    def test_no_title_is_no_bucket(self):
        assert canonical_section(None) is None
        assert canonical_section("") is None


class TestRenderTable:
    def test_flattens_rows_to_pipe_delimited_lines(self):
        wrap = ET.fromstring(
            "<table-wrap><table>"
            "<tr><th>Arm</th><th>n</th></tr>"
            "<tr><td>Placebo</td><td>120</td></tr>"
            "</table></table-wrap>"
        )
        assert _render_table(wrap) == "Arm | n\nPlacebo | 120"

    def test_a_cells_inline_markup_survives(self):
        wrap = ET.fromstring(
            "<table-wrap><table><tr><td>p &lt; 0.001<sup>a</sup></td></tr></table></table-wrap>"
        )
        assert _render_table(wrap) == "p < 0.001a"

    def test_a_wholly_empty_row_is_dropped(self):
        wrap = ET.fromstring(
            "<table-wrap><table><tr><td/><td/></tr><tr><td>x</td></tr></table></table-wrap>"
        )
        assert _render_table(wrap) == "x"

    def test_a_table_with_no_rows_renders_empty(self):
        assert _render_table(ET.fromstring("<table-wrap/>")) == ""


class TestCaptionText:
    def test_a_caption_is_flattened_to_one_line(self):
        fig = ET.fromstring(
            "<fig><caption><title>Fig 1.</title><p>Survival\ncurves.</p></caption></fig>"
        )
        assert _caption_text(fig) == "Fig 1. Survival curves."

    def test_no_caption_is_an_empty_string(self):
        assert _caption_text(ET.fromstring("<fig/>")) == ""


class TestParseSections:
    def test_titled_sections_come_back_in_document_order(self):
        body = ET.fromstring(
            "<body>"
            "<sec><title>Introduction</title><p>One.</p></sec>"
            "<sec><title>Methods</title><p>Two.</p></sec>"
            "</body>"
        )
        sections = _parse_sections(body)
        assert [s["title"] for s in sections] == ["Introduction", "Methods"]
        assert [s["canonical"] for s in sections] == ["intro", "methods"]

    def test_chars_counts_the_text_it_returns(self):
        body = ET.fromstring("<body><sec><title>Methods</title><p>abcd</p></sec></body>")
        (section,) = _parse_sections(body)
        assert section["chars"] == len(section["text"]) == 4

    def test_a_body_with_no_sec_collapses_to_one_untitled_section(self):
        body = ET.fromstring("<body><p>One.</p><p>Two.</p></body>")
        (section,) = _parse_sections(body)
        assert section["title"] is None
        assert section["text"] == "One.\n\nTwo."

    def test_loose_paragraphs_beside_a_single_sec_are_not_dropped(self):
        """The PMC3030664 regression: one <sec>, so `if not secs` never fired."""
        body = ET.fromstring(
            "<body>"
            "<p>The real article body.</p>"
            "<p>A second paragraph of it.</p>"
            "<sec><title>Supplementary Material</title><p>A stub.</p></sec>"
            "</body>"
        )
        sections = _parse_sections(body)
        assert len(sections) == 2
        assert sections[0]["title"] is None
        assert "The real article body." in sections[0]["text"]
        assert sum(s["chars"] for s in sections) > 0

    def test_loose_runs_are_interleaved_in_document_order(self):
        body = ET.fromstring(
            "<body>"
            "<p>Lead.</p>"
            "<sec><title>Methods</title><p>How.</p></sec>"
            "<p>Trailing.</p>"
            "</body>"
        )
        sections = _parse_sections(body)
        assert [s["title"] for s in sections] == [None, "Methods", None]

    def test_text_before_the_first_child_is_body_text_like_any_other(self):
        body = ET.fromstring("<body>Leading prose.<p>And a paragraph.</p></body>")
        assert "Leading prose." in _parse_sections(body)[0]["text"]

    def test_an_empty_loose_run_produces_no_section(self):
        body = ET.fromstring("<body><sec><title>Methods</title><p>How.</p></sec></body>")
        assert len(_parse_sections(body)) == 1

    def test_reparenting_loose_nodes_does_not_mutate_the_real_tree(self):
        body = ET.fromstring("<body><p>One.</p><sec><title>M</title><p>Two.</p></sec></body>")
        _parse_sections(body)
        assert [child.tag for child in body] == ["p", "sec"]


class TestResolveAsset:
    def test_an_exact_name_matches(self):
        assert _resolve_asset("fig1.jpg", {"fig1.jpg": 400}) == ("fig1.jpg", 400)

    def test_an_href_without_an_extension_matches_by_stem(self):
        assert _resolve_asset("fig1", {"fig1.jpg": 400}) == ("fig1.jpg", 400)

    def test_the_stem_match_is_case_insensitive(self):
        assert _resolve_asset("FIG1.tif", {"fig1.jpg": 400}) == ("fig1.jpg", 400)

    def test_an_href_naming_nothing_deposited_resolves_to_nothing(self):
        assert _resolve_asset("missing.jpg", {"fig1.jpg": 400}) == (None, 0)

    def test_no_href_resolves_to_nothing(self):
        assert _resolve_asset(None, {"fig1.jpg": 400}) == (None, 0)


_JATS = """<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
 <body>
  <sec><title>Results</title><p>We observed an effect.</p></sec>
 </body>
 <back>
  <fig id="f1"><label>Figure 1</label>
   <caption><p>Survival curves.</p></caption>
   <graphic xlink:href="fig1.jpg"/></fig>
  <fig id="f2"><label>Figure 2</label>
   <caption><p>Never deposited.</p></caption>
   <graphic xlink:href="ghost.jpg"/></fig>
  <fig id="f3"><label>Figure 3</label>
   <caption><p>Too large.</p></caption>
   <graphic xlink:href="huge.jpg"/></fig>
  <table-wrap id="t1"><label>Table 1</label>
   <caption><p>Baseline.</p></caption>
   <table><tr><th>Arm</th></tr><tr><td>Placebo</td></tr></table></table-wrap>
  <supplementary-material><label>Data S1</label>
   <caption><p>Source data.</p></caption>
   <media xlink:href="mmc2.xlsx"/></supplementary-material>
 </back>
</article>
"""

_OBJECTS = {"fig1.jpg": 120_000, "huge.jpg": SANDBOX_READ_CAP + 1, "mmc2.xlsx": 9_000}


class TestParseJats:
    @pytest.fixture
    def parsed(self) -> dict:
        return parse_jats(_JATS.encode(), _OBJECTS)

    def test_sections_are_parsed_out_of_the_body(self, parsed):
        assert [s["canonical"] for s in parsed["sections"]] == ["results"]

    def test_a_deposited_figure_under_the_cap_is_readable(self, parsed):
        figure = next(f for f in parsed["figures"] if f["fig_id"] == "f1")
        assert figure["readable_in_sandbox"] is True
        assert figure["unavailable_reason"] is None
        assert figure["file"] == "fig1.jpg"
        assert figure["caption"] == "Survival curves."

    def test_a_figure_pmc_never_deposited_says_so_and_points_at_the_caption(self, parsed):
        """10% of 409 measured figures. Gone for good — the remedy is the caption."""
        figure = next(f for f in parsed["figures"] if f["fig_id"] == "f2")
        assert figure["readable_in_sandbox"] is False
        assert figure["file"] is None
        assert "never deposited" in figure["unavailable_reason"]

    def test_an_oversize_figure_is_a_distinct_failure_from_a_missing_one(self, parsed):
        """5% of the same sample. A real image the sandbox just cannot hand to a model."""
        figure = next(f for f in parsed["figures"] if f["fig_id"] == "f3")
        assert figure["readable_in_sandbox"] is False
        assert figure["file"] == "huge.jpg"
        assert "over the" in figure["unavailable_reason"]

    def test_tables_carry_their_rendered_rows(self, parsed):
        (table,) = parsed["tables"]
        assert table["table_id"] == "t1"
        assert table["label"] == "Table 1"
        assert table["rows"] == "Arm\nPlacebo"

    def test_supplementary_material_resolves_to_a_real_object(self, parsed):
        (supplementary,) = parsed["supplementary"]
        assert supplementary["file"] == "mmc2.xlsx"
        assert supplementary["bytes"] == 9_000
        assert supplementary["label"] == "Data S1"

    def test_a_closed_article_with_a_front_but_no_body_parses_to_no_sections(self):
        """PMC serves these as a complete <front> with nothing under it."""
        xml = b'<article xmlns:xlink="http://www.w3.org/1999/xlink"><front/></article>'
        assert parse_jats(xml, {})["sections"] == []

    def test_an_article_nested_under_a_wrapper_is_found(self):
        xml = b'<pmc-articleset><article><body><p>Text.</p></body></article></pmc-articleset>'
        assert parse_jats(xml, {})["sections"][0]["text"] == "Text."

    def test_a_payload_with_no_article_element_is_an_error(self):
        with pytest.raises(PMCError, match="no <article> element"):
            parse_jats(b"<nothing/>", {})
