"""D15: the title-contradiction flag fired on brand names and on titles that
merely list several colours, more often on clean rows than on corrupted ones."""

import pytest

from tiger.schema import Schema
from tiger.sieve import _title_color

SCHEMA = Schema(
    attributes={"color": {"type": "enum",
                          "values": ["red", "blue", "brown", "gray", "white", "yellow"],
                          "aliases": {"navy": "blue", "silver": "gray",
                                      "gold": "yellow", "beige": "white"}}},
    categories=["chair", "necklace"],
    constraints=[],
)


def fires(title: str, declared: str, brand: str = "") -> bool:
    tc = _title_color(title, SCHEMA, brand)
    return bool(tc and declared and tc != declared)


# --- what it must still catch -------------------------------------------------

def test_a_single_contradicting_colour_still_fires():
    assert fires("Solid Red Dining Chair", "blue")


def test_alias_in_the_title_is_normalised_before_comparing():
    """D7: "Navy Shirt" + color=navy is agreement, not contradiction."""
    assert not fires("Navy Blue Cotton Chair", "blue")
    assert fires("Navy Blue Cotton Chair", "red")


# --- what it must stop catching ----------------------------------------------

def test_a_title_listing_several_colours_is_not_a_contradiction():
    """A rug whose title enumerates its colours is summarising, not disagreeing.
    The old first-match-wins scan picked whichever colour came first in the
    schema's iteration order and called the rest a contradiction."""
    assert not fires("Modern Geometric Wool Area Rug, Blue, Grey, Brown", "blue")
    assert not fires("Modern Geometric Wool Area Rug, Blue, Grey, Brown", "brown")


def test_metal_names_in_jewellery_titles_are_materials_not_colours():
    """"Sterling Silver" is what the necklace is made of; the colour attribute
    is describing the stone. Silver normalises to gray, so this read as a
    contradiction against every non-gray stone."""
    assert not fires("Sterling Silver Swarovski Crystal Halo Pendant Necklace", "purple")
    assert not fires("Platinum-Plated Sterling Silver Antique Rings", "white")
    assert not fires("10k Gold Round Checkerboard Cut Gemstone Stud Earrings", "purple")


def test_a_bare_metal_colour_still_counts_when_nothing_says_metal():
    """Without a metal cue, "silver" is being used as a colour and must count."""
    assert fires("Silver Dining Chair", "red")


def test_a_colour_word_in_a_product_name_is_not_excused_by_position():
    """Masking the "Amazon Brand - <Brand>" prefix positionally was tried and
    dropped -- it swallowed real colour words for no measured benefit."""
    assert fires("Amazon Brand - Rivet Red Label Chair", "blue")


def test_brand_attribute_is_still_honoured_when_present():
    assert not fires("Redwood Designs Chair", "blue", brand="Redwood Designs")
