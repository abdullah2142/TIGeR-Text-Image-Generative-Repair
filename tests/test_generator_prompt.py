"""D10/D8: what the generative fallback actually asks the diffusion model for.

`build_prompt` is deliberately free of the diffusers import so these run
without a GPU. Before D10 only colour and category reached the prompt, while
the write-up blamed SDXL for losing patterns the code never sent it.
"""

from tiger.generator import build_prompt


def test_pattern_and_material_reach_the_prompt():
    prompt, subject = build_prompt("fallback", "shirts",
                                   {"color": "red", "material": "denim", "pattern": "striped"})
    assert "red denim shirt" in subject
    assert "striped pattern" in subject
    assert subject in prompt


def test_solid_is_not_described_as_a_pattern():
    _, subject = build_prompt("fallback", "shirts", {"color": "red", "pattern": "solid"})
    assert "pattern" not in subject
    assert subject == "red shirt"


def test_category_noun_comes_from_the_shared_singular_table():
    """D8 gave every ABO category a real singular noun; the generator was still
    doing its own `removesuffix("s")`, so it asked SDXL for a "wall_art"."""
    _, subject = build_prompt("fallback", "wall_art", {"color": "blue"})
    assert subject == "blue piece of wall art"
    _, subject = build_prompt("fallback", "light_fixture", {"color": "brass"})
    assert subject == "brass light fixture"
    assert "_" not in subject


def test_falls_back_to_the_caption_when_there_is_nothing_to_describe():
    prompt, subject = build_prompt("a red shirt with buttons", "shirts", {})
    assert subject == "a red shirt with buttons"
    assert subject in prompt


def test_missing_attrs_is_not_an_error():
    prompt, subject = build_prompt("a chair", "chair", None)
    assert subject == "a chair"
    assert prompt.startswith("Professional studio product photo of a single a chair,")
