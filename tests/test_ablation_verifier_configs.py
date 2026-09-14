"""E13/D16: the two verifiers are separate components and get separate rows.

`run_repair_ablations` took one parameter named `vlm_judge` into which the CLI
passed whichever verifier it had built. Both notebooks run `--independent`, so
every committed row labelled "No VLM Judge" ablated SigLIP and no VLM was
involved in any reported number.
"""

import inspect

from tiger.eval import repair_ablation as RA


def test_the_two_verifiers_are_separate_parameters():
    params = inspect.signature(RA.run_repair_ablations).parameters
    assert "independent" in params, "the encoder cross-check needs its own parameter"
    assert "vlm_judge" in params, "the VLM judge needs its own parameter"
    assert params["independent"].default is None
    assert params["vlm_judge"].default is None


def test_the_ablation_row_is_named_for_what_it_ablates():
    src = inspect.getsource(RA.format_repair_ablations)
    assert '"no_independent": "No Independent Verifier"' in src
    assert '"no_vlm"' not in src, "the row name must not claim a VLM was ablated"


def test_the_vlm_row_exists_and_is_ordered_before_full():
    src = inspect.getsource(RA.format_repair_ablations)
    assert '"vlm_judge": "VLM Judge (identity-aware)"' in src
    order = src[src.index("order = ["):src.index("]", src.index("order = ["))]
    assert "vlm_judge" in order and order.index("vlm_judge") < order.index("full")


def test_the_vlm_config_runs_only_when_a_judge_is_supplied():
    """It is rate-limited and costs an API call per repair, so it must not fire
    on a run that never asked for it."""
    src = inspect.getsource(RA.run_repair_ablations)
    assert "if vlm_judge is not None:" in src
    i = src.index("if vlm_judge is not None:")
    assert 'results["vlm_judge"]' in src[i:], "the VLM row must be built inside that guard"


def test_every_other_config_uses_the_encoder_verifier():
    """Only the dedicated row may use the VLM; swapping it into the reported
    configuration would change what "Full System" means."""
    src = inspect.getsource(RA.run_repair_ablations)
    head = src[:src.index("if vlm_judge is not None:")]
    assert "independent=vlm_judge" not in head
    assert head.count("independent=independent") >= 4
