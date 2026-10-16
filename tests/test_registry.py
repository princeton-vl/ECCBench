"""Static checks of the model registry.

Model modules need GPU stacks, so only ``dummy_eval`` is imported.
"""

from pathlib import Path

import pytest

from eccbench.registry import REGISTRY, load_model_class

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SAMPLERS_DIR = SRC_DIR / "eccbench" / "samplers"

VALID_ENVS = {"full-stack", "hf448", "hf450", "api"}
VALID_KINDS = {"vlm", "api"}


@pytest.mark.parametrize("alias", sorted(REGISTRY))
def test_registry_spec_is_well_formed(alias):
    spec = REGISTRY[alias]

    assert spec.module.startswith("eccbench."), (
        f"{alias}: module {spec.module!r} must live in the eccbench package"
    )
    module_path = SRC_DIR / Path(*spec.module.split("."))
    assert module_path.with_suffix(".py").exists() or (module_path / "__init__.py").exists(), (
        f"{alias}: module {spec.module!r} has no file under src/"
    )

    assert spec.class_name, f"{alias}: class_name must be non-empty"
    assert spec.display_name, f"{alias}: display_name must be non-empty"
    assert spec.env in VALID_ENVS, f"{alias}: unknown env {spec.env!r}"
    assert spec.kind in VALID_KINDS, f"{alias}: unknown kind {spec.kind!r}"

    if spec.sampler is not None:
        parts = spec.sampler.split(".")
        assert len(parts) == 2 and all(parts), (
            f"{alias}: sampler {spec.sampler!r} must look like 'module.Class'"
        )
        sampler_file = SAMPLERS_DIR / f"{parts[0]}.py"
        assert sampler_file.exists(), (
            f"{alias}: sampler module {sampler_file} does not exist"
        )


def test_dummy_eval_loads_and_answers():
    """The integration-test model must import and run without a GPU stack."""
    model_cls = load_model_class("dummy_eval")
    assert model_cls.__name__ == "DummyEvalModel"

    model = model_cls()
    model.add_text("context", current_video_time=0.0)
    assert model.ask_question("q?") == "The answer is {0}."
    assert model.ask_question("q?") == "The answer is {1}."

    state = model.save_state()
    model.clear_context()
    model.load_state(state)
    assert model.get_state() == state
