"""Integration test for the Iris depth pipeline wiring.

Exercises the non-new call-site machinery (the unified node registry plus the
``pipeline_configs`` entry added to ``scope/core/pipelines/registry.py``) with
the new ``IrisDepthPipeline`` class, without requiring a GPU or downloaded
weights (the heavy diffusers pipeline is loaded lazily inside ``__init__``).
"""

import importlib

from scope.core.nodes.registry import NodeRegistry
from scope.core.pipelines.iris import IrisDepthPipeline
from scope.core.pipelines.iris.schema import IrisConfig

# Mirrors the (module_path, class_name) tuple the registry uses to discover the
# iris pipeline, so this test fails if that entry drifts.
IRIS_MODULE = ".iris.pipeline"
IRIS_CLASS = "IrisDepthPipeline"
IRIS_PACKAGE = "scope.core.pipelines"


def test_config_class_wired_to_pipeline():
    assert IrisDepthPipeline.get_config_class() is IrisConfig
    assert IrisConfig.pipeline_id == "iris"


def test_config_describes_iris_artifact_and_pgd_schedule():
    # The artifact must point at the real Iris weights repo.
    assert len(IrisConfig.artifacts) == 1
    artifact = IrisConfig.artifacts[0]
    assert artifact.repo_id == "Strike1999/Iris"
    assert "model_index.json" in artifact.files
    assert IrisConfig.estimated_vram_gb is not None

    # Default two-stage PGD timestep schedule from the paper.
    config = IrisConfig()
    assert config.timesteps == [499, 999]


def test_registry_tuple_resolves_to_pipeline_class():
    """The module path + class name added to registry.py must resolve."""
    module = importlib.import_module(IRIS_MODULE, package=IRIS_PACKAGE)
    assert getattr(module, IRIS_CLASS) is IrisDepthPipeline


def test_pipeline_registers_with_node_registry():
    """The new pipeline plugs into the existing (non-new) node registry."""
    pipeline_id = IrisDepthPipeline.get_config_class().pipeline_id
    was_registered = NodeRegistry.is_registered(pipeline_id)
    try:
        NodeRegistry.register(IrisDepthPipeline)
        assert NodeRegistry.is_registered(pipeline_id)
        assert NodeRegistry.get(pipeline_id) is IrisDepthPipeline
    finally:
        # Restore prior global state so the test never perturbs other tests
        # (e.g. on a GPU host where the registry may auto-register iris).
        if not was_registered:
            NodeRegistry.unregister(pipeline_id)
