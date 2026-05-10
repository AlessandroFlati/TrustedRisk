"""Unit tests for the topological branch of check_ood (cluster centroid distance)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared.abstain import _min_distance_to_centroids, check_ood
from shared import abstain


@pytest.fixture(autouse=True)
def _reset_caches():
    """Bust both lru_cache instances between tests."""
    abstain._load_ood_policy.cache_clear()
    abstain._load_cluster_centroids.cache_clear()
    yield
    abstain._load_ood_policy.cache_clear()
    abstain._load_cluster_centroids.cache_clear()


def _stage_topological_policy(tmp_path: Path, monkeypatch, threshold: float = 0.5):
    policy = {
        "policy_type": "topological_clustering",
        "ood_distance_threshold": threshold,
    }
    pol_fp = tmp_path / "policy.json"
    pol_fp.write_text(json.dumps(policy))
    monkeypatch.setenv("TRUSTEDRISK_OOD_POLICY_PATH", str(pol_fp))
    monkeypatch.setenv("TRUSTEDRISK_OOD_POLICY_TYPE", "topological")
    return pol_fp


def _stage_centroids(tmp_path: Path, monkeypatch, centroids: np.ndarray):
    npz_fp = tmp_path / "cohort_embeddings.npz"
    np.savez(npz_fp, cluster_centroids=centroids)
    monkeypatch.setenv("TRUSTEDRISK_COHORT_EMBEDDINGS_PATH", str(npz_fp))


def test_topological_close_no_trigger(tmp_path, monkeypatch):
    centroids = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    _stage_topological_policy(tmp_path, monkeypatch, threshold=0.5)
    _stage_centroids(tmp_path, monkeypatch, centroids)

    # Embedding very close to centroid 0
    result = check_ood(lace_score=8, encounter_embedding=[0.99, 0.0, 0.0])
    assert result is None


def test_topological_far_fires(tmp_path, monkeypatch):
    centroids = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    _stage_topological_policy(tmp_path, monkeypatch, threshold=0.5)
    _stage_centroids(tmp_path, monkeypatch, centroids)

    # Embedding far from all centroids
    result = check_ood(lace_score=8, encounter_embedding=[0.0, 0.0, 1.0])
    assert result is not None
    assert result.type == "out_of_distribution"
    assert result.threshold_exceeded["distance"] > 0.5


def test_topological_dim_mismatch_no_trigger(tmp_path, monkeypatch):
    centroids = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    _stage_topological_policy(tmp_path, monkeypatch)
    _stage_centroids(tmp_path, monkeypatch, centroids)

    # 2-dim embedding vs 3-dim centroid -> dim mismatch returns None silently
    assert check_ood(lace_score=10, encounter_embedding=[0.5, 0.5]) is None


def test_topological_no_centroids_file_no_trigger(tmp_path, monkeypatch):
    _stage_topological_policy(tmp_path, monkeypatch)
    monkeypatch.delenv("TRUSTEDRISK_COHORT_EMBEDDINGS_PATH", raising=False)
    # No centroids file -> distance unknown -> no trigger
    assert check_ood(lace_score=10, encounter_embedding=[0.5, 0.5, 0.5]) is None


def test_min_distance_to_centroids_picks_nearest(tmp_path, monkeypatch):
    centroids = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    npz_fp = tmp_path / "cohort_embeddings.npz"
    np.savez(npz_fp, cluster_centroids=centroids)
    monkeypatch.setenv("TRUSTEDRISK_COHORT_EMBEDDINGS_PATH", str(npz_fp))

    # Closer to first centroid
    d = _min_distance_to_centroids([0.9, 0.0])
    assert d == pytest.approx(0.1, abs=1e-5)


def test_min_distance_to_centroids_no_file_returns_none(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_COHORT_EMBEDDINGS_PATH", raising=False)
    assert _min_distance_to_centroids([0.5, 0.5]) is None
