"""
Regression tests for the experiment-level invariants that carry meaning.

Each of these locks in a defect that had already reached the reported numbers or
would have crashed a run: the Hadamard identity the E2 result rests on, the
eval-time image transform, the fold splitters, and the counterfactual pairing
contract.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.beaf.vision_mechanisms import _group_kfold
from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import compute_hadamard_coordinates


class TestHadamardIdentity:
    """Delta(S) = 2*gamma - 2*max(|alpha|,|beta|) is an identity, not an approximation."""

    def test_identity_holds_on_random_similarities(self):
        rng = np.random.default_rng(42)
        s = rng.normal(size=(4, 5000)) * 0.3
        out = compute_hadamard_coordinates(*s)
        assert float(np.max(out["verification_error"])) < 1e-12

    def test_reconstruction_from_coordinates(self):
        rng = np.random.default_rng(0)
        S11, S12, S21, S22 = rng.normal(size=(4, 200)) * 0.2
        c = compute_hadamard_coordinates(S11, S12, S21, S22)
        # S_ab = C + a*beta + b*alpha + a*b*gamma with (a,b) = (image, text) in {+1,-1}
        assert np.allclose(c["C"] + c["beta"] + c["alpha"] + c["gamma"], S11)
        assert np.allclose(c["C"] - c["beta"] + c["alpha"] - c["gamma"], S12)
        assert np.allclose(c["C"] + c["beta"] - c["alpha"] - c["gamma"], S21)
        assert np.allclose(c["C"] - c["beta"] - c["alpha"] + c["gamma"], S22)

    def test_joint_correct_iff_gamma_dominates(self):
        rng = np.random.default_rng(7)
        c = compute_hadamard_coordinates(*(rng.normal(size=(4, 3000)) * 0.3))
        dominates = c["gamma"] > np.maximum(c["abs_alpha"], c["abs_beta"])
        assert np.array_equal(c["joint_correct"], dominates)


class TestGroupKFoldGuard:
    """`min(5, n_groups)` silently produced n_splits=1 for a single-group slice."""

    def test_split_count_tracks_group_count(self):
        assert _group_kfold(np.array(["a", "a", "b", "b"])).get_n_splits() == 2
        assert _group_kfold(np.array(list("abcdefg"))).get_n_splits() == 5

    def test_single_group_raises_a_message_about_groups(self):
        with pytest.raises(ValueError, match="at least 2 distinct groups"):
            _group_kfold(np.array(["a", "a", "a"]))


class TestEvalTimeImageTransform:
    """Feature extraction must use the val transform; the train one is stochastic."""

    def test_val_transform_is_deterministic_and_train_is_not(self):
        import torch
        from PIL import Image
        import open_clip

        _, pp_train, pp_val = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
        img = Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        assert torch.equal(pp_val(img), pp_val(img)), "val transform must be deterministic"
        # Two RandomResizedCrop draws can land on the same integer crop box, so a
        # single pair is a flaky test. Several draws differing at all is the claim.
        draws = [pp_train(img) for _ in range(8)]
        assert any(not torch.equal(draws[0], d) for d in draws[1:]), (
            "train transform is expected to be stochastic -- that is why extraction must not use it"
        )

    @pytest.mark.parametrize("module_name", [
        "analysis.run_analysis",
        "analysis.run_beaf_analysis_v2",
        "analysis.run_beaf_flexible_probing",
        "analysis.run_beaf_train_val_per_object",
    ])
    def test_entrypoints_unpack_the_val_transform(self, module_name):
        """These four took `model, preprocess, _`, i.e. preprocess_train, by mistake."""
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(module_name))
        assert "model, preprocess, _ = open_clip.create_model_and_transforms" not in src, (
            f"{module_name} unpacks preprocess_train for feature extraction"
        )


class TestCounterfactualPairingContract:
    """BEAF rows come in consecutive pairs; the E-series relies on that ordering."""

    CSV = "benchmarks/data/images/beaf_counterfactual_6col.csv"

    @pytest.fixture
    def df(self):
        import os
        # Resolve against the repo root so the test runs from either the root or
        # benchmarks/ (which is where `make test` invokes pytest from).
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(repo_root, self.CSV)
        if not os.path.exists(path):
            pytest.skip(f"{self.CSV} not present")
        return pd.read_csv(path)

    def test_consecutive_rows_form_valid_pairs(self, df):
        assert len(df) % 2 == 0
        a, b = df.iloc[0::2].reset_index(drop=True), df.iloc[1::2].reset_index(drop=True)
        assert (a["object_name"] == b["object_name"]).all()
        assert (a["source_template"] == b["source_template"]).all()
        assert (a["object_in_image"].astype(bool) != b["object_in_image"].astype(bool)).all()

    def test_positional_regrouping_recovers_the_same_pairs(self, df):
        """The E-series re-pairs by position within an object instead of using the loader."""
        for obj, g in df.groupby("object_name"):
            present = g[g["object_in_image"].astype(bool)].reset_index(drop=True)
            absent = g[~g["object_in_image"].astype(bool)].reset_index(drop=True)
            n = min(len(present), len(absent))
            assert (
                present["source_template"][:n].values == absent["source_template"][:n].values
            ).all(), f"positional pairing crosses templates for {obj!r}"


class TestPairedProbeCrossValidation:
    """A counterfactual pair split across folds inverts the probe it is meant to measure.

    E1's image probe reported a macro 29.06% -- 28 of 33 concepts below chance -- for
    exactly this reason: ``StratifiedKFold`` put one half of a minimal pair in train
    and the other in test, and since the two vectors are near-identical with opposite
    labels, memorising the scene assigns the held-out half its twin's label. Grouping
    the halves into one fold gives 62.65% on the same features.
    """

    @staticmethod
    def _paired_features(n=60, dim=32, eps=0.3, seed=0):
        """Scenes shared within a pair, plus a small +/- shift along one direction."""
        rng = np.random.default_rng(seed)
        scenes = rng.normal(size=(n, dim))
        d = rng.normal(size=dim)
        d /= np.linalg.norm(d)
        return scenes + eps * d, scenes - eps * d

    def test_stratified_cv_inverts_a_weak_paired_signal(self):
        from benchmarks.src.evaluation.eval_unary_mechanistic_analysis import fit_linear_probe

        X_pos, X_neg = self._paired_features(eps=0.02)
        acc, _, _, _, acc_leaky = fit_linear_probe(X_pos, X_neg, seed=0)
        assert acc_leaky < 25.0, (
            f"StratifiedKFold on paired data should invert, got {acc_leaky:.2f}%"
        )
        assert acc >= 45.0, f"pair grouping should not be inverted, got {acc:.2f}%"

    def test_pair_grouping_recovers_the_signal(self):
        from benchmarks.src.evaluation.eval_unary_mechanistic_analysis import fit_linear_probe

        X_pos, X_neg = self._paired_features(eps=0.3)
        acc, _, _, _, acc_leaky = fit_linear_probe(X_pos, X_neg, seed=0)
        assert acc > 55.0 > acc_leaky, (
            f"expected grouped {acc:.2f}% > 55% > leaky {acc_leaky:.2f}%"
        )

    def test_grouping_is_the_default_and_leaky_value_is_still_reported(self):
        from benchmarks.src.evaluation.eval_unary_mechanistic_analysis import fit_linear_probe

        X_pos, X_neg = self._paired_features(eps=0.02)
        default = fit_linear_probe(X_pos, X_neg, seed=0)
        assert len(default) == 5, "the leaky StratifiedKFold value must stay auditable"
        opted_out = fit_linear_probe(X_pos, X_neg, seed=0, paired=False)
        assert default[0] != opted_out[0]
        assert opted_out[0] == opted_out[4], "paired=False must report the stratified score"
