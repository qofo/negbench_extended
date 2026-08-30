"""
Regression tests for the experiment-level invariants that carry meaning.

Each of these locks in a defect that had already reached the reported numbers or
would have crashed a run: the Hadamard identity the E2 result rests on, the
eval-time image transform, the fold splitters, and the counterfactual pairing
contract.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

# `make test` runs pytest from benchmarks/, other invocations from the repo root.
# Resolve the tree to scan from this file instead, so cwd cannot silently empty it.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "benchmarks" / "src"

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


class TestTieAwarePrediction:
    """A tie at the maximum must not be resolved to option 0.

    Every row of the NegBench MCQ CSVs carries ``correct_answer = 0`` -- the options
    are never shuffled -- so ``torch.argmax``'s lowest-index tie-break scores a
    scorer that ranks nothing at exactly 100%. Zeroing the image embedding does
    exactly that to the three image-linear heads, which is where the reported
    "zero vision = 100.00%" came from.
    """

    def test_all_tied_row_is_flagged_and_not_forced_to_zero(self):
        import torch
        from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report

        scores = torch.zeros(400, 4)
        preds, ties = predict_with_tie_report(scores, seed=0)
        assert ties.all(), "an all-equal row must be reported as a tie"
        assert len(set(preds.tolist())) > 1, "ties must not all collapse onto option 0"
        assert (preds == 0).mean() < 0.5, "tie-breaking must not be biased toward index 0"

    def test_unique_maximum_is_untouched(self):
        import torch
        from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report

        scores = torch.tensor([[0.1, 0.9, 0.2, 0.3], [0.7, 0.1, 0.2, 0.0]])
        preds, ties = predict_with_tie_report(scores, seed=0)
        assert list(preds) == [1, 0]
        assert not ties.any()

    def test_same_seed_gives_the_same_tie_break(self):
        import torch
        from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report

        scores = torch.zeros(50, 4)
        a, _ = predict_with_tie_report(scores, seed=7)
        b, _ = predict_with_tie_report(scores, seed=7)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("head", ["cosine", "weighted_cosine", "bilinear"])
    def test_zeroed_vision_reads_as_chance_not_perfect(self, head):
        """The regression itself: these three heads tie on every row without an image."""
        import torch
        from benchmarks.src.evaluation.scoring_heads import build_scorer, predict_with_tie_report

        torch.manual_seed(0)
        n, k, d = 400, 4, 32
        imgs = torch.zeros(n, d)
        texts = torch.randn(n, k, d)
        targets = np.zeros(n, dtype=int)  # NegBench MCQ: the answer is always option 0

        with torch.no_grad():
            scores = build_scorer(head, d).eval()(imgs, texts)

        preds, ties = predict_with_tie_report(scores, seed=0)
        assert ties.mean() == 1.0, f"{head} should tie on every row when the image is zeroed"
        acc = float((preds == targets).mean())
        assert acc < 0.5, f"{head} scored {acc:.0%} on pure ties; the old argmax gave 100%"


class TestAlignmentInterventionDeterminism:
    """
    The alignment-intervention script trains three matchers per concept. Two start
    from ``torch.eye`` and are deterministic on their own, but the low-rank head
    draws ``proj_v``/``proj_t`` from torch's global RNG (``nn.init.normal_``). The
    script never called ``set_seed``, so that one condition produced a different
    matrix -- and a different accuracy -- on every run, while its summary JSON
    recorded a ``seed`` that controlled none of it.
    """

    @staticmethod
    def _quad(seed=0, n=24, d=64):
        import torch
        rng = np.random.RandomState(seed)
        mk = lambda: torch.tensor(rng.randn(n, d) / np.sqrt(d), dtype=torch.float32)
        return mk(), mk(), mk(), mk()

    def test_lowrank_matcher_is_seed_dependent(self):
        """Without seeding, the low-rank condition is genuinely non-reproducible."""
        import torch
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            train_lowrank_bilinear_matcher,
        )

        quad = self._quad()
        torch.seed()
        w1 = train_lowrank_bilinear_matcher(*quad, rank=8, epochs=20)
        torch.seed()
        w2 = train_lowrank_bilinear_matcher(*quad, rank=8, epochs=20)
        assert not np.allclose(w1, w2), (
            "expected the unseeded low-rank matcher to differ between runs; if this "
            "now passes, the init changed and the set_seed guarantee needs rechecking"
        )

    def test_set_seed_makes_every_condition_reproducible(self):
        """With set_seed, all three trained conditions repeat exactly."""
        from analysis.config import set_seed
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            train_lowrank_bilinear_matcher, train_bilinear_matcher, train_labclip_matcher,
        )

        quad = self._quad()
        for train, kw in ((train_lowrank_bilinear_matcher, dict(rank=8, epochs=20)),
                          (train_bilinear_matcher, dict(epochs=20)),
                          (train_labclip_matcher, dict(epochs=20))):
            set_seed(42)
            a = train(*quad, **kw)
            set_seed(42)
            b = train(*quad, **kw)
            assert np.array_equal(a, b), f"{train.__name__} is not reproducible under set_seed"

    def test_default_run_stays_in_sample_and_says_so(self):
        """
        The default path is still in-sample -- held-out scoring is opt-in behind
        ``--oof`` because it costs ~5x the runtime. Both facts have to keep reaching
        the reader, so the summary must carry the protocol note either way.
        """
        import inspect
        from benchmarks.src.evaluation import eval_per_object_alignment_intervention as mod

        run_src = inspect.getsource(mod.run_per_object_alignment_intervention)
        assert "evaluation_protocol" in run_src, "the in-sample caveat must stay in provenance"
        assert "oof: bool = False" in inspect.getsource(mod.run_per_object_alignment_intervention), \
            "held-out scoring must stay opt-in; flipping the default silently changes every number"
        assert "GroupKFold" in inspect.getsource(mod.evaluate_conditions_out_of_fold), (
            "the OOF path must group folds, not split rows at random"
        )


class TestAlignmentInterventionOutOfFold:
    """
    The alignment-intervention script scores in-sample by default. ``--oof`` adds a
    held-out column, grouped on the base image so two counterfactual pairs cut from
    one photo cannot straddle a split. Conditions 1 (identity) and 7 (a random
    rotation drawn from a fixed seed) fit nothing, so their held-out accuracy must
    equal their in-sample accuracy exactly -- that equality is what proves the
    harness routes scores to the right rows.
    """

    @staticmethod
    def _concept(n=40, d=48, seed=0):
        rng = np.random.RandomState(seed)
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        dv, dt = unit(rng.randn(d)), unit(rng.randn(d))
        base_v, base_t = rng.randn(n, d), rng.randn(n, d)
        v_p, v_m = unit(base_v + dv), unit(base_v - dv)
        t_p, t_m = unit(base_t + dt), unit(base_t - dt)
        groups = np.repeat(np.arange(n // 2), 2)  # two pairs share each base scene
        return v_p, v_m, t_p, t_m, groups

    def test_parameter_free_conditions_match_in_sample(self):
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            build_condition_matrices, evaluate_condition_scoring,
            evaluate_conditions_out_of_fold,
        )

        v_p, v_m, t_p, t_m, groups = self._concept()
        d = v_p.shape[1]
        d_I = np.ones(d) / np.sqrt(d)
        d_T = np.ones(d) / np.sqrt(d)

        oof, n_folds = evaluate_conditions_out_of_fold(
            v_p, v_m, t_p, t_m, groups, rank=4, embed_dim=d, seed=42, use_bias=True)
        assert n_folds == 5

        mats = build_condition_matrices(v_p, v_m, t_p, t_m, d_I, d_T,
                                        rank=4, embed_dim=d, seed=42)
        for cond in ("1_Baseline_Cosine", "7_Control_Random_Rotation"):
            A, is_lab = mats[cond]
            in_sample = evaluate_condition_scoring(v_p, v_m, t_p, t_m, A, d_I, d_T, is_lab)
            assert in_sample["acc_joint_pct"] == pytest.approx(oof[cond]["acc_joint_pct"]), (
                f"{cond} fits nothing, so held-out scoring must reproduce it exactly"
            )

    def test_base_scene_never_straddles_a_fold(self):
        """The grouping contract: no base image appears in both halves of a split."""
        from sklearn.model_selection import GroupKFold

        _, _, _, _, groups = self._concept()
        for tr, te in GroupKFold(n_splits=5).split(np.arange(len(groups)), groups=groups):
            assert not (set(groups[tr]) & set(groups[te])), "a base scene leaked across the split"

    def test_free_matrix_scores_lower_out_of_fold(self):
        """
        A 512x512-shaped free matrix fitted on noise should collapse out of fold.
        This is the defect the OOF column exists to expose.
        """
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            build_condition_matrices, evaluate_condition_scoring,
            evaluate_conditions_out_of_fold,
        )

        rng = np.random.RandomState(1)
        n, d = 40, 48
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        v_p, v_m, t_p, t_m = (unit(rng.randn(n, d)) for _ in range(4))  # pure noise
        groups = np.repeat(np.arange(n // 2), 2)
        d_I = d_T = np.ones(d) / np.sqrt(d)

        cond = "6_Learned_Full_Bilinear"
        A, is_lab = build_condition_matrices(v_p, v_m, t_p, t_m, d_I, d_T,
                                             rank=4, embed_dim=d, seed=42)[cond]
        fit = evaluate_condition_scoring(v_p, v_m, t_p, t_m, A, d_I, d_T, is_lab)["acc_joint_pct"]
        oof, _ = evaluate_conditions_out_of_fold(
            v_p, v_m, t_p, t_m, groups, rank=4, embed_dim=d, seed=42, use_bias=True)

        assert fit > oof[cond]["acc_joint_pct"], (
            f"on label-free noise the free matrix scored {fit:.1f}% in-sample and "
            f"{oof[cond]['acc_joint_pct']:.1f}% out of fold; the in-sample number "
            "should be the inflated one"
        )


class TestRotationZeroAlphaCondition:
    """
    Conditions 8 and 9 test the "alignment + gap orthogonalisation" row of the
    additive model: rotate the text so cos(d_I, R d_T) = 1, then project the text
    polarity vector off the image mean so the text main effect alpha vanishes.
    Condition 9 is the same projection without the rotation -- the control that makes
    a gain attributable to the combination rather than to removing alpha alone.
    """

    @staticmethod
    def _quads(n=40, d=32, seed=3):
        rng = np.random.RandomState(seed)
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        dv, dt = unit(rng.randn(d)), unit(rng.randn(d))
        bv, bt = rng.randn(n, d), rng.randn(n, d)
        return (unit(bv + 0.3 * dv), unit(bv - 0.3 * dv),
                unit(bt + 0.3 * dt), unit(bt - 0.3 * dt), unit(dv), unit(dt))

    def test_hadamard_reconstruction_is_exact(self):
        """
        Conditions 8 and 9 recover scores from Hadamard coordinates. The transform is
        its own inverse up to a factor of 4; if that ever stops holding, both
        conditions silently report the wrong numbers.
        """
        from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
            compute_hadamard_coordinates,
        )

        rng = np.random.RandomState(0)
        S11, S12, S21, S22 = (rng.randn(50) for _ in range(4))
        h = compute_hadamard_coordinates(S11, S12, S21, S22)
        C, a, b, g = h["C"], h["alpha"], h["beta"], h["gamma"]

        assert np.allclose(C + b + a + g, S11)
        assert np.allclose(C - b + a - g, S12)
        assert np.allclose(C + b - a - g, S21)
        assert np.allclose(C - b - a + g, S22)

    @pytest.mark.parametrize("rotate", [True, False])
    def test_alpha_is_removed_exactly(self, rotate):
        """Both conditions exist to zero the text main effect; verify they do."""
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            rotation_zero_alpha_scores, metrics_from_quad_scores,
        )

        v_p, v_m, t_p, t_m, d_I, d_T = self._quads()
        scores, align = rotation_zero_alpha_scores(v_p, v_m, t_p, t_m, d_I, d_T, rotate=rotate)
        m = metrics_from_quad_scores(*scores, align)

        assert m["abs_alpha_mean"] < 1e-9, (
            f"alpha survived the projection at {m['abs_alpha_mean']:.3e}; the "
            "intervention's whole claim is that it does not"
        )

    def test_rotation_reaches_perfect_alignment_and_the_control_does_not(self):
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            rotation_zero_alpha_scores,
        )

        v_p, v_m, t_p, t_m, d_I, d_T = self._quads()
        _, rotated = rotation_zero_alpha_scores(v_p, v_m, t_p, t_m, d_I, d_T, rotate=True)
        _, control = rotation_zero_alpha_scores(v_p, v_m, t_p, t_m, d_I, d_T, rotate=False)

        assert rotated == pytest.approx(1.0, abs=1e-6), "R is built so that R d_T = d_I"
        assert control == pytest.approx(float(np.dot(d_I, d_T)), abs=1e-6), (
            "the control applies no rotation, so alignment must stay at the raw value"
        )

    def test_conditions_8_and_9_survive_the_oof_harness(self):
        """Both are closed-form given the probe normals, so they must be OOF-scorable."""
        from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
            evaluate_conditions_out_of_fold, CONDITION_NAMES,
        )

        v_p, v_m, t_p, t_m, _, _ = self._quads()
        groups = np.repeat(np.arange(len(v_p) // 2), 2)
        oof, _ = evaluate_conditions_out_of_fold(
            v_p, v_m, t_p, t_m, groups, rank=4, embed_dim=v_p.shape[1],
            seed=42, use_bias=True)

        assert set(oof) == set(CONDITION_NAMES), "every condition must reach the OOF column"
        for cond in ("8_Rotation_Zero_Alpha", "9_Zero_Alpha_Only"):
            assert oof[cond]["abs_alpha_mean"] < 1e-9, f"{cond} must zero alpha out of fold too"


class TestUpstreamArtifactResolution:
    """
    Several experiments read another experiment's output through a default path
    under logs/, which is gitignored -- so a fresh clone has none of them. The
    three call sites used to handle that three different ways: a clear error, a
    raw pandas traceback, and a silent skip that dropped a cross-check from the
    report with nothing recording the omission.
    """

    def test_missing_required_artifact_names_the_producing_command(self, tmp_path):
        from analysis.feature_cache import resolve_upstream_artifact

        with pytest.raises(FileNotFoundError) as e:
            resolve_upstream_artifact(str(tmp_path / "absent.csv"),
                                      produced_by="python -m some.producer")
        msg = str(e.value)
        assert "python -m some.producer" in msg, "the error must say how to produce the file"
        assert "gitignored" in msg, "and why a fresh clone lacks it"

    def test_optional_artifact_returns_none_instead_of_raising(self, tmp_path, capsys):
        from analysis.feature_cache import resolve_upstream_artifact

        got = resolve_upstream_artifact(str(tmp_path / "absent.csv"),
                                        produced_by="python -m some.producer",
                                        required=False)
        assert got is None
        assert "skipped" in capsys.readouterr().out, "an optional skip must still be visible"

    def test_existing_artifact_passes_through(self, tmp_path):
        from analysis.feature_cache import resolve_upstream_artifact

        p = tmp_path / "present.csv"
        p.write_text("a,b\n1,2\n")
        assert resolve_upstream_artifact(str(p), produced_by="x") == str(p)

    def test_hadamard_summary_records_whether_the_e1_check_ran(self):
        """
        The silent-skip case: the summary has to distinguish a report whose E1
        cross-check ran from one where the CSV was simply absent.
        """
        import inspect
        from benchmarks.src.evaluation import eval_e2_hadamard_decomposition as mod

        src = inspect.getsource(mod)
        assert '"e1_cross_check"' in src, "the summary must record whether the E1 check ran"
        assert "SKIPPED" in src, "and say so explicitly when it did not"


class TestProbeRegistry:
    """
    A probe used to be declared in three places that had to agree by hand: the
    SUPPORTED_PROBES name list, an elif chain for its hyperparameter grid, and
    another for its constructor. Nothing checked the agreement, so a probe added
    to two of the three would raise "Unsupported probe_type" from the one that was
    missed -- or silently get tuned against another probe's grid.
    """

    def test_name_list_is_derived_from_the_registry(self):
        from analysis.beaf.probe_factory import SUPPORTED_PROBES, PROBE_REGISTRY

        assert SUPPORTED_PROBES == list(PROBE_REGISTRY), (
            "SUPPORTED_PROBES must be derived from PROBE_REGISTRY, not maintained beside it"
        )

    def test_every_registered_probe_has_a_grid_and_builds(self):
        from analysis.beaf.probe_factory import (
            SUPPORTED_PROBES, get_param_candidates, create_probe_classifier,
        )

        for name in SUPPORTED_PROBES:
            grid = get_param_candidates(name)
            assert grid, f"{name} has an empty hyperparameter grid"
            for params in (grid[0], grid[-1]):
                assert create_probe_classifier(name, seed=42, **params) is not None

    def test_unknown_probe_fails_the_same_way_from_both_entry_points(self):
        from analysis.beaf.probe_factory import get_param_candidates, create_probe_classifier

        for fn in (get_param_candidates, create_probe_classifier):
            with pytest.raises(ValueError, match="Unsupported probe_type"):
                fn("no_such_probe")

    def test_no_bias_reaches_the_probes_that_can_honor_it(self):
        """`--no_bias` is a claim about the fitted model, so it has to land."""
        from analysis.beaf.probe_factory import create_probe_classifier

        for name in ("logistic", "ridge", "sgd_log", "sgd_hinge"):
            clf = create_probe_classifier(name, seed=42, fit_intercept=False)
            assert clf.get_params()["fit_intercept"] is False, f"{name} ignored fit_intercept"

    def test_svm_probes_warn_that_they_cannot_honor_no_bias(self, capsys):
        """sklearn's SVC has no fit_intercept; reporting a 'no-bias' SVM number would lie."""
        import analysis.beaf.probe_factory as pf

        pf._NO_BIAS_WARNED.discard("svm_rbf")
        pf.create_probe_classifier("svm_rbf", seed=42, fit_intercept=False)
        assert "NOT applied" in capsys.readouterr().out


class TestPyTorchProbeSklearnContract:
    """
    PyTorchProbeEstimator set ``classes_`` to [0, 1] in ``__init__``. In the
    sklearn contract ``classes_`` is a fitted attribute read from the data, and
    hardcoding it meant any other label pair was cast straight into BCE while
    predict() still returned 0/1 -- scoring 0% on perfectly separable data with
    nothing raised. Every experiment here happens to use {0, 1}, so the bug was
    invisible until a probe was reused on differently-labelled data.
    """

    @staticmethod
    def _separable(seed=0, n=40, d=8):
        rng = np.random.RandomState(seed)
        X = np.vstack([rng.randn(n, d) + 1.5, rng.randn(n, d) - 1.5])
        return X, np.array([1] * n + [0] * n)

    @pytest.mark.parametrize("labels", [(0, 1), (2, 5), ("neg", "pos")])
    def test_any_binary_label_pair_scores_correctly(self, labels):
        from analysis.beaf.probe_factory import create_probe_classifier

        X, y01 = self._separable()
        y = np.where(y01 == 1, labels[1], labels[0])
        clf = create_probe_classifier("mlp", seed=42, epochs=60)
        clf.fit(X, y)

        assert clf.score(X, y) > 0.9, (
            f"labels {labels} scored {clf.score(X, y):.0%} on separable data"
        )
        assert set(np.unique(clf.predict(X))) <= set(labels), "predict must return the caller's labels"
        assert list(clf.classes_) == sorted(labels, key=str)

    def test_classes_is_not_set_before_fit(self):
        from analysis.beaf.probe_factory import create_probe_classifier

        clf = create_probe_classifier("mlp", seed=42)
        assert not hasattr(clf, "classes_"), "classes_ is a fitted attribute, not a constructor one"

    def test_predict_before_fit_raises_instead_of_crashing_on_none(self):
        from analysis.beaf.probe_factory import create_probe_classifier

        X, _ = self._separable()
        with pytest.raises(ValueError, match="not fitted"):
            create_probe_classifier("mlp", seed=42).predict(X)

    def test_non_binary_target_is_rejected(self):
        from analysis.beaf.probe_factory import create_probe_classifier

        rng = np.random.RandomState(0)
        X = rng.randn(60, 8)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        with pytest.raises(ValueError, match="binary probe"):
            create_probe_classifier("mlp", seed=42, epochs=5).fit(X, y)


class TestSharedModelLoader:
    """
    ``create_model_and_transforms`` returns (model, preprocess_train,
    preprocess_val). The train transform is a stochastic RandomResizedCrop, so an
    entrypoint that grabs the wrong element makes every embedding a different
    sample of the same image. Four entrypoints once did exactly that, and the rest
    each repeated the same three-line incantation by hand. load_clip_for_eval is
    the one place that unpacking happens.
    """

    E_SERIES = [
        "eval_e1_minimal_pair_auc",
        "eval_e1_placebo_test",
        "eval_e2_hadamard_decomposition",
        "eval_e2_final_gamma_resolution",
        "eval_e2_sanity_check_and_grounding",
        "eval_unary_mechanistic_analysis",
        "eval_per_object_alignment_intervention",
        "eval_4condition_decomposition",
        "eval_probe_failure_inspector",
    ]

    def test_loader_returns_the_deterministic_transform_and_an_eval_model(self):
        import torch
        from PIL import Image
        from analysis.model_loader import load_clip_for_eval

        model, preprocess, tokenizer = load_clip_for_eval("ViT-B-32", None, device="cpu")
        assert model.training is False, "extraction must run with the model in eval mode"

        img = Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
        assert torch.equal(preprocess(img), preprocess(img)), (
            "the loader handed back a stochastic transform, i.e. preprocess_train"
        )
        assert tokenizer(["a photo of a cat"]).shape[0] == 1

    @pytest.mark.parametrize("module_name", E_SERIES)
    def test_e_series_does_not_hand_roll_the_unpack(self, module_name):
        import importlib
        import inspect

        mod = importlib.import_module(f"benchmarks.src.evaluation.{module_name}")
        src = inspect.getsource(mod)
        assert "load_clip_for_eval" in src, f"{module_name} should load through the shared helper"
        assert "create_model_and_transforms" not in src, (
            f"{module_name} still unpacks create_model_and_transforms itself, which is how "
            f"preprocess_train gets picked up by accident"
        )


class TestSharedCLIGroups:
    """
    Thirty-nine entrypoints each re-declared the same flags -- --model in 29 of
    them, --seed in 25 -- with nothing tying the copies together. That is why
    --use_cache and --restrict_objects reached only 6 and 7 call sites while the
    older flags were everywhere: a flag added later only lands where someone
    remembers to add it.
    """

    E_SERIES = TestSharedModelLoader.E_SERIES

    @pytest.mark.parametrize("module_name", E_SERIES)
    def test_e_series_declares_standard_flags_through_the_shared_groups(self, module_name):
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(f"benchmarks.src.evaluation.{module_name}"))
        for flag in ('"--model"', '"--pretrained"', '"--output_dir"', '"--batch_size"'):
            assert f"add_argument({flag}" not in src.replace(" ", ""), (
                f"{module_name} still declares {flag} itself instead of using the shared group"
            )
        assert "from benchmarks.src.analysis.cli import" in src

    def test_min_pairs_has_no_hidden_default(self):
        """
        min_pairs silently selects the population: the defaults across this repo are
        6, 10 and 20, and the paper's 33-concept set only appears at 20 -- the E2
        decomposition's own default of 10 yields 53 concepts instead. So the shared
        group refuses to supply one.
        """
        import argparse
        import inspect
        from analysis.cli import add_concept_args

        assert inspect.signature(add_concept_args).parameters["min_pairs"].default \
            is inspect.Parameter.empty, "min_pairs must stay a required argument"

        p = argparse.ArgumentParser()
        add_concept_args(p, 20)
        assert p.parse_args([]).min_pairs == 20
        assert "33-concept set requires 20" in p.format_help()

    def test_groups_can_skip_flags_a_script_does_not_honor(self):
        """An accepted flag that changes nothing is worse than an absent one."""
        import argparse
        from analysis.cli import add_run_args, add_data_args

        p = argparse.ArgumentParser()
        add_run_args(p, "logs/x", seed=None, batch_size=None)
        add_data_args(p, csv_path=None)
        got = vars(p.parse_args([]))
        assert set(got) == {"output_dir", "image_root"}, got


class TestPackageLayering:
    """
    ``analysis/`` holds the shared primitives -- config, paths, feature cache,
    model loader, CLI groups, probes, extractors -- and ``evaluation/`` builds
    experiments on top of them: 90 import edges run that way. One file ran the
    other way, ``analyze_internal_weights.py``, which imported
    ``evaluation.scoring_heads`` while using nothing from ``analysis`` at all. It
    now lives in ``evaluation/``, where its dependencies already were.
    """

    @staticmethod
    def _imports(pkg):
        import io
        import re

        out = []
        for f in (SRC_ROOT / pkg).rglob("*.py"):
            src = io.open(f, encoding="utf-8").read()
            for m in re.finditer(
                    r"from\s+(?:benchmarks\.src\.)?(\w+)[\w\.]*\s+import|"
                    r"import\s+(?:benchmarks\.src\.)?(\w+)\.", src):
                out.append((str(f.relative_to(REPO_ROOT)), m.group(1) or m.group(2)))
        assert out, f"no imports found under {SRC_ROOT / pkg}; the scan root is wrong"
        return out

    def test_analysis_never_imports_evaluation(self):
        bad = [f for f, tgt in self._imports("analysis") if tgt == "evaluation"]
        assert not bad, (
            "analysis/ is the lower layer; these files invert it: " + ", ".join(sorted(set(bad)))
        )

    def test_evaluation_still_builds_on_analysis(self):
        """The guard above must not be satisfied by severing the intended direction."""
        good = [f for f, tgt in self._imports("evaluation") if tgt == "analysis"]
        assert len(set(good)) > 15, (
            f"only {len(set(good))} evaluation modules import analysis; the shared primitives "
            "should be reused, not re-implemented"
        )


class TestDualPathImports:
    """
    Some entrypoints are importable two ways -- ``benchmarks.src.evaluation.x``
    from the repo root, or ``evaluation.x`` via the editable install -- and carry
    a try/except pair of import blocks for it. A bare ``except ImportError``
    catches far more than that: a typo, a name renamed on one side, a circular
    import. The two branches had in fact drifted, so several names existed only
    in the first block and a standalone run died on NameError.
    """

    @staticmethod
    def _dual_blocks(path):
        """Yield (try_names, except_names) for each try/except ImportError pair."""
        import ast
        import io

        tree = ast.parse(io.open(path, encoding="utf-8").read())

        def bound(stmts):
            out = set()
            for st in ast.walk(ast.Module(body=list(stmts), type_ignores=[])):
                if isinstance(st, (ast.Import, ast.ImportFrom)):
                    for a in st.names:
                        out.add(a.asname or a.name.split(".")[0])
            return out

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for h in node.handlers:
                name = getattr(h.type, "id", None)
                if name == "ImportError":
                    yield bound(node.body), bound(h.body)

    @staticmethod
    def _dual_path_files():
        import io

        out = []
        for f in SRC_ROOT.rglob("*.py"):
            if "open_clip" in f.parts or "training" in f.parts:
                continue
            s = io.open(f, encoding="utf-8").read()
            if "except ImportError:" in s and "benchmarks.src." in s:
                out.append(str(f))
        assert out, f"no dual-path modules found under {SRC_ROOT}; the scan root is wrong"
        return sorted(out)

    def test_both_branches_bind_the_same_names(self):
        drifted = []
        for f in self._dual_path_files():
            for tnames, enames in self._dual_blocks(f):
                # the guard helper is expected only in the fallback branch
                enames = enames - {"reraise_unless_standalone"}
                if tnames != enames:
                    drifted.append((f, sorted(tnames ^ enames)))
        assert not drifted, (
            "a name is bound in only one import branch, so one invocation "
            "convention will fail at runtime: " + repr(drifted)
        )

    def test_fallback_branches_are_guarded(self):
        unguarded = []
        for f in self._dual_path_files():
            for _, enames in self._dual_blocks(f):
                if "reraise_unless_standalone" not in enames:
                    unguarded.append(f)
        assert not unguarded, (
            "a bare except ImportError swallows real failures; call "
            "reraise_unless_standalone() first in: " + repr(sorted(set(unguarded)))
        )

    def test_guard_reraises_when_benchmarks_is_importable(self):
        from analysis.import_compat import reraise_unless_standalone

        with pytest.raises(ImportError):
            try:
                from benchmarks.src.analysis.no_such_module import thing  # noqa: F401
            except ImportError:
                reraise_unless_standalone()

        with pytest.raises(ImportError):
            try:
                from benchmarks.src.analysis.feature_cache import no_such_name  # noqa: F401
            except ImportError:
                reraise_unless_standalone()


class TestLayerKeyScheme:
    """
    A transformer block had three names. `extract_all_features_unified` returned
    the very same array as ``layers["Layer 1"]`` and ``pipeline["Layer1"]``, and
    `PipelineStep` supplied a third scheme for the steps around them. Nothing
    crashed on the mismatch: `get_layer_features` answers an unknown key with the
    final embedding, so a consumer holding the wrong spelling drew a flat
    "layerwise" curve. `config.layer_key` is now the only producer of the name.
    """

    def test_formatter_covers_the_embedding_and_the_blocks(self):
        from analysis.config import layer_key

        assert layer_key(0) == "Embedding"
        assert [layer_key(i) for i in (1, 9, 12)] == ["Layer 1", "Layer 9", "Layer 12"]

    def test_extractor_emits_one_scheme_per_dict(self):
        from analysis.config import PipelineStep, layer_key
        from analysis.extractor import extract_all_features_unified
        from analysis.model_loader import load_clip_for_eval

        model, _, tokenizer = load_clip_for_eval("ViT-B-32", None, device="cpu")
        res = extract_all_features_unified(
            model, tokenizer, ["a photo of a cat", "a photo with no cat"],
            device="cpu", batch_size=2)

        n_blocks = len(model.transformer.resblocks)
        assert list(res["layers"]) == [layer_key(i) for i in range(n_blocks + 1)]
        assert set(res["pipeline"]) == {s.value for s in PipelineStep}, (
            "pipeline carried a second spelling of the blocks; they belong to "
            "'layers' and were bit-identical copies"
        )

    def test_no_module_spells_a_layer_name_inline(self):
        import io
        import re

        # `layer_key` itself, and prose/labels, are exempt: only f-string or
        # concatenated *construction* of the key from an index is a second producer.
        pattern = re.compile(r'f"Layer ?\{|"Layer ?" ?\+ ?str\(')
        offenders = []
        for f in SRC_ROOT.rglob("*.py"):
            if "open_clip" in f.parts or f.name == "config.py":
                continue
            for i, line in enumerate(io.open(f, encoding="utf-8"), 1):
                if pattern.search(line):
                    offenders.append(f"{f.relative_to(REPO_ROOT)}:{i}")
        assert not offenders, (
            "build layer names with analysis.config.layer_key, not inline: "
            + ", ".join(offenders)
        )


class TestNestedFeatureMasking:
    """
    `filter_vision_dict` named ``"layers"`` as the one nested dict to mask. The
    text extractor's dict nests ``"pipeline"`` the same way, and it fell to the
    passthrough branch -- returned whole while every sibling array was subset, so
    rows no longer lined up across the dict.
    """

    def test_every_nested_dict_is_masked(self):
        from analysis.config import filter_vision_dict

        vis = {
            "layers": {"Layer 1": np.arange(20).reshape(10, 2)},
            "pipeline": {"Step4_Final_L2Norm": np.arange(30).reshape(10, 3)},
            "final_l2norm": np.arange(10).reshape(10, 1),
            "loaded_flags": np.ones(10, dtype=bool),
            "model_name": "ViT-B-32",
        }
        mask = np.zeros(10, dtype=bool)
        mask[[1, 4, 7]] = True
        out = filter_vision_dict(vis, mask)

        assert out["layers"]["Layer 1"].shape == (3, 2)
        assert out["pipeline"]["Step4_Final_L2Norm"].shape == (3, 3)
        assert out["final_l2norm"].shape == (3, 1)
        assert out["model_name"] == "ViT-B-32"
        assert np.array_equal(out["pipeline"]["Step4_Final_L2Norm"],
                              vis["pipeline"]["Step4_Final_L2Norm"][mask])


class TestPipelineStepsAreRequired:
    """
    The layerwise probe looked each post-block step up with a fallback to a legacy
    spelling and then added it only ``if key in pipeline_dict``. A step the
    extractor stopped emitting would silently vanish from the report and its figure.
    """

    def test_missing_step_raises_instead_of_dropping_a_curve(self):
        import benchmarks.src.evaluation.eval_layerwise_linear_probe as mod

        def fake_extract(**kwargs):
            return {
                "layers": {"Embedding": np.zeros((2, 4))},
                "pipeline": {"Step2_Layer12_LN": np.zeros((2, 4))},
            }

        original = mod.extract_all_features_unified
        mod.extract_all_features_unified = fake_extract
        try:
            with pytest.raises(KeyError) as excinfo:
                mod.extract_layerwise_feature_dict(None, None, ["x"], device="cpu")
        finally:
            mod.extract_all_features_unified = original

        message = str(excinfo.value)
        assert "Step3_Projected_Unnorm" in message and "Step4_Final_L2Norm" in message


class TestHeadlessPlotting:
    """
    These scripts run on GPU nodes with no display. pyplot picks its backend at
    import time, so the backend has to be selected *before* the import -- which is
    why every plotting module carries `matplotlib.use("Agg")`. Four modules carried
    only the pyplot line and worked by luck: pyplot falls back to Agg when DISPLAY
    is unset, but under X11 forwarding it would try to open a window instead.
    """

    @staticmethod
    def _pyplot_modules():
        import ast
        import io

        out = []
        for f in SRC_ROOT.rglob("*.py"):
            if "open_clip" in f.parts:
                continue
            src = io.open(f, encoding="utf-8").read()
            if "matplotlib.pyplot" not in src:
                continue
            out.append((f, ast.parse(src)))
        return out

    def test_every_pyplot_import_is_preceded_by_a_backend_selection(self):
        import ast

        offenders = []
        for path, tree in self._pyplot_modules():
            pyplot_line = backend_line = None
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name == "matplotlib.pyplot" and pyplot_line is None:
                            pyplot_line = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    # `from analysis.plotting import plt` -- the module that owns the
                    # backend; importing plt from it is already ordered correctly.
                    if node.module and node.module.endswith("analysis.plotting"):
                        backend_line = 0
                elif isinstance(node, ast.Call):
                    fn = node.func
                    if (isinstance(fn, ast.Attribute) and fn.attr == "use"
                            and isinstance(fn.value, ast.Name) and fn.value.id == "matplotlib"):
                        backend_line = min(node.lineno, backend_line if backend_line is not None else node.lineno)
            if pyplot_line is not None and (backend_line is None or backend_line > pyplot_line):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, (
            'select the backend before importing pyplot -- matplotlib.use("Agg") first, '
            "or import plt from analysis.plotting: " + ", ".join(sorted(offenders))
        )

    def test_shared_module_selects_agg(self):
        import matplotlib

        from analysis.plotting import plt  # noqa: F401

        assert matplotlib.get_backend().lower() == "agg"


class TestTopObjectsGridHasOneBody:
    """
    Three modules defined `render_top_objects_grid` with the same body -- 95%
    identical by normalized AST -- differing only in title, filename, and, in one
    copy, where the legend sat. The drawing lives in `analysis.plotting` now and
    the three keep only their own title and path.
    """

    CALLERS = [
        "benchmarks/src/evaluation/eval_negation_existence_probe.py",
        "benchmarks/src/analysis/run_beaf_train_val_per_object.py",
        "benchmarks/src/analysis/run_beaf_flexible_probing.py",
    ]

    def test_callers_delegate_instead_of_drawing(self):
        import ast
        import io

        redrawn = []
        for rel in self.CALLERS:
            tree = ast.parse(io.open(REPO_ROOT / rel, encoding="utf-8").read())
            for node in tree.body:
                if not (isinstance(node, ast.FunctionDef) and node.name == "render_top_objects_grid"):
                    continue
                body = ast.dump(node)
                if "subplots" in body or "savefig" in body:
                    redrawn.append(rel)
        assert not redrawn, (
            "these still draw the grid themselves instead of calling "
            "analysis.plotting.render_top_objects_grid: " + ", ".join(redrawn)
        )

    def test_shared_renderer_writes_the_requested_path(self, tmp_path):
        from analysis.plotting import render_top_objects_grid

        rng = np.random.default_rng(0)
        layers = ["Embedding"] + [f"Layer {i}" for i in range(1, 13)]
        df = pd.DataFrame([
            dict(object_name=o, layer_name=L, n_pairs=40 - 3 * k,
                 train_acc_pct=float(rng.uniform(70, 99)),
                 val_acc_pct=float(rng.uniform(45, 85)))
            for k, o in enumerate(["dog", "cat", "car", "tree", "boat"])
            for L in layers
        ])
        out = tmp_path / "nested" / "grid.png"
        assert render_top_objects_grid(df, str(out), "title", top_k=4) == str(out)
        assert out.exists() and out.stat().st_size > 0

    def test_empty_frame_is_skipped_not_crashed(self):
        from analysis.plotting import render_top_objects_grid

        empty = pd.DataFrame(columns=["object_name", "layer_name", "n_pairs",
                                      "train_acc_pct", "val_acc_pct"])
        assert render_top_objects_grid(empty, "/nonexistent/x.png", "t") is None
