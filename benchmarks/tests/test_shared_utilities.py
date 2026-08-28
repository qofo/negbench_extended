"""
Regression tests for the shared single-source-of-truth helpers.

These cover behaviors that were previously duplicated or silently wrong across
entrypoints, so a regression shows up here instead of as a quietly different
number in one experiment's report.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.config import (
    FINAL_L2NORM_KEY,
    PRE_PROJECTION_KEY,
    coerce_bool_column,
    filter_vision_dict,
    get_layer_features,
    l2_normalize,
    to_bool,
)
from analysis.paths import resolve_image_path


class TestToBoolAndColumnCoercion:
    """`object_in_image` had 13 inlined parsers that did not agree with each other."""

    @pytest.mark.parametrize("value", ["True", "true", " TRUE ", "1", "t", "yes", "y", True, 1])
    def test_truthy_spellings(self, value):
        assert to_bool(value) is True

    @pytest.mark.parametrize("value", ["False", "false", "0", "f", "no", "n", False, 0])
    def test_falsy_spellings(self, value):
        assert to_bool(value) is False

    def test_unrecognized_falls_back_to_default(self):
        assert to_bool(None, default=True) is True
        assert to_bool(None, default=False) is False

    def test_column_coercion_matches_pandas_native_bools(self):
        """The current CSVs are written True/False, so the fix must not move any number."""
        df = pd.DataFrame({"object_in_image": [True, False, True]})
        expected = df["object_in_image"].tolist()
        coerce_bool_column(df, "object_in_image")
        assert df["object_in_image"].tolist() == expected

    def test_column_coercion_handles_one_zero_encoding(self):
        """A 1/0-encoded CSV used to read as all-False under the old inlined parser."""
        df = pd.DataFrame({"object_in_image": ["1", "0", "yes", "no"]})
        coerce_bool_column(df, "object_in_image")
        assert df["object_in_image"].tolist() == [True, False, True, False]

    def test_missing_column_is_a_noop(self):
        df = pd.DataFrame({"other": [1, 2]})
        coerce_bool_column(df, "object_in_image")
        assert list(df.columns) == ["other"]


class TestGetLayerFeatures:
    """An unknown key silently returned the final embedding, faking a layerwise curve."""

    @pytest.fixture
    def vis(self):
        return {
            "layers": {"Embedding": np.ones((3, 768)), "Layer 1": np.full((3, 768), 2.0)},
            "pre_proj": np.full((3, 768), 3.0),
            "final_l2norm": np.full((3, 512), 4.0),
        }

    def test_known_layer(self, vis):
        assert get_layer_features(vis, "Layer 1")[0, 0] == 2.0

    def test_pre_projection_sentinel(self, vis):
        assert get_layer_features(vis, PRE_PROJECTION_KEY)[0, 0] == 3.0

    def test_final_sentinel_is_silent(self, vis, capsys):
        assert get_layer_features(vis, FINAL_L2NORM_KEY)[0, 0] == 4.0
        assert "WARNING" not in capsys.readouterr().out

    def test_unknown_key_still_falls_back_but_warns(self, vis, capsys):
        assert get_layer_features(vis, "Layer 99")[0, 0] == 4.0
        assert "unknown layer key" in capsys.readouterr().out


class TestResolveImagePath:
    """Four E1/E2 entrypoints carried byte-identical private copies of this."""

    def test_absolute_path_untouched(self):
        assert resolve_image_path("/abs/x.jpg", "root") == "/abs/x.jpg"

    def test_relative_joined_when_it_exists(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x.jpg").write_bytes(b"")
        assert resolve_image_path("sub/x.jpg", str(tmp_path)) == str(tmp_path / "sub" / "x.jpg")

    def test_unresolvable_returns_the_csv_path_as_written(self):
        assert resolve_image_path("missing/x.jpg", "/no/such/root") == "missing/x.jpg"


class TestGeometryHelpers:
    def test_l2_normalize_unit_norm(self):
        x = np.random.default_rng(0).normal(size=(5, 16))
        assert np.allclose(np.linalg.norm(l2_normalize(x), axis=-1), 1.0, atol=1e-6)

    def test_l2_normalize_survives_a_zero_vector(self):
        assert np.all(np.isfinite(l2_normalize(np.zeros((1, 8)))))

    def test_filter_vision_dict_masks_every_array(self):
        vis = {
            "layers": {"Layer 1": np.arange(8).reshape(4, 2)},
            "pre_proj": np.arange(8).reshape(4, 2),
            "final_l2norm": np.arange(8).reshape(4, 2),
            "loaded_flags": np.array([True, False, True, False]),
            "scalar_meta": "unchanged",
        }
        out = filter_vision_dict(vis, np.array([True, False, True, False]))
        assert out["layers"]["Layer 1"].shape[0] == 2
        assert out["pre_proj"].shape[0] == 2
        assert out["final_l2norm"].shape[0] == 2
        assert out["scalar_meta"] == "unchanged"
