"""Re-key the AB-swap counterfactual file so the concept unit is the asserted object.

`beaf_counterfactual_ab_swap_diverse.csv` carries `object_name` as the *pair* of objects
("truck, train"), which leaves 1,032 groups of two or three pairs each -- far below the
`--min_pairs 20` floor every E2 run uses. The concept the 2x2 block is actually about is
`object_a`: the object the positive caption asserts and the negative caption denies, and
the one whose presence `object_in_image` tracks. Grouping on it yields 42 concepts.

The output is schema A (DATA_SCHEMA.md), so `eval_e2_hadamard_decomposition.py` consumes it
unmodified. The row order of the input is preserved, which keeps the consecutive-pair
contract `beaf_loader.load_and_verify_counterfactual_pairs` enforces.

Usage (from the repo root):
    python -m benchmarks.src.data_generation.make_ab_swap_by_concept
"""
import pandas as pd

SRC = "benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv"
DST = "benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv"


def main() -> None:
    df = pd.read_csv(SRC)
    out = pd.DataFrame({
        "image_path": df["image_path"],
        "object_name": df["object_a"],
        "positive_caption": df["positive_caption"],
        "negative_caption": df["negative_caption"],
        "object_in_image": df["object_in_image"],
        "source_template": df["source_template"],
    })

    # object_in_image must track object_a, not object_b -- verify rather than assume.
    a_present = df["object_a_present"].astype(str).str.lower() == "true"
    in_image = out["object_in_image"].astype(str).str.lower() == "true"
    assert (a_present == in_image).all(), "object_in_image does not track object_a"

    out.to_csv(DST, index=False)
    print(f"wrote {DST}: {len(out)} rows, {out.object_name.nunique()} concepts")


if __name__ == "__main__":
    main()
