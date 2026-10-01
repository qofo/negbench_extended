"""Put every number of the camera-ready paper next to its from-scratch re-run value.

Reads the run folder written by benchmarks/scripts/rerun_paper.sh and the logs the accepted numbers
came from, and writes one row per reported quantity:

    item, quantity, paper (as printed), original (full precision from the original log),
    rerun, rerun - original, rerun rounded as the paper prints it == paper

Beyond the printed numbers it compares the trained matrices themselves (Ours rank-32 W and the
LABCLIP W): max |W_rerun - W_original| says whether training reproduced bit for bit or only
approximately, which decides how far the downstream Table 3 numbers can move.

Missing re-run outputs are reported as missing, so the script can be run while stages are pending.

Usage (repo root):
    python -m benchmarks.src.evaluation.compare_paper_rerun --rerun logs/evaluation/01_paper/2026-09-19_rerun
Outputs:
    <rerun>/compare_paper_rerun.csv
"""
import argparse
import json
import os

import pandas as pd
import torch

P = "logs/evaluation/01_paper"
A = "logs/evaluation/02_archive"
ORIG = {
    "t1_openai": f"{P}/2026-08-20_probe_failure_inspection",
    "t1_random": f"{A}/2026-08-26_probe_failure_inspection_rerun",
    "t1_negft": f"{A}/2026-08-27_probe_failure_cc12m_negfull",
    "t2": f"{P}/2026-08-29_r7_rotation_zeroalpha_33concepts_bias/per_object_intervention_summary.json",
    "f2": f"{P}/2026-09-18_single_w_abswap_delta_ws_rank1to512/single_w_summary.json",
    "star": f"{P}/2026-09-10_labclip_rank_strategy_ext/rank_strategy_report.json",
    "ours_ckpt": f"{P}/2026-09-02_narrow_rank32_w_checkpoints/delta_warmstart_rank32.pt",
    "labclip_ckpt": f"{P}/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt",
    "labclip_report": f"{P}/2026-09-01_labclip_official_recipe/train_report.json",
    "fig1": "paper_figures/fig1_example_pair.json",
}
T3_ORIG = {  # (benchmark, method) -> zero_shot_transfer_results.json holding it under "Pretrained_bilinear"
    ("COCO", "Ours"): f"{P}/2026-09-02_coco_mcq_canonical_tiefixed/naive",
    ("COCO", "LABCLIP"): f"{P}/2026-09-02_coco_mcq_canonical_tiefixed/labclip",
    ("VOC2007", "Ours"): f"{P}/2026-09-02_voc_mcq_tiefixed/naive",
    ("VOC2007", "LABCLIP"): f"{P}/2026-09-02_voc_mcq_tiefixed/labclip",
    ("CheXpert", "Ours"): f"{P}/2026-09-02_chexpert_mcq/naive",
    ("CheXpert", "LABCLIP"): f"{P}/2026-09-01_labclip_official_recipe/eval_chexpert_binary_mcq_control_valid_only",
}
T3_CSV = {"COCO": "COCO_val_mcq_llama3.1_rephrased.csv", "VOC2007": "VOC2007_mcq_llama3.1_rephrased.csv",
          "CheXpert": "chexpert_binary_mcq_control_valid_only.csv"}
T3_CKPT = {"Ours": "delta_warmstart_rank32.pt", "LABCLIP": "labclip_official_recipe_bilinear.pt"}
T3_PAPER = {  # Avg (= total accuracy), Pos, Neg as printed in Table 3
    ("COCO", "Cosine"): (39.30, 69.14, 6.84), ("COCO", "LABCLIP"): (47.02, 82.23, 6.42),
    ("COCO", "Ours"): (45.25, 61.52, 24.22),
    ("VOC2007", "Cosine"): (38.72, 82.55, 3.35), ("VOC2007", "LABCLIP"): (40.95, 84.75, 6.46),
    ("VOC2007", "Ours"): (51.88, 73.61, 22.40),
    ("CheXpert", "Cosine"): (55.61,), ("CheXpert", "LABCLIP"): (37.43,), ("CheXpert", "Ours"): (58.82,),
}
F2_PAPER_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def common_objects(run_dir):
    """The objects both probes evaluated (image and text), as recorded by the inspector's report."""
    j = jload(f"{run_dir}/probe_failure_comprehensive_report.json")
    return j["concept_sets"]["common_objects"] if j else None


def probe_macro(run_dir, objects=None):
    """Table 1 cell: per-object validation accuracy at the final L2-normalised layer, averaged over objects
    (optionally restricted to ``objects``)."""
    out = {}
    for modality, layer in (("vision", "+Final L2Norm"), ("text", "Final (L2 Normed)")):
        path = f"{run_dir}/beaf_{modality}_per_object_layerwise.csv"
        if not os.path.exists(path):
            out[modality] = None
            continue
        df = pd.read_csv(path)
        sub = df[df.layer_name == layer]
        if objects is not None:
            sub = sub[sub.object_name.isin(objects)]
        assert len(sub), (path, layer)
        out[modality] = sub.val_acc_pct.mean() / 100
    return out


def w_of(path):
    return torch.load(path, map_location="cpu")["state_dict"]["W"].double() if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", default="logs/evaluation/01_paper/2026-09-19_rerun")
    args = ap.parse_args()
    R = args.rerun
    rows = []

    def add(item, quantity, paper, orig, rerun, digits):
        rows.append(dict(item=item, quantity=quantity, paper=paper, original=orig, rerun=rerun,
                         rerun_minus_original=None if rerun is None or orig is None else rerun - orig,
                         rerun_prints_as_paper=None if rerun is None or paper is None
                         else round(rerun, digits) == round(paper, digits)))

    # Table 1, on the AB-swap population: image probes use the 42 AB-swap objects, text probes the 58 objects with
    # >= 20 captions per class, and the table reports the 42 objects both share (the population of Sec 2.2,
    # Figure 2 and Table 3 Ours). The original is the Sep 1 run (logs/.../2026-09-01_probing_abswap42).
    for key, tag in (("t1_random", "random"), ("t1_openai", "openai"), ("t1_negft", "ccnegfull")):
        orig_dir = f"{P}/2026-09-01_probing_abswap42/{tag}"
        objs = common_objects(orig_dir)
        o = probe_macro(orig_dir, objs)
        r = probe_macro(f"{R}/{key.replace('t1_', 't1_abswap_')}", objs)
        add("Table 1", f"{key} image (42 common objects)", None, o["vision"], r["vision"], 2)
        add("Table 1", f"{key} text (42 common objects)", None, o["text"], r["text"], 2)
        r58 = probe_macro(f"{R}/{key.replace('t1_', 't1_abswap_')}")
        add("Table 1 (all valid objects)", f"{key} text (58 objects)", None, probe_macro(orig_dir)["text"],
            r58["text"], 2)

    # Table 2
    o, r = jload(ORIG["t2"]), jload(f"{R}/t2_rotation/per_object_intervention_summary.json")
    for cond, label, p_cos, p_acc in (("1_Baseline_Cosine", "cosine", 0.219, 1.35),
                                      ("2_Closed_Form_Rotation", "rotation", 1.000, 3.25),
                                      ("7_Control_Random_Rotation", "random rotation", 0.002, 0.12)):
        acc_key = "acc_joint_mean_pct" if cond == "1_Baseline_Cosine" else "oof_acc_joint_mean_pct"
        add("Table 2", f"{label} cos(d_I, R d_T)", p_cos, o[cond]["direction_alignment_mean"],
            r[cond]["direction_alignment_mean"] if r else None, 3)
        add("Table 2", f"{label} 2x2 acc %", p_acc, o[cond][acc_key], r[cond][acc_key] if r else None, 2)
    if r:
        add("Table 2", "n objects / pairs", 33, o["provenance"]["n_concepts_analyzed"],
            r["provenance"]["n_concepts_analyzed"], 0)

    # Sec 2.2 and Figure 2
    o, r = jload(ORIG["f2"]), jload(f"{R}/f2_rank_sweep/single_w_summary.json")
    fo = {f["family"]: f for f in o["families"]}
    fr = {f["family"]: f for f in r["families"]} if r else {}
    add("Sec 2.2", "cosine 2x2 acc %", 4.03, fo["identity"]["oof_pooled_acc_pct"],
        fr["identity"]["oof_pooled_acc_pct"] if r else None, 2)
    if r:
        add("Sec 2.2", "n pairs", 2480, o["n_pairs"], r["n_pairs"], 0)
    for rank in F2_PAPER_RANKS:
        fam = f"lowrank_{rank}"
        for key, label in (("oof_pooled_acc_pct", "unseen"), ("in_sample_acc_pct", "training")):
            add("Figure 2", f"r={rank} {label} %", round(fo[fam][key], 2), fo[fam][key],
                fr[fam][key] if r else None, 2)

    # Figure 2 star and the trained matrices
    o = [x for x in jload(ORIG["star"])["conditions"]["A_posthoc_svd"] if x["rank"] == 512][0]["delta_s"]
    r = jload(f"{R}/f2_labclip_star/delta_s_report.json")
    add("Figure 2", "LABCLIP star %", 10.04, o, r["labclip"] if r else None, 2)
    for name, orig_path, new_path in (
            ("Ours rank-32 W", ORIG["ours_ckpt"], f"{R}/ours_rank32/delta_warmstart_rank32.pt"),
            ("LABCLIP W", ORIG["labclip_ckpt"], f"{R}/labclip/labclip_official_recipe_bilinear.pt")):
        wo, wr = w_of(orig_path), w_of(new_path)
        add("Trained W", f"{name} ||W||_F", None, wo.norm().item(), wr.norm().item() if wr is not None else None, 2)
        add("Trained W", f"{name} max|W_rerun - W_orig|", None, 0.0,
            (wr - wo).abs().max().item() if wr is not None else None, 6)
    o, r = jload(ORIG["labclip_report"]), jload(f"{R}/labclip/train_report.json")
    for k in ("n_pairs", "final_loss", "W_minus_I_frobenius"):
        add("LABCLIP training", k, None, o[k], r[k] if r else None, 4)

    # Table 3
    r = jload(f"{R}/t3_mcq/mcq_by_ckpt.json")
    metrics = ("total_accuracy", "positive_accuracy", "negative_accuracy")
    for bench in ("COCO", "VOC2007", "CheXpert"):
        for method in ("Cosine", "LABCLIP", "Ours"):
            src = T3_ORIG[(bench, "Ours")] if method == "Cosine" else T3_ORIG[(bench, method)]
            o = jload(f"{src}/zero_shot_transfer_results.json")
            o = o["Baseline Cosine"] if method == "Cosine" else o["Pretrained_bilinear"]
            rr = None
            if r:
                block = r[T3_CSV[bench]]
                rr = block["cosine"] if method == "Cosine" else block[T3_CKPT[method]]
            for metric, paper in zip(metrics, T3_PAPER[(bench, method)]):
                add("Table 3", f"{bench} {method} {metric.split('_')[0]}", paper, o[metric],
                    rr[metric] if rr else None, 2)

    # Figure 1
    o, r = jload(ORIG["fig1"]), jload(f"{R}/figures/fig1_example_pair.json")
    for k in ("Spp", "Spm", "Smp", "Smm", "delta"):
        add("Figure 1", k, round(o[k], 4 if k != "delta" else 3), o[k], r[k] if r else None,
            4 if k != "delta" else 3)

    df = pd.DataFrame(rows)
    out = f"{R}/compare_paper_rerun.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.6g}"):
        print(df.to_string(index=False))
    done = df.rerun.notna()
    printed = df[done & df.paper.notna()]
    print(f"\n{done.sum()}/{len(df)} quantities re-run; "
          f"{int((printed.rerun_prints_as_paper == True).sum())}/{len(printed)} print as the paper does")
    print("saved:", out)


if __name__ == "__main__":
    main()
