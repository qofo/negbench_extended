"""
BEAF Single-Object & Multi-Object Generalization Analysis Script.

Executes single-object evaluations with 1:1 balanced present/absent image pairs
and 255 expanded templates (123 negative, 132 positive), repeating across objects
to assess generalization capabilities of CLIP visual and text representations.

Usage:
  python -m analysis.run_beaf_object_generalization \
      --csv_path csvOLD/beaf_counterfactual_6col.csv \
      --template_json benchmarks/data/beaf_expanded_templates.json \
      --model ViT-B-32 \
      --pretrained openai \
      --output_dir logs/evaluation/beaf_object_generalization/openai_vit_b32
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import open_clip

# Add current module path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.beaf.object_experiment import (
    instantiate_templates,
    get_balanced_beaf_object_df,
    run_single_object_analysis,
    run_leave_one_object_out_text_probe_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="BEAF Single-Object & Multi-Object Generalization Analysis")
    parser.add_argument("--csv_path", type=str, default="csvOLD/beaf_counterfactual_6col.csv", help="Path to BEAF dataset CSV")
    parser.add_argument("--template_json", type=str, default="benchmarks/data/beaf_expanded_templates.json", help="Path to expanded templates JSON")
    parser.add_argument("--image_root", type=str, default="", help="Image root directory prefix if relative paths")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights tag")
    parser.add_argument("--target_object", type=str, default="all", help="Single object name or 'all'")
    parser.add_argument("--min_pairs", type=int, default=2, help="Minimum 1:1 image pairs for object inclusion")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_object_generalization/openai_vit_b32", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("======================================================================")
    print("🚀 BEAF Single-Object & Multi-Object Generalization Experiment")
    print(f" Dataset CSV      : {args.csv_path}")
    print(f" Template JSON    : {args.template_json}")
    print(f" Model / Pretrained: {args.model} ({args.pretrained})")
    print(f" Target Object    : {args.target_object}")
    print(f" Min Pairs        : {args.min_pairs}")
    print(f" Output Directory : {args.output_dir}")
    print("======================================================================")

    # 1. Load Templates JSON
    if not os.path.exists(args.template_json):
        raise FileNotFoundError(f"Template JSON not found: {args.template_json}")
    with open(args.template_json, "r", encoding="utf-8") as f:
        template_data = json.load(f)

    print(f"Loaded {len(template_data['negative_templates'])} negative and {len(template_data['positive_templates'])} positive templates.")

    # 2. Load BEAF Data
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"BEAF CSV not found: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)
    if args.image_root:
        df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.join(args.image_root, p))
    else:
        df["abs_image_path"] = df["image_path"]

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP model {args.model} ({args.pretrained}) on device: {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.eval()

    # 4. Filter objects to run
    unique_objects = df["object_name"].str.lower().unique().tolist()
    if args.target_object != "all":
        target_objs = [args.target_object.lower()]
    else:
        target_objs = sorted(unique_objects)

    print(f"Found {len(unique_objects)} total objects in BEAF dataset. Evaluating {len(target_objs)} target objects...")

    # 5. Iterative Single-Object Experiments
    object_results: List[Dict[str, Any]] = []
    object_t_embs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for obj in target_objs:
        df_obj = get_balanced_beaf_object_df(df, obj)
        if df_obj.empty or (len(df_obj) // 2) < args.min_pairs:
            continue

        neg_prompts, pos_prompts, neg_groups, pos_groups = instantiate_templates(obj, template_data)

        res = run_single_object_analysis(
            df_balanced=df_obj,
            object_name=obj,
            neg_prompts=neg_prompts,
            pos_prompts=pos_prompts,
            neg_groups=neg_groups,
            pos_groups=pos_groups,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )

        if "error" not in res:
            object_t_embs[obj] = (res["_pos_t_emb"], res["_neg_t_emb"])
            object_results.append(res)
            print(f"  [{obj:15s}] Pairs:{res['n_present_images']:3d} | Text Probe CV:{res['text_probe_cv_acc']*100:.1f}% | Unseen Tmpl Group:{res['unseen_template_group_acc_mean']*100:.1f}% | Vision Probe:{res['vision_probe_cv_acc']*100:.1f}% | Dual Probe:{res['dual_probe_overall_acc']*100:.1f}% | ZeroShot Acc:{res['overall_accuracy']*100:.1f}%")

    if not object_results:
        print("❌ No valid objects found for analysis.")
        return

    # 6. Leave-One-Object-Out (LOOO) Cross-Object Text Linear Probing
    looo_results = {}
    if len(object_t_embs) > 1:
        print("\n⚡ Running Leave-One-Object-Out (LOOO) Cross-Object Text Linear Probe Generalization...")
        looo_results = run_leave_one_object_out_text_probe_experiment(object_t_embs)
        for res in object_results:
            obj_name = res["object_name"]
            res["unseen_object_text_probe_acc"] = looo_results["per_object_unseen_acc"].get(obj_name, None)

    # Clean up raw embeddings from results dict before saving CSV
    for res in object_results:
        res.pop("_pos_t_emb", None)
        res.pop("_neg_t_emb", None)
        res.pop("_pos_v_emb", None)
        res.pop("_neg_v_emb", None)

    # 7. Aggregate Results & Save
    res_df = pd.DataFrame(object_results)
    csv_out_path = os.path.join(args.output_dir, "per_object_results.csv")
    res_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved per-object results to {csv_out_path}")

    # Summary Stats Across Objects
    summary_stats = {
        "model": args.model,
        "pretrained": args.pretrained,
        "n_evaluated_objects": int(len(res_df)),
        "total_pairs_evaluated": int(res_df["n_present_images"].sum()),
        
        # Text Linear Probe Averages
        "macro_text_probe_cv_acc_mean": float(res_df["text_probe_cv_acc"].mean()),
        "macro_text_probe_cv_acc_std": float(res_df["text_probe_cv_acc"].std()),
        "macro_unseen_object_text_probe_acc_mean": float(looo_results.get("looo_unseen_acc_mean", 0.0)),
        "macro_unseen_object_text_probe_acc_std": float(looo_results.get("looo_unseen_acc_std", 0.0)),

        # Unseen Template Group Cross Validation Averages
        "macro_unseen_template_group_acc_mean": float(res_df["unseen_template_group_acc_mean"].mean()),
        "macro_unseen_template_group_acc_std": float(res_df["unseen_template_group_acc_mean"].std()),

        # Vision Probe & Dual Classifier Product Scorer Averages
        "macro_vision_probe_cv_acc_mean": float(res_df["vision_probe_cv_acc"].mean()),
        "macro_vision_probe_cv_acc_std": float(res_df["vision_probe_cv_acc"].std()),
        "macro_dual_probe_overall_acc_mean": float(res_df["dual_probe_overall_acc"].mean()),
        "macro_dual_probe_overall_acc_std": float(res_df["dual_probe_overall_acc"].std()),

        # Image-Text Zero-Shot Macro Averages
        "macro_pos_acc_mean": float(res_df["pos_image_accuracy"].mean()),
        "macro_pos_acc_std": float(res_df["pos_image_accuracy"].std()),
        "macro_neg_acc_mean": float(res_df["neg_image_accuracy"].mean()),
        "macro_neg_acc_std": float(res_df["neg_image_accuracy"].std()),
        "macro_overall_acc_mean": float(res_df["overall_accuracy"].mean()),
        "macro_overall_acc_std": float(res_df["overall_accuracy"].std()),

        "macro_pos_v_margin_mean": float(res_df["mean_pos_v_margin"].mean()),
        "macro_pos_v_margin_std": float(res_df["mean_pos_v_margin"].std()),

        "macro_sim_pos_v_pos_t_mean": float(res_df["mean_sim_pos_v_pos_t"].mean()),
        "macro_sim_pos_v_neg_t_mean": float(res_df["mean_sim_pos_v_neg_t"].mean()),
        "macro_sim_neg_v_pos_t_mean": float(res_df["mean_sim_neg_v_pos_t"].mean()),
        "macro_sim_neg_v_neg_t_mean": float(res_df["mean_sim_neg_v_neg_t"].mean()),

        # Weighted Micro average
        "weighted_overall_acc": float((res_df["overall_accuracy"] * res_df["n_present_images"]).sum() / res_df["n_present_images"].sum()),
    }

    json_out_path = os.path.join(args.output_dir, "overall_generalization_summary.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved generalization summary to {json_out_path}")

    print("\n======================================================================")
    print("📊 GENERALIZATION & COMPLEMENTARY PROBE EXPERIMENT SUMMARY")
    print(f" Evaluated Objects                      : {summary_stats['n_evaluated_objects']}")
    print(f" Single-Object Text Probe CV Acc        : {summary_stats['macro_text_probe_cv_acc_mean']*100:.2f}% ± {summary_stats['macro_text_probe_cv_acc_std']*100:.2f}%")
    print(f" Unseen Object Text Probe Acc (LOOO)    : {summary_stats['macro_unseen_object_text_probe_acc_mean']*100:.2f}% ± {summary_stats['macro_unseen_object_text_probe_acc_std']*100:.2f}%")
    print(f" Unseen Template Group Text Probe Acc   : {summary_stats['macro_unseen_template_group_acc_mean']*100:.2f}% ± {summary_stats['macro_unseen_template_group_acc_std']*100:.2f}%")
    print(f" Single-Object Vision Probe CV Acc      : {summary_stats['macro_vision_probe_cv_acc_mean']*100:.2f}% ± {summary_stats['macro_vision_probe_cv_acc_std']*100:.2f}%")
    print(f" Joint Dual Classifier Product Acc      : {summary_stats['macro_dual_probe_overall_acc_mean']*100:.2f}% ± {summary_stats['macro_dual_probe_overall_acc_std']*100:.2f}%")
    print(f" Zero-Shot Overall Acc (Cosine Baseline): {summary_stats['macro_overall_acc_mean']*100:.2f}% ± {summary_stats['macro_overall_acc_std']*100:.2f}%")
    print("======================================================================")


if __name__ == "__main__":
    main()


