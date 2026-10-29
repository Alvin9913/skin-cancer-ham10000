import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "./outputs"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
CLASS_ORDER = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
MALIGNANT_INDICES = [1, 3, 4]

TRACK_ORDER = ["A", "B1", "B2", "C", "B"]
TRACK_NAMES = {
    "A":  "A: CE, frozen (baseline)",
    "B1": "B1: CE + progressive FT",
    "B2": "B2: Focal + frozen",
    "C":  "C: weighted CE + progressive FT",
    "B":  "B: Focal + progressive FT"
}

def load_all_full_results():
    files = glob.glob(os.path.join(METRICS_DIR, "track_*_seed_*.json"))
    results = []
    for f in files:
        with open(f, "r") as fp:
            results.append(json.load(fp))
    print(f"[Info] 共加载了 {len(results)} 个正式实验结果文件。")
    return results

def compute_bootstrap_ci(track_a="A", track_b="B", n_boot=1000, seed=42):
    """
    严格实现报告 Section 5 承诺的 1,000 resamples of the test set 95% Bootstrap CI
    """
    rng = np.random.default_rng(seed)
    seeds = [42, 100, 2026, 7, 99]
    probs_a, probs_b = [], []
    y_true = None

    for s in seeds:
        pa_path = os.path.join(METRICS_DIR, f"preds_track_{track_a}_seed_{s}.npz")
        pb_path = os.path.join(METRICS_DIR, f"preds_track_{track_b}_seed_{s}.npz")
        if os.path.exists(pa_path) and os.path.exists(pb_path):
            da = np.load(pa_path)
            db = np.load(pb_path)
            y_true = da["y_true"]
            probs_a.append(da["y_prob"])
            probs_b.append(db["y_prob"])

    if not probs_a:
        return None

    # 对每个测试样本跨 5 个 seed 的平均预测或逐 seed 差值进行 1,000 次测试集重采样
    probs_a = np.stack(probs_a, axis=0)  # [S, N, 7]
    probs_b = np.stack(probs_b, axis=0)  # [S, N, 7]
    preds_a = np.argmax(probs_a, axis=-1) # [S, N]
    preds_b = np.argmax(probs_b, axis=-1) # [S, N]

    is_mal_true = np.isin(y_true, MALIGNANT_INDICES)
    is_mal_pred_a = np.isin(preds_a, MALIGNANT_INDICES) # [S, N]
    is_mal_pred_b = np.isin(preds_b, MALIGNANT_INDICES) # [S, N]

    mal_indices = np.where(is_mal_true)[0]
    n_mal = len(mal_indices)

    boot_gains = []
    for _ in range(n_boot):
        sample_idx = rng.choice(mal_indices, size=n_mal, replace=True)
        rec_a = is_mal_pred_a[:, sample_idx].mean() * 100.0
        rec_b = is_mal_pred_b[:, sample_idx].mean() * 100.0
        boot_gains.append(rec_b - rec_a)

    ci_low, ci_high = np.percentile(boot_gains, [2.5, 97.5])
    mean_gain = np.mean(boot_gains)
    return mean_gain, ci_low, ci_high

def generate_polished_figure_2(results):
    """
    采纳审查建议：
    1. 左右两图统一使用一致的学术配色 (Blues)，且各自配备清晰的 Colorbar (0-100%)
    2. Y 轴行标签直接标注测试集真实样本数 NV (n=1005), MEL (n=168) 等
    """
    runs_a = [r for r in results if r["track"] == "A"]
    runs_b = [r for r in results if r["track"] == "B"]
    if not runs_a or not runs_b:
        print("[Warn] 尚未找到完整的 Track A 或 Track B 结果。")
        return

    # 单个测试集的各类别真实样本数（取第一个 run 的行和）
    single_cm = np.array(runs_a[0]["test_metrics"]["confusion_matrix"])
    class_counts = single_cm.sum(axis=1)

    cm_a = np.sum([np.array(r["test_metrics"]["confusion_matrix"]) for r in runs_a], axis=0)
    cm_b = np.sum([np.array(r["test_metrics"]["confusion_matrix"]) for r in runs_b], axis=0)

    cm_a_norm = cm_a.astype(float) / np.maximum(cm_a.sum(axis=1, keepdims=True), 1) * 100.0
    cm_b_norm = cm_b.astype(float) / np.maximum(cm_b.sum(axis=1, keepdims=True), 1) * 100.0

    x_labels = [c.upper() for c in CLASS_ORDER]
    y_labels = [f"{c.upper()}\n(n={cnt})" for c, cnt in zip(CLASS_ORDER, class_counts)]

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)

    # 左图：Track A
    sns.heatmap(
        cm_a_norm, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
        xticklabels=x_labels, yticklabels=y_labels,
        cbar=True, cbar_kws={'label': 'Row Percentage (%)', 'shrink': 0.85},
        ax=axes[0], annot_kws={"size": 9.5, "weight": "bold"}, linewidths=0.5, linecolor='#e0e0e0'
    )
    axes[0].set_title(f"(a) Track A: Baseline (CE, Frozen Backbone)\nRow-Normalised Confusion Matrix (Mean over {len(runs_a)} Seeds)",
                      fontsize=11, fontweight="bold", pad=12)
    axes[0].set_xlabel("Predicted Class", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("True Class (with Test Set Sample Size)", fontsize=10.5, fontweight="bold")
    axes[0].tick_params(axis='y', rotation=0)

    # 右图：Track B (统一使用同色系 Blues 并配独立色条，消除跨图视觉误导)
    sns.heatmap(
        cm_b_norm, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
        xticklabels=x_labels, yticklabels=y_labels,
        cbar=True, cbar_kws={'label': 'Row Percentage (%)', 'shrink': 0.85},
        ax=axes[1], annot_kws={"size": 9.5, "weight": "bold"}, linewidths=0.5, linecolor='#e0e0e0'
    )
    axes[1].set_title(f"(b) Track B: Proposed (Focal Loss + Progressive FT)\nRow-Normalised Confusion Matrix (Mean over {len(runs_b)} Seeds)",
                      fontsize=11, fontweight="bold", pad=12)
    axes[1].set_xlabel("Predicted Class", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("True Class (with Test Set Sample Size)", fontsize=10.5, fontweight="bold")
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "Figure_2_Confusion_Matrices_Polished.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Success] 已生成改进版 Figure 2 (统一色标 + 样本量标注): {out_path}")

if __name__ == "__main__":
    results = load_all_full_results()
    generate_polished_figure_2(results)

    for ref_track in ["A", "B1", "B2", "C"]:
        ci_res = compute_bootstrap_ci(track_a=ref_track, track_b="B", n_boot=1000)
        if ci_res is not None:
            mean_g, low, high = ci_res
            print(f"[Bootstrap 95% CI] Track B vs Track {ref_track} (Malignant Recall Gain): "
                  f"{mean_g:+.2f}% (95% CI: [{low:+.2f}%, {high:+.2f}%])")