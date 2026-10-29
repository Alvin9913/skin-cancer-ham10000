import os
import argparse
import json
import time
import random
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    recall_score, precision_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 动态加载 02_core_modules.py
spec = importlib.util.spec_from_file_location("core_modules", "./02_core_modules.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

CLASS_ORDER = core.CLASS_ORDER
MALIGNANT_INDICES = core.MALIGNANT_INDICES
get_transforms = core.get_transforms
HAM10000Dataset = core.HAM10000Dataset
FocalLoss = core.FocalLoss
compute_inverse_freq_weights = core.compute_inverse_freq_weights
SkinCancerResNet50 = core.SkinCancerResNet50
run_unit_tests = core.run_unit_tests

# ======================= 全局实验超参数配置 =======================
OUTPUT_DIR = "./outputs"
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

ALL_SEEDS = [42, 100, 2026, 7, 99]
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRACK_CONFIGS = {
    "A":  {"name": "A: CE, frozen (baseline)",         "loss": "ce",          "backbone": "frozen",      "gamma": 0.0},
    "B1": {"name": "B1: CE + progressive FT",          "loss": "ce",          "backbone": "progressive", "gamma": 0.0},
    "B2": {"name": "B2: Focal + frozen",               "loss": "focal",       "backbone": "frozen",      "gamma": 2.0},
    "C":  {"name": "C: weighted CE + progressive FT",  "loss": "weighted_ce", "backbone": "progressive", "gamma": 0.0},
    "B":  {"name": "B: Focal + progressive FT",        "loss": "focal",       "backbone": "progressive", "gamma": 2.0},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def evaluate_predictions(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(7)), zero_division=0) * 100.0
    mel_recall = float(per_class_recall[1])

    y_true_mal = np.isin(y_true, MALIGNANT_INDICES).astype(int)
    y_pred_mal = np.isin(y_pred, MALIGNANT_INDICES).astype(int)
    y_prob_mal = y_prob[:, MALIGNANT_INDICES].sum(axis=1)

    mal_recall = recall_score(y_true_mal, y_pred_mal, zero_division=0) * 100.0
    mal_precision = precision_score(y_true_mal, y_pred_mal, zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_true_mal, y_pred_mal, labels=[0, 1]).ravel()
    mal_specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    mal_f1 = f1_score(y_true_mal, y_pred_mal, zero_division=0) * 100.0

    y_true_onehot = np.eye(7)[y_true]
    try:
        macro_auroc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr') * 100.0
        macro_prauc = average_precision_score(y_true_onehot, y_prob, average='macro') * 100.0
        mal_auroc = roc_auc_score(y_true_mal, y_prob_mal) * 100.0
        mal_prauc = average_precision_score(y_true_mal, y_prob_mal) * 100.0
    except ValueError:
        macro_auroc, macro_prauc, mal_auroc, mal_prauc = 0.0, 0.0, 0.0, 0.0

    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(7)))

    return {
        "accuracy": float(acc),
        "malignant_recall": float(mal_recall),
        "malignant_precision": float(mal_precision),
        "malignant_specificity": float(mal_specificity),
        "malignant_f1": float(mal_f1),
        "melanoma_recall": float(mel_recall),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "macro_auroc": float(macro_auroc),
        "macro_prauc": float(macro_prauc),
        "malignant_auroc": float(mal_auroc),
        "malignant_prauc": float(mal_prauc),
        "per_class_recall": {CLASS_ORDER[i]: float(per_class_recall[i]) for i in range(7)},
        "confusion_matrix": cm_raw.tolist()
    }


@torch.no_grad()
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    use_amp = (DEVICE.type == 'cuda')
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=-1)

        total_loss += loss.item() * images.size(0)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = evaluate_predictions(all_targets, all_probs)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, all_targets, all_probs


def build_criterion(loss_type, alpha_weights, gamma=2.0):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted_ce":
        return FocalLoss(alpha=alpha_weights, gamma=0.0)
    elif loss_type == "focal":
        return FocalLoss(alpha=alpha_weights, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def train_single_run(track_key, seed, train_loader, val_loader, test_loader, alpha_weights, max_epochs=18, patience=6, prefix=""):
    cfg = TRACK_CONFIGS[track_key]
    metric_path = os.path.join(METRICS_DIR, f"{prefix}track_{track_key}_seed_{seed}.json")
    ckpt_path = os.path.join(CKPT_DIR, f"{prefix}track_{track_key}_seed_{seed}.pt")

    if prefix == "" and os.path.exists(metric_path):
        print(f"[Skip] Track {track_key} | Seed {seed} 已存在正式评估结果，直接读取。")
        with open(metric_path, "r") as f:
            return json.load(f)

    set_seed(seed)
    model = SkinCancerResNet50(num_classes=7, dropout_p=0.3, pretrained=True).to(DEVICE)
    criterion = build_criterion(cfg["loss"], alpha_weights, gamma=cfg["gamma"]).to(DEVICE)
    use_amp = (DEVICE.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    current_stage = None
    optimizer = None
    best_selection_score = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    t0 = time.time()
    print(f"\n>>> 开始训练 [{prefix or 'FULL'}] Track {track_key} ({cfg['name']}) | Seed {seed} <<<")

    for epoch in range(1, max_epochs + 1):
        # Demo 模式下第 2 个 epoch 直接测一下解冻，验证渐进式解冻逻辑无误
        effective_epoch = 4 if (prefix == "demo_" and epoch == 2) else epoch
        new_stage, param_groups = model.set_trainable_stage(cfg["backbone"], epoch=effective_epoch)
        if new_stage != current_stage:
            current_stage = new_stage
            optimizer = torch.optim.Adam(param_groups)
            print(f"  [Epoch {epoch:02d}] 切换微调阶段 -> {current_stage}")

        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion)

        selection_score = 0.5 * val_metrics["macro_f1"] + 0.5 * val_metrics["malignant_recall"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mal_recall": val_metrics["malignant_recall"],
            "val_mel_recall": val_metrics["melanoma_recall"]
        })

        is_best = selection_score > best_selection_score
        if is_best:
            best_selection_score = selection_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:02d}/{max_epochs:02d} | TrainLoss: {train_loss:.4f} | "
              f"ValAcc: {val_metrics['accuracy']:.1f}% | ValMalRec: {val_metrics['malignant_recall']:.1f}% | "
              f"ValMelRec: {val_metrics['melanoma_recall']:.1f}% | ValMacroF1: {val_metrics['macro_f1']:.1f}%"
              f"{' [*BEST*]' if is_best else ''}")

        if patience_counter >= patience and epoch >= 10:
            print(f"  [Early Stopping] 验证集指标连续 {patience} 轮未提升，在 Epoch {epoch} 提前停止 (Best Epoch={best_epoch})。")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    test_metrics, y_true_test, y_prob_test = evaluate_model(model, test_loader, criterion)
    elapsed = time.time() - t0

    np.savez_compressed(
        os.path.join(METRICS_DIR, f"{prefix}preds_track_{track_key}_seed_{seed}.npz"),
        y_true=y_true_test, y_prob=y_prob_test
    )

    result_record = {
        "track": track_key,
        "track_name": cfg["name"],
        "seed": seed,
        "best_epoch": best_epoch,
        "elapsed_sec": round(elapsed, 1),
        "epoch_1_val": history[0],
        "test_metrics": test_metrics,
        "history": history
    }
    with open(metric_path, "w") as f:
        json.dump(result_record, f, indent=2)

    print(f">>> 完成 Track {track_key} | Seed {seed} (耗时 {elapsed:.1f}s) | "
          f"Test Acc: {test_metrics['accuracy']:.2f}% | Test MalRecall: {test_metrics['malignant_recall']:.2f}% | "
          f"Test MelRecall: {test_metrics['melanoma_recall']:.2f}% | Test MacroF1: {test_metrics['macro_f1']:.2f}% <<<")
    return result_record


def summarize_and_generate_outputs(all_results, prefix=""):
    print("\n" + "=" * 95)
    print(f"📊 正在汇总 [{prefix or 'FULL'}] 实验结果并生成报告表格与 Figure 2...")
    print("=" * 95)

    df_rows = []
    for r in all_results:
        tm = r["test_metrics"]
        df_rows.append({
            "track": r["track"],
            "track_name": r["track_name"],
            "seed": r["seed"],
            "accuracy": tm["accuracy"],
            "malignant_recall": tm["malignant_recall"],
            "malignant_precision": tm["malignant_precision"],
            "malignant_specificity": tm["malignant_specificity"],
            "melanoma_recall": tm["melanoma_recall"],
            "macro_f1": tm["macro_f1"],
            "balanced_acc": tm["balanced_acc"],
            "macro_auroc": tm["macro_auroc"],
            "macro_prauc": tm["macro_prauc"],
            "malignant_auroc": tm["malignant_auroc"],
            "malignant_prauc": tm["malignant_prauc"],
        })
    res_df = pd.DataFrame(df_rows)
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"{prefix}all_test_results.csv"), index=False)

    track_a_runs = [r for r in all_results if r["track"] == "A"]
    if track_a_runs:
        a_acc = np.mean([r["test_metrics"]["accuracy"] for r in track_a_runs])
        a_mel = np.mean([r["test_metrics"]["melanoma_recall"] for r in track_a_runs])
        a_mal = np.mean([r["test_metrics"]["malignant_recall"] for r in track_a_runs])
        print(f"\n📌 [Section 2 填空数据 - Where early modelling is succeeding or failing]:")
        print(f"   Baseline Overall Accuracy = {a_acc:.1f}% | Melanoma Recall = {a_mel:.1f}% | Malignant Group Recall = {a_mal:.1f}%")

    print("\n" + "=" * 105)
    print("📋 Table 2 结果汇总 (Mean ± SD, unit: %):")
    print("=" * 105)
    header = f"{'Configuration':<34} | {'Accuracy':<13} | {'Malignant rec.':<14} | {'Melanoma rec.':<13} | {'Macro-F1':<13} | {'Balanced acc.':<13}"
    print(header)
    print("-" * 105)

    for t_key in ["A", "B1", "B2", "C", "B"]:
        sub = res_df[res_df["track"] == t_key]
        if len(sub) == 0:
            continue
        name = TRACK_CONFIGS[t_key]["name"]
        def fmt(col):
            sd = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
            return f"{sub[col].mean():.1f} ± {sd:.1f}"
        print(f"{name:<34} | {fmt('accuracy'):<13} | {fmt('malignant_recall'):<14} | "
              f"{fmt('melanoma_recall'):<13} | {fmt('macro_f1'):<13} | {fmt('balanced_acc'):<13}")
    print("=" * 105)

    print("\n📋 补充指标 (Precision / Specificity / PR-AUC / AUROC):")
    print(f"{'Track':<8} | {'Mal. Precision':<15} | {'Mal. Specificity':<16} | {'Mal. PR-AUC':<14} | {'Macro AUROC':<14} | {'Macro PR-AUC':<14}")
    print("-" * 90)
    for t_key in ["A", "B1", "B2", "C", "B"]:
        sub = res_df[res_df["track"] == t_key]
        if len(sub) == 0:
            continue
        def fmt(col):
            sd = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
            return f"{sub[col].mean():.1f} ± {sd:.1f}"
        print(f"{t_key:<8} | {fmt('malignant_precision'):<15} | {fmt('malignant_specificity'):<16} | "
              f"{fmt('malignant_prauc'):<14} | {fmt('macro_auroc'):<14} | {fmt('macro_prauc'):<14}")

    sub_a = res_df[res_df["track"] == "A"].sort_values("seed")
    sub_b = res_df[res_df["track"] == "B"].sort_values("seed")
    if len(sub_a) == len(sub_b) and len(sub_a) >= 2:
        diff_mal = sub_b["malignant_recall"].values - sub_a["malignant_recall"].values
        t_stat, p_val = stats.ttest_rel(sub_b["malignant_recall"].values, sub_a["malignant_recall"].values)
        print(f"\n📌 [统计检验 Track B vs Track A on Malignant Recall]:")
        print(f"   平均提升 (Mean Gain) = +{diff_mal.mean():.2f}% (SD = {diff_mal.std(ddof=1):.2f}%), Paired t-test: t = {t_stat:.3f}, p = {p_val:.5f}")

    runs_a = [r for r in all_results if r["track"] == "A"]
    runs_b = [r for r in all_results if r["track"] == "B"]
    if runs_a and runs_b:
        cm_a = np.sum([np.array(r["test_metrics"]["confusion_matrix"]) for r in runs_a], axis=0)
        cm_b = np.sum([np.array(r["test_metrics"]["confusion_matrix"]) for r in runs_b], axis=0)

        cm_a_norm = cm_a.astype(float) / np.maximum(cm_a.sum(axis=1, keepdims=True), 1) * 100.0
        cm_b_norm = cm_b.astype(float) / np.maximum(cm_b.sum(axis=1, keepdims=True), 1) * 100.0

        labels_upper = [c.upper() for c in CLASS_ORDER]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)

        sns.heatmap(cm_a_norm, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=labels_upper, yticklabels=labels_upper, cbar=False, ax=axes[0],
                    annot_kws={"size": 9.5, "weight": "bold"})
        axes[0].set_title("(a) Track A: Baseline (Cross-Entropy, Frozen)\nRow-Normalised Confusion Matrix (%)",
                          fontsize=11.5, fontweight="bold", pad=10)
        axes[0].set_xlabel("Predicted Label", fontsize=10.5)
        axes[0].set_ylabel("True Label", fontsize=10.5)

        sns.heatmap(cm_b_norm, annot=True, fmt=".1f", cmap="Oranges", vmin=0, vmax=100,
                    xticklabels=labels_upper, yticklabels=labels_upper, cbar=True, ax=axes[1],
                    annot_kws={"size": 9.5, "weight": "bold"})
        axes[1].set_title("(b) Track B: Proposed (Focal Loss + Progressive FT)\nRow-Normalised Confusion Matrix (%)",
                          fontsize=11.5, fontweight="bold", pad=10)
        axes[1].set_xlabel("Predicted Label", fontsize=10.5)
        axes[1].set_ylabel("True Label", fontsize=10.5)

        plt.tight_layout()
        fig2_path = os.path.join(OUTPUT_DIR, f"{prefix}Figure_2_Confusion_Matrices.png")
        plt.savefig(fig2_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"\n[Info] Figure 2 已成功生成并保存至: {fig2_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "quick", "full"],
                        help="demo: 1分钟极速验证全流程 | quick: 跑全量数据的 Track A & B (1 seed) | full: 跑全部 5 Tracks x 5 Seeds")
    args = parser.parse_args()

    print(f"[Device] 当前使用计算设备: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU - 警告：未检测到CUDA GPU！'})")
    run_unit_tests()

    split_csv = os.path.join(OUTPUT_DIR, "splits.csv")
    df = pd.read_csv(split_csv)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    alpha_weights = compute_inverse_freq_weights(train_df, num_classes=7)
    print(f"[Info] 训练集逆频率类别权重 alpha (归一化后总和=7): {[round(x, 4) for x in alpha_weights.tolist()]}")

    if args.mode == "demo":
        print("\n⚡ [DEMO 模式] 仅抽样 500 张训练 / 150 张验证 / 150 张测试，运行 Track A & Track B 各 2 Epochs 验证管线...")
        train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 75), random_state=42)).reset_index(drop=True)
        val_df = val_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 25), random_state=42)).reset_index(drop=True)
        test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 25), random_state=42)).reset_index(drop=True)
        tracks_to_run = ["A", "B"]
        seeds_to_run = [42]
        max_epochs = 2
        patience = 2
        prefix = "demo_"
    elif args.mode == "quick":
        print("\n🚀 [QUICK 模式] 使用全量数据，先跑最核心的 Track A 与 Track B (Seed 42)...")
        tracks_to_run = ["A", "B"]
        seeds_to_run = [42]
        max_epochs = 18
        patience = 6
        prefix = ""
    else:
        print("\n🏆 [FULL 模式] 使用全量数据，运行全部 5 Tracks × 5 Seeds (共 25 轮实验)...")
        tracks_to_run = ["A", "B", "B1", "B2", "C"]
        seeds_to_run = ALL_SEEDS
        max_epochs = 18
        patience = 6
        prefix = ""

    train_tf, eval_tf = get_transforms()
    train_ds = HAM10000Dataset(train_df, transform=train_tf, preload_to_ram=True)
    val_ds = HAM10000Dataset(val_df, transform=eval_tf, preload_to_ram=True)
    test_ds = HAM10000Dataset(test_df, transform=eval_tf, preload_to_ram=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    all_results = []
    for track_key in tracks_to_run:
        for seed in seeds_to_run:
            res = train_single_run(track_key, seed, train_loader, val_loader, test_loader,
                                   alpha_weights, max_epochs=max_epochs, patience=patience, prefix=prefix)
            all_results.append(res)

    summarize_and_generate_outputs(all_results, prefix=prefix)


if __name__ == "__main__":
    main()