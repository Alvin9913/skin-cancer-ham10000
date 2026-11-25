"""
06_extras.py - Final-report extras (run from the repo root).

Stages
  gamma   : train Track B (Focal + progressive FT) for gamma in {1.0, 3.0} x 5 seeds
  val     : re-run every existing checkpoint on the validation split, save probabilities
  calib   : expected calibration error / Brier score on test, reliability diagram
  thresh  : validation-selected P_mal thresholds -> test operating points
  errors  : error decomposition by class and by metadata subgroup
  gradcam : Grad-CAM overlays for Track A and Track B

Artifacts go to outputs/ and figures/ inside the repo.
Source checkpoints / test predictions are read from outputs/ (override with SKIN_DATA_ROOT).
"""
import os
import sys
import json
import glob
import argparse
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
# 数据与权重的根目录：默认就是本仓库；指向别处就设环境变量 SKIN_DATA_ROOT
ROOT = os.environ.get("SKIN_DATA_ROOT", ROOT)
SRC = ROOT
OUT = ROOT
OUT_OUTPUTS = os.path.join(OUT, "outputs")
OUT_FIG = os.path.join(OUT, "figures")
OUT_METRICS = os.path.join(OUT_OUTPUTS, "metrics")
os.makedirs(OUT_METRICS, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
MALIGNANT_INDICES = [1, 3, 4]
SEEDS = [42, 100, 2026, 7, 99]
TRACKS = ["A", "B1", "B2", "C", "B"]
TRACK_NAMES = {
    "A": "A: CE, frozen (baseline)",
    "B1": "B1: CE + progressive FT",
    "B2": "B2: Focal + frozen",
    "C": "C: weighted CE + progressive FT",
    "B": "B: Focal + progressive FT",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = load_module("core_modules", "./02_core_modules.py")
exp = load_module("exp_modules", "./03_run_experiments.py")
DEVICE = exp.DEVICE


def load_splits():
    df = pd.read_csv(os.path.join(OUT_OUTPUTS, "splits.csv"))
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}, df


def build_loaders(batch_size=64):
    parts, df = load_splits()
    train_tf, eval_tf = core.get_transforms()
    train_ds = core.HAM10000Dataset(parts["train"], transform=train_tf, preload_to_ram=True)
    val_ds = core.HAM10000Dataset(parts["val"], transform=eval_tf, preload_to_ram=True)
    test_ds = core.HAM10000Dataset(parts["test"], transform=eval_tf, preload_to_ram=True)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=0, pin_memory=True)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), parts, df


def ckpt_path(track, seed, src=True):
    root = os.path.join(SRC, "outputs", "checkpoints") if src else os.path.join(OUT_OUTPUTS, "checkpoints")
    return os.path.join(root, f"track_{track}_seed_{seed}.pt")


def load_model(track, seed, src=True):
    model = core.SkinCancerResNet50(num_classes=7, dropout_p=0.3, pretrained=False).to(DEVICE)
    state = torch.load(ckpt_path(track, seed, src), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer(model, loader):
    probs, targets = [], []
    for images, y in loader:
        images = images.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
        probs.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(probs)


# ----------------------------------------------------------------------------- stage: gamma
def stage_gamma(val_loader=None, test_loader=None, train_loader=None, parts=None):
    gammas = [1.0, 3.0]
    for g in gammas:
        key = f"B_g{int(g)}"
        exp.TRACK_CONFIGS[key] = {
            "name": f"B: Focal (gamma={g:g}) + progressive FT",
            "loss": "focal",
            "backbone": "progressive",
            "gamma": g,
        }
    train_loader, val_loader, test_loader, parts, df = build_loaders()
    alpha = core.compute_inverse_freq_weights(parts["train"], num_classes=7)
    print(f"[gamma] alpha = {[round(x, 4) for x in alpha.tolist()]}", flush=True)

    rows = []
    for g in gammas:
        key = f"B_g{int(g)}"
        for seed in SEEDS:
            rec = exp.train_single_run(key, seed, train_loader, val_loader, test_loader,
                                       alpha, max_epochs=18, patience=6, prefix="")
            tm = rec["test_metrics"]
            rows.append({
                "gamma": g, "seed": seed,
                "accuracy": tm["accuracy"],
                "malignant_recall": tm["malignant_recall"],
                "malignant_precision": tm["malignant_precision"],
                "malignant_specificity": tm["malignant_specificity"],
                "melanoma_recall": tm["melanoma_recall"],
                "macro_f1": tm["macro_f1"],
                "balanced_acc": tm["balanced_acc"],
                "macro_auroc": tm["macro_auroc"],
                "macro_prauc": tm["macro_prauc"],
                "malignant_prauc": tm["malignant_prauc"],
            })
            pd.DataFrame(rows).to_csv(os.path.join(OUT_OUTPUTS, "extras_gamma_sweep.csv"), index=False)
    print("[gamma] done", flush=True)


# ----------------------------------------------------------------------------- stage: val
def stage_val(train_loader=None, val_loader=None, test_loader=None, parts=None):
    if val_loader is None:
        train_loader, val_loader, test_loader, parts, df = build_loaders()
    done = 0
    for track in TRACKS:
        for seed in SEEDS:
            dst = os.path.join(OUT_METRICS, f"valpreds_track_{track}_seed_{seed}.npz")
            if os.path.exists(dst):
                done += 1
                continue
            if not os.path.exists(ckpt_path(track, seed, src=True)):
                print(f"[val] missing checkpoint {track}/{seed}", flush=True)
                continue
            model = load_model(track, seed, src=True)
            y_true, y_prob = infer(model, val_loader)
            np.savez_compressed(dst, y_true=y_true, y_prob=y_prob)
            done += 1
            print(f"[val] {track}/{seed} -> {dst}", flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"[val] {done} checkpoints processed", flush=True)


# ----------------------------------------------------------------------------- calibration helpers
def ece_score(conf, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability(conf, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins[1:-1], right=True)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return out


def load_test_preds(track, seed):
    p = os.path.join(SRC, "outputs", "metrics", f"preds_track_{track}_seed_{seed}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["y_true"], d["y_prob"]


def stage_calib(train_loader=None, val_loader=None, test_loader=None, parts=None):
    rows = []
    curves = {}
    for track in TRACKS:
        cls_ece, bin_ece, brier, tops = [], [], [], []
        for seed in SEEDS:
            d = load_test_preds(track, seed)
            if d is None:
                continue
            y, p = d
            conf = p.max(axis=1)
            correct = (p.argmax(axis=1) == y).astype(float)
            cls_ece.append(ece_score(conf, correct))
            p_mal = p[:, MALIGNANT_INDICES].sum(axis=1)
            is_mal = np.isin(y, MALIGNANT_INDICES).astype(float)
            bin_ece.append(ece_score(p_mal, is_mal))
            brier.append(float(np.mean((p_mal - is_mal) ** 2)))
            tops.append((conf.mean(), correct.mean()))
        if not cls_ece:
            continue
        rows.append({
            "track": track,
            "ece_7class": float(np.mean(cls_ece)),
            "ece_malignant": float(np.mean(bin_ece)),
            "brier_malignant": float(np.mean(brier)),
            "mean_confidence": float(np.mean([t[0] for t in tops])),
            "accuracy": float(np.mean([t[1] for t in tops])),
        })
        # reliability curve on concatenated predictions across seeds
        cs, cc = [], []
        for seed in SEEDS:
            d = load_test_preds(track, seed)
            if d is None:
                continue
            y, p = d
            pm = p[:, MALIGNANT_INDICES].sum(axis=1)
            cs.append(pm)
            cc.append(np.isin(y, MALIGNANT_INDICES).astype(float))
        curves[track] = reliability(np.concatenate(cs), np.concatenate(cc))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_OUTPUTS, "extras_calibration.csv"), index=False)
    print(res.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="grey", label="Perfect calibration")
    colors = {"A": "#4C72B0", "B1": "#55A868", "B2": "#C44E52", "C": "#8172B2", "B": "#CCB974"}
    for track, pts in curves.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", ms=3.4, lw=1.4, color=colors.get(track), label=TRACK_NAMES[track])
    ax.set_xlabel("Predicted malignant probability")
    ax.set_ylabel("Observed malignant frequency")
    ax.set_title("Reliability of the malignant-risk score (test set)")
    ax.legend(fontsize=6.6, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "Figure_reliability.png"), bbox_inches="tight")
    plt.close(fig)
    print("[calib] done", flush=True)


# ----------------------------------------------------------------------------- stage: threshold
def stage_thresh(train_loader=None, val_loader=None, test_loader=None, parts=None):
    rows = []
    for track in TRACKS:
        for target in [0.90, 0.95, 0.99]:
            sens_v, sens_t, spec_t, prec_t = [], [], [], []
            for seed in SEEDS:
                vp = os.path.join(OUT_METRICS, f"valpreds_track_{track}_seed_{seed}.npz")
                tp = load_test_preds(track, seed)
                if not os.path.exists(vp) or tp is None:
                    continue
                v = np.load(vp)
                vy, vpp = v["y_true"], v["y_prob"][:, MALIGNANT_INDICES].sum(axis=1)
                vmal = np.isin(vy, MALIGNANT_INDICES)
                if vmal.sum() == 0:
                    continue
                thr = float(np.quantile(vpp[vmal], 1.0 - target))
                ty, tpp = tp
                tpm = tpp[:, MALIGNANT_INDICES].sum(axis=1)
                tmal = np.isin(ty, MALIGNANT_INDICES)
                pred = tpm >= thr
                tp_n = int((pred & tmal).sum())
                fn = int((~pred & tmal).sum())
                fp = int((pred & ~tmal).sum())
                tn = int((~pred & ~tmal).sum())
                sens_v.append(100.0 * (vpp[vmal] >= thr).mean())
                sens_t.append(100.0 * tp_n / max(tp_n + fn, 1))
                spec_t.append(100.0 * tn / max(tn + fp, 1))
                prec_t.append(100.0 * tp_n / max(tp_n + fp, 1))
            if not sens_t:
                continue
            rows.append({
                "track": track,
                "target_val_sensitivity_%": int(target * 100),
                "val_sensitivity_%": float(np.mean(sens_v)),
                "test_sensitivity_%": float(np.mean(sens_t)),
                "test_specificity_%": float(np.mean(spec_t)),
                "test_precision_%": float(np.mean(prec_t)),
                "test_prauc_%": float("nan"),
                "n_seeds": len(sens_t),
            })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_OUTPUTS, "extras_thresholds.csv"), index=False)
    print(res.to_string(index=False), flush=True)
    print("[thresh] done", flush=True)


# ----------------------------------------------------------------------------- stage: errors
def stage_errors(train_loader=None, val_loader=None, test_loader=None, parts=None):
    if parts is None:
        _, _, _, parts, _ = build_loaders()
    meta = pd.read_csv(os.path.join(SRC, "outputs", "splits.csv"))
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    rows = []
    for track in TRACKS:
        cm = np.zeros((7, 7), dtype=float)
        n = 0
        for seed in SEEDS:
            d = load_test_preds(track, seed)
            if d is None:
                continue
            y, p = d
            pred = p.argmax(axis=1)
            for t, pr in zip(y, pred):
                cm[t, pr] += 1
            n += 1
        if n == 0:
            continue
        cm = cm / n
        for i in range(7):
            for j in range(7):
                if i != j and cm[i, j] > 0.4:
                    rows.append({"track": track, "true": CLASS_ORDER[i], "pred": CLASS_ORDER[j],
                                 "row_%": round(100 * cm[i, j] / cm[i].sum(), 1)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_OUTPUTS, "extras_confusions.csv"), index=False)

    # subgroup error rates for Track B vs Track A (7-class top-1 error)
    sub_rows = []
    for track in ["A", "B"]:
        acc = np.zeros(len(test_meta))
        cnt = 0
        for seed in SEEDS:
            d = load_test_preds(track, seed)
            if d is None:
                continue
            y, p = d
            acc += (p.argmax(axis=1) == y).astype(float)
            cnt += 1
        if cnt == 0:
            continue
        acc /= cnt
        tmp = test_meta.copy()
        tmp["correct"] = acc
        tmp["age_group"] = pd.cut(tmp["age"], [0, 40, 60, 80, 100],
                                  labels=["<40", "40-59", "60-79", "80+"]).astype(str)
        for col in ["localization", "age_group", "sex"]:
            g = tmp.groupby(col)["correct"].agg(["mean", "count"]).reset_index()
            g = g[g["count"] >= 15]
            for _, r in g.iterrows():
                sub_rows.append({"track": track, "attribute": col, "value": r[col],
                                 "n": int(r["count"]), "accuracy_%": round(100 * r["mean"], 1)})
    pd.DataFrame(sub_rows).to_csv(os.path.join(OUT_OUTPUTS, "extras_subgroups.csv"), index=False)
    print("[errors] done", flush=True)


# ----------------------------------------------------------------------------- stage: gradcam
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0, cls]
        score.backward(retain_graph=False)
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()


def stage_gradcam(train_loader=None, val_loader=None, test_loader=None, parts=None):
    if parts is None:
        _, _, _, parts, _ = build_loaders()
    test_df = parts["test"]
    picks = []
    wanted = [("mel", 3), ("mel", 3), ("nv", 3), ("nv", 3), ("bcc", 2), ("akiec", 2)]
    for cls, k in wanted:
        sub = test_df[test_df["dx"] == cls].head(k)
        picks.extend(sub.itertuples())
    picks = picks[:8]

    _, eval_tf = core.get_transforms()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    models = {"A": load_model("A", 7), "B": load_model("B", 7)}
    cams_all = {}
    for tag, model in models.items():
        layer = model.backbone.layer4[-1].conv3
        gc = GradCAM(model, layer)
        cams = []
        for row in picks:
            img = Image.open(row.image_path).convert("RGB")
            tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
            with torch.enable_grad():
                cam = gc(tensor, image_cls_idx := int(row.label))
            cams.append((img, cam, image_cls_idx))
        gc.close()
        cams_all[tag] = cams
        del model
        torch.cuda.empty_cache()

    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.6))
    for j, row in enumerate(picks):
        img, _, _ = cams_all["A"][j]
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{CLASS_ORDER[int(row.label)].upper()}", fontsize=8)
        axes[0, j].axis("off")
        for i, tag in enumerate(["A", "B"], start=1):
            _, cam, _ = cams_all[tag][j]
            axes[i, j].imshow(np.asarray(img.resize((224, 224))))
            axes[i, j].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(f"Track {tag}", fontsize=9)
    axes[0, 0].annotate("Input", xy=(0.02, 0.98), xycoords="axes fraction", fontsize=8, va="top",
                        bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.suptitle("Grad-CAM: Track A (middle) vs Track B (bottom), class-activated regions", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "Figure_gradcam.png"), bbox_inches="tight")
    plt.close(fig)
    print("[gradcam] done", flush=True)


STAGES = {
    "gamma": stage_gamma,
    "val": stage_val,
    "calib": stage_calib,
    "thresh": stage_thresh,
    "errors": stage_errors,
    "gradcam": stage_gradcam,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="gamma,val,calib,thresh,errors,gradcam")
    args = ap.parse_args()
    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        print(f"\n===== STAGE {name} =====", flush=True)
        try:
            STAGES[name]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[stage {name}] FAILED: {e}", flush=True)
