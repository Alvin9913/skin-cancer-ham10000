import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from PIL import Image

# 固定随机种子，确保每次划分 100% 可复现
RANDOM_SEED = 42
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 类别与临床分组定义（与报告 Section 1 & Table 1 严格对齐）
CLASS_ORDER = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_NAMES = {
    'nv': 'NV (Melanocytic nevi)',
    'mel': 'MEL (Melanoma)',
    'bkl': 'BKL (Benign keratosis)',
    'bcc': 'BCC (Basal cell carcinoma)',
    'akiec': 'AKIEC (Actinic keratoses)',
    'vasc': 'VASC (Vascular lesions)',
    'df': 'DF (Dermatofibroma)'
}
MALIGNANT_CLASSES = {'mel', 'bcc', 'akiec'}  # 恶性/癌前病变组

def locate_or_download_dataset(local_data_dir="./data"):
    """
    先检查本地 ./data 是否有 HAM10000_metadata.csv，
    如果没有，则通过 kagglehub 自动下载。
    """
    meta_candidates = glob.glob(os.path.join(local_data_dir, "**", "HAM10000_metadata.csv"), recursive=True)
    if meta_candidates:
        base_dir = os.path.dirname(meta_candidates[0])
        print(f"[Info] 在本地找到数据集: {base_dir}")
        return base_dir

    print("[Info] 本地未检测到数据集，正在通过 kagglehub 自动下载 HAM10000 (约 2.9 GB)...")
    import kagglehub
    download_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print(f"[Info] 数据集下载完成，路径: {download_path}")
    return download_path

def build_dataframe(data_dir):
    meta_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(meta_path):
        # 递归查找
        meta_path = glob.glob(os.path.join(data_dir, "**", "HAM10000_metadata.csv"), recursive=True)[0]
    
    df = pd.read_csv(meta_path)
    
    # 扫描所有 .jpg 图像并建立 image_id -> 完整路径映射
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    
    df['image_path'] = df['image_id'].map(path_map)
    missing = df['image_path'].isna().sum()
    if missing > 0:
        raise FileNotFoundError(f"有 {missing} 张图片未找到对应文件，请检查数据集完整性！")
    
    # 映射 0-6 整数标签和二分类恶性标签
    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASS_ORDER)}
    df['label'] = df['dx'].map(label_to_idx)
    df['is_malignant'] = df['dx'].apply(lambda x: 1 if x in MALIGNANT_CLASSES else 0)
    
    print(f"[Info] 成功加载元数据：共 {len(df)} 张图片，来自 {df['lesion_id'].nunique()} 个独立病变 (unique lesions)。")
    return df

def create_grouped_splits(df):
    """
    使用 StratifiedGroupKFold(n_splits=20) 实现按 lesion_id 分组的 70% / 15% / 15% 划分
    14 折 -> Train (70%), 3 折 -> Val (15%), 3 折 -> Test (15%)
    """
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df['dx'], df['lesion_id'])):
        df.loc[val_idx, 'fold'] = fold

    def assign_split(fold):
        if fold < 14:
            return 'train'
        elif fold < 17:
            return 'val'
        else:
            return 'test'

    df['split'] = df['fold'].apply(assign_split)

    # 严格验证零数据泄露（Zero Lesion Leakage）
    train_lesions = set(df[df['split'] == 'train']['lesion_id'])
    val_lesions = set(df[df['split'] == 'val']['lesion_id'])
    test_lesions = set(df[df['split'] == 'test']['lesion_id'])
    
    assert len(train_lesions.intersection(val_lesions)) == 0, "发现 Train-Val 病变重叠！"
    assert len(train_lesions.intersection(test_lesions)) == 0, "发现 Train-Test 病变重叠！"
    assert len(val_lesions.intersection(test_lesions)) == 0, "发现 Val-Test 病变重叠！"
    print("[Verified] 零病变泄露检查通过 (Zero lesion_id overlap across Train/Val/Test)！")

    split_csv_path = os.path.join(OUTPUT_DIR, "splits.csv")
    df.to_csv(split_csv_path, index=False)
    print(f"[Info] 固定划分文件已保存至: {split_csv_path}")
    return df

def print_appendix_b_table(df):
    print("\n" + "="*65)
    print("📋 请将以下数据直接填入报告 Appendix B (Image Counts per Split):")
    print("="*65)
    print(f"{'Class':<10} | {'Train (~70%)':<15} | {'Validation (~15%)':<18} | {'Test (~15%)':<15} | {'Total':<8}")
    print("-" * 65)
    
    for cls in CLASS_ORDER:
        sub = df[df['dx'] == cls]
        n_train = (sub['split'] == 'train').sum()
        n_val = (sub['split'] == 'val').sum()
        n_test = (sub['split'] == 'test').sum()
        n_total = len(sub)
        print(f"{cls.upper():<10} | {n_train:<15} | {n_val:<18} | {n_test:<15} | {n_total:<8}")
    
    print("-" * 65)
    t_train = (df['split'] == 'train').sum()
    t_val = (df['split'] == 'val').sum()
    t_test = (df['split'] == 'test').sum()
    print(f"{'Total':<10} | {t_train:<15} | {t_val:<18} | {t_test:<15} | {len(df):<8}")
    print("="*65 + "\n")

def generate_figure_1(df):
    """
    生成报告 Figure 1：左图类别分布柱状图，右图 7 个类别的代表性样本图片
    """
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = fig.add_gridspec(2, 8, width_ratios=[1.3, 1.3, 1.3, 1.3, 0.2, 1, 1, 1], wspace=0.35, hspace=0.35)

    # 左侧：类别分布柱状图 (占据前 4 列)
    ax_bar = fig.add_subplot(gs[:, :4])
    counts = [len(df[df['dx'] == c]) for c in CLASS_ORDER]
    labels = [c.upper() for c in CLASS_ORDER]
    colors = ['#d95f02' if c in MALIGNANT_CLASSES else '#1f78b4' for c in CLASS_ORDER]

    bars = ax_bar.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax_bar.set_title("(a) HAM10000 Class Distribution (Severe Imbalance)", fontsize=12, fontweight='bold', pad=10)
    ax_bar.set_ylabel("Number of Images", fontsize=11)
    ax_bar.set_ylim(0, max(counts) * 1.15)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, count in zip(bars, counts):
        pct = count / len(df) * 100
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 120,
                    f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # 图例说明良恶性
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f78b4', edgecolor='black', label='Benign Group (80.5%)'),
        Patch(facecolor='#d95f02', edgecolor='black', label='Malignant / Pre-malignant Group (19.5%)')
    ]
    ax_bar.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9.5)

    # 右侧：展示 7 个类别的代表性皮肤镜图像 (2行 x 4列网格中的 7 个位置)
    positions = [(0, 5), (0, 6), (0, 7), (1, 5), (1, 6), (1, 7)]
    # 调整为 2x4 展示 7 张图 + 1 个说明位
    fig.delaxes(ax_bar) # 重新用更清晰的 GridSpec 布局
    plt.clf()

    fig = plt.figure(figsize=(14, 5.2), dpi=300)
    gs = fig.add_gridspec(2, 7, width_ratios=[1.4, 1.4, 1.4, 0.25, 1, 1, 1], wspace=0.25, hspace=0.35)
    
    # 重新绘制左图
    ax_bar = fig.add_subplot(gs[:, :3])
    bars = ax_bar.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax_bar.set_title("(a) HAM10000 Class Distribution", fontsize=12, fontweight='bold', pad=10)
    ax_bar.set_ylabel("Number of Images", fontsize=10.5)
    ax_bar.set_ylim(0, max(counts) * 1.18)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, count in zip(bars, counts):
        pct = count / len(df) * 100
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8)
    ax_bar.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)

    # 右侧 2行 x 4列 (把第4列也利用起来放样本)
    plt.clf()
    fig = plt.figure(figsize=(15, 5.2), dpi=300)
    gs = fig.add_gridspec(2, 8, width_ratios=[1.2, 1.2, 1.2, 0.25, 1, 1, 1, 1], wspace=0.25, hspace=0.35)
    
    ax_bar = fig.add_subplot(gs[:, :3])
    bars = ax_bar.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax_bar.set_title("(a) HAM10000 Class Distribution", fontsize=12, fontweight='bold', pad=10)
    ax_bar.set_ylabel("Number of Images", fontsize=10.5)
    ax_bar.set_ylim(0, max(counts) * 1.18)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, count in zip(bars, counts):
        pct = count / len(df) * 100
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8)
    ax_bar.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)

    sample_coords = [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6)]
    np.random.seed(RANDOM_SEED)
    for idx, cls in enumerate(CLASS_ORDER):
        r, c = sample_coords[idx]
        ax_img = fig.add_subplot(gs[r, c])
        sample_row = df[df['dx'] == cls].iloc[0]
        img = Image.open(sample_row['image_path'])
        ax_img.imshow(img)
        tag = "Malignant" if cls in MALIGNANT_CLASSES else "Benign"
        title_color = '#b30000' if cls in MALIGNANT_CLASSES else '#004c99'
        ax_img.set_title(f"{cls.upper()} ({tag})", fontsize=9, fontweight='bold', color=title_color)
        ax_img.axis('off')

    # 右下角最后一个格子放简短图注说明
    ax_note = fig.add_subplot(gs[1, 7])
    ax_note.axis('off')
    ax_note.text(0.05, 0.5, "(b) Representative\ndermatoscopic\nsamples across\nthe 7 lesion\ncategories.\n(600×450 px)",
                 fontsize=9.5, va='center', ha='left', style='italic',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc'))

    fig_path = os.path.join(OUTPUT_DIR, "Figure_1_EDA.png")
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Info] Figure 1 已成功生成并保存至: {fig_path}")

if __name__ == "__main__":
    data_dir = locate_or_download_dataset("./data")
    df = build_dataframe(data_dir)
    df = create_grouped_splits(df)
    print_appendix_b_table(df)
    generate_figure_1(df)