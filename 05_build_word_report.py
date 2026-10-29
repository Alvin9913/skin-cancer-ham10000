import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# ======= 请在这里直接填入你的真实学号与课程代码 =======
AUTHOR_NAME = "Zhipeng Zhang"
STUDENT_ID = "a1990000"       # <-- 替换为你的真实 Student ID
COURSE_CODE = "AIML-XXXX"     # <-- 替换为你的真实 Course Code
# ====================================================

OUTPUT_DOCX = "./outputs/Milestone_Report_Final_Submission.docx"
FIG1_PATH = "./outputs/Figure_1_EDA.png"
FIG2_PATH = "./outputs/Figure_2_Confusion_Matrices_Polished.png"


def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge, val in kwargs.items():
        edge_el = OxmlElement(f'w:{edge}')
        edge_el.set(qn('w:val'), val.get('val', 'single'))
        edge_el.set(qn('w:sz'), str(val.get('sz', 4)))
        edge_el.set(qn('w:space'), '0')
        edge_el.set(qn('w:color'), val.get('color', '000000'))
        tcBorders.append(edge_el)
    tcPr.append(tcBorders)


def prevent_table_row_split(table):
    """防止表格跨页断行，并保持表头与下一行同页"""
    nrows = len(table.rows)
    for r_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if r_idx == 0:
            trPr.append(OxmlElement('w:tblHeader'))
        if r_idx < nrows - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True


def format_academic_table(table, col_alignments=None, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    nrows = len(table.rows)
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            if col_widths and c_idx < len(col_widths):
                cell.width = Inches(col_widths[c_idx])

            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = OxmlElement('w:tcMar')
            for m in ['top', 'bottom']:
                node = OxmlElement(f'w:{m}')
                node.set(qn('w:w'), '40')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            for m in ['left', 'right']:
                node = OxmlElement(f'w:{m}')
                node.set(qn('w:w'), '80')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            tcPr.append(tcMar)

            borders = {}
            if r_idx == 0:
                borders['top'] = {'val': 'single', 'sz': 10, 'color': '000000'}
                borders['bottom'] = {'val': 'single', 'sz': 6, 'color': '000000'}
            elif r_idx == nrows - 1:
                borders['bottom'] = {'val': 'single', 'sz': 10, 'color': '000000'}
            if borders:
                set_cell_border(cell, **borders)

            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.05
                if col_alignments and c_idx < len(col_alignments):
                    p.alignment = col_alignments[c_idx]
                for run in p.runs:
                    run.font.name = 'Calibri'
                    run.font.size = Pt(9.0)
                    if r_idx == 0:
                        run.font.bold = True

    prevent_table_row_split(table)


def add_heading_styled(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7 if level == 1 else 4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    else:
        run.font.size = Pt(10.0)
        run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB5)
    return p


def add_body_p(doc, text, bold_prefix=None, space_after=3.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.10
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = 'Calibri'
        r_bold.font.size = Pt(10)
        r_bold.font.bold = True
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(10)
    return p


def add_bullet_p(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.08
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_b = p.add_run(bold_prefix)
    r_b.font.name = 'Calibri'
    r_b.font.size = Pt(10)
    r_b.font.bold = True
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(10)
    return p


def add_table_caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(9.5)
    r.font.bold = True
    return p


def build_report():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.60)
        section.bottom_margin = Inches(0.60)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)

    # Header
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(1.5)
    t_run = title_p.add_run("Milestone Report: Improving Skin Cancer Classification on HAM10000 Using Transfer Learning and Focal Loss")
    t_run.font.name = 'Calibri'
    t_run.font.size = Pt(13)
    t_run.font.bold = True
    t_run.font.color.rgb = RGBColor(0x0F, 0x24, 0x3E)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(5)
    s_run = sub_p.add_run(f"{AUTHOR_NAME} | {STUDENT_ID} | {COURSE_CODE} | Milestone Report Submission")
    s_run.font.name = 'Calibri'
    s_run.font.size = Pt(9.5)
    s_run.font.italic = True
    s_run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # ==================== SECTION 1 ====================
    add_heading_styled(doc, "1. Project Goals and Updated Problem Understanding", level=1)
    add_body_p(doc,
        "Skin cancer represents a major public health burden in Australia, where approximately two-thirds of the population are expected to be diagnosed with a skin malignancy by age 70 (Cancer Council Australia, 2023). Although early detection of melanoma yields a five-year relative survival rate exceeding 99% (Australian Institute of Health and Welfare [AIHW], 2022), visual inspection of dermatoscopic images is subjective, time-consuming, and prone to missed malignancies. Deep convolutional neural networks (CNNs) can match dermatologists on constrained classification tasks (Esteva et al., 2017), yet standard classifiers trained on clinical archives inherit severe class imbalance. Because benign lesions vastly outnumber malignant ones, a network optimised for overall accuracy learns that defaulting to benign predictions is statistically rewarded, masking high false-negative rates on clinically critical cases.")
    add_body_p(doc,
        "This project is a proof-of-concept study asking one question: does an imbalance-aware training strategy detect malignant lesions measurably better than a standard transfer-learning baseline? We compare a baseline ResNet-50 trained with Cross-Entropy (CE) loss and a frozen backbone (Track A) against an upgraded ResNet-50 combining class-balanced Focal Loss (Lin et al., 2017) with progressive fine-tuning (Track B). The target is a decision-support prototype that maps a dermatoscopic image to a probability distribution over seven diagnostic categories alongside an aggregated malignancy risk score.")
    add_body_p(doc,
        "Exploratory data analysis (EDA) and implementation have refined our problem formulation in four ways not anticipated in the proposal:", space_after=2)

    add_bullet_p(doc, "Long-tailed multi-class skewness beyond binary imbalance: ",
        "Within the seven-class label space, Melanocytic Nevi (NV, 6,705 images, 66.9%) outnumber Dermatofibroma (DF, 115 images, 1.1%) by 58.3 to 1. A trivial Zero-Rule (ZeroR) classifier that predicts NV for every image already achieves 66.9% accuracy. The task is therefore a long-tailed seven-class problem where raw accuracy is uninformative and recall on rare classes has high sampling variance.")
    add_bullet_p(doc, "Explicit clinical grouping: ",
        "We group Melanoma (MEL), Basal Cell Carcinoma (BCC), and Actinic Keratoses / Intraepithelial Carcinoma (AKIEC) into the clinically actionable Malignant/Pre-malignant Group (1,954 images, 19.5%), with NV, Benign Keratosis (BKL), Vascular Lesions (VASC), and DF grouped as Benign (8,061 images, 80.5%). Results are reported both per class and for this binary clinical view.")
    add_bullet_p(doc, "De-confounding the two upgrades: ",
        "The proposal changed the loss function and fine-tuning strategy simultaneously, which would confound attribution in Track B. We therefore added three controlled ablation tracks (Tracks B1, B2, and C; Section 3).")
    add_bullet_p(doc, "Lesion-level data leakage risk: ",
        "HAM10000 contains multiple images of the same physical lesion. Random image-level splitting places near-duplicate views across training and test sets, inflating test metrics. Our pipeline enforces grouped stratification by lesion_id (Section 2).")

    # ==================== SECTION 2 ====================
    add_heading_styled(doc, "2. Data and Pre-processing Workflow", level=1)
    add_body_p(doc,
        "Dataset. HAM10000 (Tschandl et al., 2018) contains 10,015 dermatoscopic images (600×450 pixels) across seven classes. Metadata maps each image to its diagnosis (dx), verification method (histopathology, follow-up, consensus, or confocal microscopy), age, sex, body site, and lesion_id. Table 1 and Figure 1 summarise the dataset.")

    add_table_caption(doc, "Table 1. HAM10000 class distribution and clinical risk grouping.")
    t1_data = [
        ["Class Code & Diagnostic Name", "Clinical Group", "Images", "Share (%)"],
        ["NV (Melanocytic nevi)", "Benign", "6,705", "66.9%"],
        ["MEL (Melanoma)", "Malignant", "1,113", "11.1%"],
        ["BKL (Benign keratosis-like lesions)", "Benign", "1,099", "11.0%"],
        ["BCC (Basal cell carcinoma)", "Malignant", "514", "5.1%"],
        ["AKIEC (Actinic keratoses / Bowen's disease)", "Malignant (Pre-malignant)", "327", "3.3%"],
        ["VASC (Vascular lesions)", "Benign", "142", "1.4%"],
        ["DF (Dermatofibroma)", "Benign", "115", "1.1%"],
        ["Total (7,470 Unique Lesions)", "80.5% Benign / 19.5% Malignant", "10,015", "100.0%"]
    ]
    t1 = doc.add_table(rows=len(t1_data), cols=4)
    for r_i, row in enumerate(t1_data):
        for c_i, val in enumerate(row):
            t1.cell(r_i, c_i).text = val
    format_academic_table(t1,
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
        [3.1, 2.0, 0.9, 0.9])

    if os.path.exists(FIG1_PATH):
        p_f1 = doc.add_paragraph()
        p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f1.paragraph_format.space_before = Pt(3)
        p_f1.paragraph_format.space_after = Pt(1)
        p_f1.paragraph_format.keep_with_next = True
        p_f1.add_run().add_picture(FIG1_PATH, width=Inches(6.0))

        p_f1_cap = doc.add_paragraph()
        p_f1_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f1_cap.paragraph_format.space_before = Pt(0)
        p_f1_cap.paragraph_format.space_after = Pt(3.5)
        r_cap1 = p_f1_cap.add_run("Figure 1. (a) HAM10000 class distribution showing benign vs. malignant grouping. (b) Representative samples across the seven classes.")
        r_cap1.font.name = 'Calibri'
        r_cap1.font.size = Pt(8.5)
        r_cap1.font.italic = True

    add_body_p(doc,
        "The 10,015 images originate from 7,470 unique lesions. To prevent near-duplicate leakage across splits, we partitioned the data using a 20-fold StratifiedGroupKFold grouped by lesion_id (14 folds train, 3 validation, 3 test) and verified zero lesion overlap across subsets. This yields 7,013 training (70.0%), 1,501 validation (15.0%), and 1,501 test (15.0%) images, preserving class allocation ratios within about 1 percentage point of the 70/15/15 target across all categories (Appendix B). The fixed split is saved once and reused across all tracks and seeds. Because of the long tail, the validation and test sets contain only 17–18 DF and 21–22 VASC images; per-class recall on these rare classes has wide uncertainty, so primary conclusions rest on the aggregated malignant group (n = 292 in the test set).",
        bold_prefix="What Changed Since the Proposal. ")

    add_body_p(doc,
        "Images are resized to 224×224 and normalised with ImageNet channel statistics. Training augmentation is applied on the fly: random horizontal and vertical flips (p = 0.5), random rotations up to 180°, random resized crops (scale 85%–100%), and conservative colour jitter (brightness, contrast, saturation ±0.2; hue ±0.02). Hue shifts are kept small because colour is diagnostic in dermatoscopy. Validation and test images receive deterministic resizing and normalisation only. Regularisation uses L2 weight decay (1e-4), dropout (p = 0.3) before the linear head, and early stopping (patience = 6 epochs after epoch 10) monitoring an equally weighted composite of validation Macro-F1 and Malignant Recall rather than validation loss, which is dominated by NV.",
        bold_prefix="Pre-processing, Augmentation, and Regularisation. ")

    add_body_p(doc,
        "In baseline runs (Track A), the model reached an overall test accuracy of 72.7 ± 0.8%—only 5.8 percentage points above a trivial ZeroR classifier (66.9%)—while recalling just 43.0 ± 7.5% of melanomas and 56.8 ± 6.5% of malignant lesions overall. Augmentation reduced overfitting but did not fix this bias, confirming that minority classes contribute insufficient gradient updates under unweighted Cross-Entropy.",
        bold_prefix="Where Early Modelling Succeeded and Failed. ")

    # ==================== SECTION 3 ====================
    add_heading_styled(doc, "3. Modelling Approach and Ablation Design", level=1)
    add_body_p(doc,
        "All models use an ImageNet-pretrained ResNet-50 backbone (He et al., 2016) in PyTorch, replacing the output layer with Dropout(p = 0.3) and Linear(2048, 7).")

    add_bullet_p(doc, "Track A — Baseline (Standard CE + Frozen Backbone): ",
        "All residual layers remain frozen and only the classification head is trained with standard CE loss (Adam, lr = 1e-3, batch size = 64). Frozen BatchNorm2d layers are explicitly kept in evaluation mode (.eval()) so mini-batch updates do not corrupt pre-trained running statistics.")
    add_bullet_p(doc, "Track B — Upgraded Model (Focal Loss + Progressive Fine-Tuning): ",
        "First, CE is replaced by Focal Loss (Lin et al., 2017): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), where p_t is the true-class probability and (1 - p_t)^gamma (with gamma = 2.0) down-weights easy majority examples. In early trials, setting alpha_t to raw inverse frequencies (1/N_c, weighting DF 58.3x above NV) destabilised early training and collapsed predictions toward DF and VASC. We resolved this by defining alpha_c proportional to (1 / N_c)^0.5 multiplied by a 1.25x clinical priority factor for the malignant classes (MEL, BCC, AKIEC) and normalised to sum to 7. In the class order (NV, MEL, BKL, BCC, AKIEC, VASC, DF), this yields alpha = [0.22, 0.69, 0.55, 1.01, 1.27, 1.54, 1.72] (sum = 7.00). The custom FocalLoss passed unit tests verifying numerical equivalence (difference < 1e-6) with CrossEntropyLoss at gamma = 0 and alpha = 1. Second, progressive fine-tuning runs in three stages: (i) Epochs 1–3 train the head only (lr = 1e-3); (ii) Epochs 4–8 unfreeze layer4 (lr = 1e-4); and (iii) Epochs 9–18 unfreeze layer3 (lr = 5e-5) while shallow layers stay frozen.")
    add_bullet_p(doc, "Ablation Tracks (B1, B2, and C): ",
        "To separate the two modifications, we added Track B1 (standard CE + progressive fine-tuning, isolating domain adaptation), Track B2 (Focal Loss + frozen backbone, isolating the loss on fixed features), and Track C (weighted CE with the same alpha vector and gamma = 0 + progressive fine-tuning, testing whether Focal focusing beats static re-weighting). All five tracks are complete across 5 seeds (Appendix C).")

    # ==================== SECTION 4 ====================
    add_heading_styled(doc, "4. System Approach and Workflow", level=1)
    add_body_p(doc,
        "The experimental workflow follows a five-stage modular pipeline designed for reproducibility: (1) Ingestion parses metadata, builds lesion_id groups, verifies zero leakage, and exports the fixed split; (2) Data loading caches pre-resized images in memory to avoid disk I/O bottlenecks and applies seeded transforms; (3) Training runs any (track, seed) configuration using mixed precision (torch.amp) on a single NVIDIA GeForce RTX 3080 Ti GPU (12 GB VRAM), logging validation metrics and saving the best validation checkpoint; (4) Evaluation runs the held-out test set (n = 1,501) once per checkpoint, outputting both the 7-class softmax distribution and a derived Malignant Probability P_mal = P(MEL) + P(BCC) + P(AKIEC) for clinical screening; and (5) Aggregation computes cross-seed statistics and confusion matrices. All 25 runs (5 tracks × 5 seeds) completed in ~45 minutes of compute time.")

    # ==================== SECTION 5 ====================
    add_heading_styled(doc, "5. Testing and Evaluation Approach", level=1)
    add_body_p(doc,
        "Overall accuracy is reported only for context. Models are evaluated using: (1) Malignant-group sensitivity (recall) across the n = 292 malignant test cases (primary metric) and standalone Melanoma recall (n = 168); (2) Malignant precision and specificity (on n = 1,209 benign cases) to quantify false-positive referrals, alongside seven-class Macro-F1 and Balanced Accuracy; (3) Threshold-independent AUROC and PR-AUC (one-vs-rest macro and binary malignant using P_mal) to separate representation quality from threshold shifts; and (4) Row-normalised confusion matrices.")
    add_body_p(doc,
        "Statistical Protocol and Limitations. Each track is trained with 5 paired random seeds (42, 100, 2026, 7, 99) on the fixed split, reported as Mean ± SD. We report paired two-sided t-tests on malignant recall alongside 95% bootstrap confidence intervals (1,000 test-set resamples). An important methodological limitation is that all 5 seeds share a single fixed train/validation/test split and test-set bootstrap resampling captures finite test-sample variance rather than variance across alternative data partitions; repeating outer cross-validation folds was omitted due to compute constraints.")

    # ==================== SECTION 6 ====================
    add_heading_styled(doc, "6. Progress, Preliminary Results, and Critical Reflection", level=1)
    add_body_p(doc,
        "Status Against Plan. Phases 1, 2, and 3 (data pipeline, Track A, Track B, and ablations B1, B2, and C across all 5 seeds) are 100% complete. Phase 4 core evaluation is complete; only the gamma hyperparameter sweep, probability calibration, and Grad-CAM inspection remain for the final report (Appendix A). Tables 2 and 3 summarise the held-out test results.")

    add_table_caption(doc, "Table 2. Core test-set classification performance across all five tracks (Mean ± SD over 5 seeds, unit: %).")
    t2_data = [
        ["Track & Configuration", "Accuracy", "Malignant Recall", "Melanoma Recall", "Macro-F1", "Balanced Acc."],
        ["A: CE, frozen (Baseline)", "72.7 ± 0.8", "56.8 ± 6.5", "43.0 ± 7.5", "44.8 ± 3.7", "43.0 ± 5.1"],
        ["B1: CE + progressive FT", "80.2 ± 0.9", "71.8 ± 2.4", "56.2 ± 5.1", "67.5 ± 1.9", "67.5 ± 2.5"],
        ["B2: Focal (γ=2) + frozen", "66.3 ± 4.5", "74.0 ± 8.8", "59.0 ± 13.8", "46.5 ± 2.5", "51.8 ± 1.6"],
        ["C: Weighted CE (γ=0) + prog. FT", "75.8 ± 2.6", "80.9 ± 2.9", "71.3 ± 5.3", "65.1 ± 4.0", "69.5 ± 5.7"],
        ["B: Focal (γ=2) + prog. FT (Proposed)", "73.0 ± 3.5", "86.0 ± 1.3", "76.0 ± 2.7", "64.3 ± 4.4", "69.6 ± 5.7"]
    ]
    t2 = doc.add_table(rows=len(t2_data), cols=6)
    for r_i, row in enumerate(t2_data):
        for c_i, val in enumerate(row):
            t2.cell(r_i, c_i).text = val
    format_academic_table(t2,
        [WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER]*5,
        [2.3, 1.0, 1.15, 1.15, 0.95, 1.05])

    add_table_caption(doc, "Table 3. Binary triage specificity/precision and threshold-independent ranking metrics (Mean ± SD over 5 seeds, unit: %).")
    t3_data = [
        ["Track & Configuration", "Mal. Precision", "Mal. Specificity", "Mal. PR-AUC", "Macro AUROC", "Macro PR-AUC"],
        ["A: CE, frozen (Baseline)", "53.1 ± 3.4", "87.7 ± 2.6", "58.8 ± 0.9", "90.1 ± 0.4", "51.3 ± 1.0"],
        ["B1: CE + progressive FT", "64.4 ± 2.3", "90.4 ± 1.2", "76.1 ± 1.4", "95.3 ± 0.4", "73.0 ± 2.4"],
        ["B2: Focal (γ=2) + frozen", "43.7 ± 6.8", "75.7 ± 7.8", "61.0 ± 2.4", "89.5 ± 0.3", "50.5 ± 0.9"],
        ["C: Weighted CE (γ=0) + prog. FT", "53.1 ± 3.1", "82.6 ± 2.6", "72.7 ± 3.4", "94.8 ± 0.5", "71.8 ± 4.0"],
        ["B: Focal (γ=2) + prog. FT (Proposed)", "49.2 ± 5.0", "78.2 ± 4.4", "72.4 ± 4.2", "94.2 ± 0.5", "69.6 ± 3.3"]
    ]
    t3 = doc.add_table(rows=len(t3_data), cols=6)
    for r_i, row in enumerate(t3_data):
        for c_i, val in enumerate(row):
            t3.cell(r_i, c_i).text = val
    format_academic_table(t3,
        [WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER]*5,
        [2.3, 1.05, 1.1, 1.05, 1.05, 1.05])

    if os.path.exists(FIG2_PATH):
        p_f2 = doc.add_paragraph()
        p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f2.paragraph_format.space_before = Pt(3)
        p_f2.paragraph_format.space_after = Pt(1)
        p_f2.paragraph_format.keep_with_next = True
        p_f2.add_run().add_picture(FIG2_PATH, width=Inches(6.3))

        p_f2_cap = doc.add_paragraph()
        p_f2_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_f2_cap.paragraph_format.space_before = Pt(0)
        p_f2_cap.paragraph_format.space_after = Pt(3.5)
        r_cap2 = p_f2_cap.add_run("Figure 2. Row-normalised test confusion matrices (%) averaged over 5 seeds for (a) Track A and (b) Track B, with test-set sample sizes (n).")
        r_cap2.font.name = 'Calibri'
        r_cap2.font.size = Pt(8.5)
        r_cap2.font.italic = True

    add_body_p(doc,
        "Results and Ablation Analysis. First, Track A illustrates the majority-class bias: it achieves 72.7% accuracy by labelling 89.4% of benign NV correctly (Figure 2a), but misses 43.2% of malignant lesions (Malignant Recall = 56.8%) and 57.0% of melanomas (Melanoma Recall = 43.0%, with 35.5% of melanomas misclassified as NV). Second, Track B increases Malignant Recall by +29.11 percentage points (pp) to 86.0 ± 1.3% (paired t(4) = 9.37, p = 0.0007; 95% bootstrap CI: [+24.79, +33.08] pp) and Melanoma Recall by +33.0 pp to 76.0 ± 2.7%, reducing MEL-to-NV errors from 35.5% to 9.6% (Figure 2b). Balanced Accuracy rises by +26.6 pp (43.0% to 69.6%) and Macro-F1 by +19.5 pp (44.8% to 64.3%). Recall also improves on rare benign classes—VASC (n = 22) from 46.4% to 84.5% (10 to 19 images) and DF (n = 18) from 16.7% to 64.4% (3 to 12 images)—though these two small classes carry wide uncertainty.",
        bold_prefix="Key Findings. ")

    add_body_p(doc,
        "The ablations clarify what drives each metric (Tables 2–3). Unfreezing layer3 and layer4 under standard CE (Track B1) produces the highest overall accuracy (80.2%), Macro-F1 (67.5%), Malignant PR-AUC (76.1%), and Macro AUROC (95.3%), suggesting that domain-specific feature adaptation is the primary driver of representation and ranking quality. Conversely, applying Focal Loss to a frozen backbone (Track B2) leaves Macro AUROC at 89.5% with high seed variance (Melanoma Recall SD = ±13.8%). However, under standard argmax decision rules, Track B1 still misses 43.8% of melanomas (recall = 56.2%). Adding class weights (Track C) raises Malignant Recall to 80.9%, and adding Focal focusing (gamma = 2 in Track B) raises it further to 86.0% (a +5.10 pp gain over Track C where the 95% bootstrap interval [+3.29, +6.99] pp excludes zero, paired t(4) = 3.12, p = 0.035; and +14.09 pp over Track B1, 95% CI: [+11.37, +17.19] pp, p = 0.0004). Crucially, Table 3 shows that Malignant PR-AUC (72.4% vs. 72.7%) and Macro AUROC (94.2% vs. 94.8%) are virtually identical between Track B and Track C, and slightly below Track B1 (76.1% and 95.3%). This indicates that once the backbone is fine-tuned, the primary effect of Focal Loss and class weighting is shifting the classifier's operating point (decision boundary) toward higher malignant sensitivity rather than improving underlying ranking separability. This operating-point shift comes with a clear precision-recall trade-off: 17.1% of benign NV are flagged as MEL in Track B (Figure 2b), lowering Malignant Specificity from 90.4% (B1) and 82.6% (C) to 78.2% and Malignant Precision to 49.2%.",
        bold_prefix="Disentangling Feature Adaptation vs. Operating-Point Shift. ")

    add_body_p(doc,
        "What Worked, What Did Not, and Lessons Learned. Grouped splitting and per-class recall worked well to expose baseline bias without leakage. What did not work initially was combining unsmoothed inverse-frequency weights (1/N_c) with gamma = 2.0, which caused early gradient instability and collapsed predictions toward DF and VASC (observed in initial diagnostic confusion matrices); power-smoothed weights (N_c^-0.5) plus a 3-epoch head warm-up resolved this issue. The central lesson from our five-track ablation is that progressive fine-tuning provides the feature separability (AUROC/PR-AUC), while Focal Loss adjusts the decision threshold behavior to prioritise malignant recall.",
        bold_prefix="Reflection. ")

    # ==================== SECTION 7 ====================
    add_heading_styled(doc, "7. Remaining Work and Risk Management", level=1)
    add_body_p(doc,
        "Three tasks remain for Weeks 4–5: (1) completing the gamma in {0, 1, 2, 3} sensitivity sweep on Track B (gamma = 0 is already evaluated as Track C); (2) conducting post-hoc threshold tuning on validation P_mal and probability calibration (Expected Calibration Error) to test whether Track B1 or Track B can achieve ~85% malignant sensitivity with higher specificity than default argmax; and (3) generating Grad-CAM maps to verify attention on lesion structures rather than skin markings. Risks of small-class noise (DF, VASC) and false-positive trade-offs are managed via group aggregation and threshold analysis (Appendix A).")

    # ==================== REFERENCES ====================
    add_heading_styled(doc, "References", level=1)
    refs = [
        "Australian Institute of Health and Welfare. (2022). Cancer data in Australia (Cat. no. CAN 122). AIHW.",
        "Cancer Council Australia. (2023). Skin cancer incidence and mortality statistics. https://www.cancer.org.au/cancer-information/types-of-cancer/skin-cancer",
        "Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115–118.",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778).",
        "Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2980–2988).",
        "Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5(1), 180161."
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(1.5)
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = 'Calibri'
        r_ref.font.size = Pt(8.5)

    # ==================== APPENDICES ====================
    doc.add_page_break()
    add_heading_styled(doc, "Appendix A: Revised Project Timeline and Milestone Status", level=1)
    ta_data = [
        ["Phase", "Task Description & Deliverables", "Timeframe", "Status"],
        ["Phase 1", "EDA, lesion_id StratifiedGroupKFold split (splits.csv), DataLoaders", "Week 1", "Done (100%)"],
        ["Phase 2", "Track A Baseline (Frozen ResNet-50 + CE) across 5 seeds", "Week 2", "Done (100%)"],
        ["Phase 3", "Focal Loss unit tests, smoothed alpha, Track B & ablations (B1, B2, C) across 5 seeds", "Week 3", "Done (100%)"],
        ["Phase 4", "Test evaluation, paired t-tests, bootstrap CIs, confusion matrices; gamma sweep remaining", "Week 4", "In Progress (85%)"],
        ["Phase 4b", "Calibration (ECE), validation threshold tuning on P_mal, Grad-CAM inspection", "Week 4–5", "Planned"],
        ["Phase 5", "Final report synthesis and documentation", "Week 5", "Not Started"]
    ]
    ta = doc.add_table(rows=len(ta_data), cols=4)
    for r_i, row in enumerate(ta_data):
        for c_i, val in enumerate(row):
            ta.cell(r_i, c_i).text = val
    format_academic_table(ta,
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        [0.8, 3.9, 1.0, 1.2])

    add_heading_styled(doc, "Appendix B: Image Counts per Lesion-Grouped Split (Zero Lesion Overlap)", level=1)
    tb_data = [
        ["Class", "Clinical Group", "Train (70.0%)", "Validation (15.0%)", "Test (15.0%)", "Total"],
        ["NV", "Benign", "4,695 (70.0%)", "1,005 (15.0%)", "1,005 (15.0%)", "6,705"],
        ["MEL", "Malignant", "778 (69.9%)", "167 (15.0%)", "168 (15.1%)", "1,113"],
        ["BKL", "Benign", "770 (70.1%)", "165 (15.0%)", "164 (14.9%)", "1,099"],
        ["BCC", "Malignant", "361 (70.2%)", "77 (15.0%)", "76 (14.8%)", "514"],
        ["AKIEC", "Malignant (Pre-mal.)", "230 (70.3%)", "49 (15.0%)", "48 (14.7%)", "327"],
        ["VASC", "Benign", "99 (69.7%)", "21 (14.8%)", "22 (15.5%)", "142"],
        ["DF", "Benign", "80 (69.6%)", "17 (14.8%)", "18 (15.7%)", "115"],
        ["Total", "1,954 Mal. / 8,061 Ben.", "7,013 (5,229 lesions)", "1,501 (1,120 lesions)", "1,501 (1,121 lesions)", "10,015 (7,470)"]
    ]
    tb = doc.add_table(rows=len(tb_data), cols=6)
    for r_i, row in enumerate(tb_data):
        for c_i, val in enumerate(row):
            tb.cell(r_i, c_i).text = val
    format_academic_table(tb,
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
        [0.7, 1.5, 1.35, 1.35, 1.35, 0.95])

    add_heading_styled(doc, "Appendix C: Experimental Matrix and Track B Malignant-Recall Comparison", level=1)
    tc_data = [
        ["Track", "Loss Function", "Class Weighting (α)", "Backbone Regime", "Seeds", "Track B's Malignant-Recall Gain Over This Track (95% Bootstrap CI)"],
        ["A (Baseline)", "Cross-Entropy (γ = 0)", "None (Uniform)", "Frozen (BN in eval mode)", "5", "+28.99 pp (95% CI: [+24.79, +33.08] pp; paired p = 0.0007)"],
        ["B1 (Ablation 1)", "Cross-Entropy (γ = 0)", "None (Uniform)", "Progressive FT (head -> layer4 -> layer3)", "5", "+14.09 pp (95% CI: [+11.37, +17.19] pp; paired p = 0.0004)"],
        ["B2 (Ablation 2)", "Focal Loss (γ = 2.0)", "Smoothed inv-freq + boost", "Frozen (BN in eval mode)", "5", "+11.97 pp (95% CI: [+8.63, +15.21] pp; paired p = 0.0211)"],
        ["C (Ablation 3)", "Weighted CE (γ = 0)", "Smoothed inv-freq + boost", "Progressive FT (head -> layer4 -> layer3)", "5", "+5.10 pp (95% CI: [+3.29, +6.99] pp; paired p = 0.0351)"],
        ["B (Proposed)", "Focal Loss (γ = 2.0)", "Smoothed inv-freq + boost", "Progressive FT (head -> layer4 -> layer3)", "5", "Reference Track (86.0 ± 1.3% Malignant Recall)"]
    ]
    tc = doc.add_table(rows=len(tc_data), cols=6)
    for r_i, row in enumerate(tc_data):
        for c_i, val in enumerate(row):
            tc.cell(r_i, c_i).text = val
    format_academic_table(tc,
        [WD_ALIGN_PARAGRAPH.LEFT]*4 + [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        [0.95, 1.15, 1.25, 1.45, 0.45, 1.95])

    add_body_p(doc,
        "Shared Hyperparameters: ResNet-50 (IMAGENET1K_V1), 224×224 input, batch size 64, Adam, weight decay 1e-4, Dropout(0.3), max 18 epochs, early stopping patience 6 on 0.5*Macro-F1 + 0.5*Malignant_Recall, learning rates 1e-3 (head), 1e-4 (layer4), 5e-5 (layer3), alpha = [0.22, 0.69, 0.55, 1.01, 1.27, 1.54, 1.72] for (NV, MEL, BKL, BCC, AKIEC, VASC, DF). Planned Track B sweep: gamma in {0, 1, 2, 3}.",
        bold_prefix="Common Protocol: ", space_after=2)

    doc.save(OUTPUT_DOCX)
    print(f"[Success] 已生成全面修正版 Word 报告: {OUTPUT_DOCX}")


if __name__ == "__main__":
    build_report()