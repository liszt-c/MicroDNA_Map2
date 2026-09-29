"""
scripts/verify.py - 模型全量量化评估报告与混淆矩阵绘制

特性:
  1. 支持评估单折模型在专属 Held-out 测试染色体上的泛化性能 (--fold 1..5)
  2. 支持评估 5 折集成模型 (Ensemble)
  3. 自动计算 ACC, AUC, PR-AUC, Sensitivity/Recall, Specificity, F1, MCC
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CHROM_FOLDS,
    CV_MODEL_DIR,
    DEFAULT_BATCH_SIZE,
    METRICS_DIR,
    MODEL_DIR,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
)
from src.dataloader import (
    ECC_LABEL,
    build_chromosome_cv_loaders,
    build_full_loader,
)
from scripts.predict import load_model
from src.utils import setup_logger

logger = setup_logger('verify')
CLASS_NAMES = ['otherDNA', 'eccDNA']


def collect_predictions(model, loader, device, ecc_class_index: int):
    probs, labels = [], []
    with torch.no_grad():
        for inputs, lbs in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs.append(torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy())
            labels.append(lbs.numpy())
    y_prob = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    return y_true, y_prob


def plot_metrics(y_true, y_prob, y_pred, cm, report_txt, threshold, auc_roc, pr_auc, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)

    fig = plt.figure(figsize=(14, 11))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_roc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, ls='--')
    ax1.set_xlim([0.0, 1.0]); ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve')
    ax1.legend(loc='lower right')

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(rec_c, prec_c, color='steelblue', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
    ax2.set_xlim([0.0, 1.0]); ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')

    ax3 = plt.subplot(2, 2, 3)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title(f'Confusion Matrix (threshold={threshold})')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    ticks = np.arange(len(CLASS_NAMES))
    ax3.set_xticks(ticks); ax3.set_yticks(ticks)
    ax3.set_xticklabels(CLASS_NAMES); ax3.set_yticklabels(CLASS_NAMES)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    ax4.text(0.0, 0.95, report_txt, fontfamily='monospace', fontsize=9, va='top')
    ax4.set_title('Classification Report')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure saved -> {out_png}")


def main():
    p = argparse.ArgumentParser(description="Evaluate MicroDNA Map Model")
    p.add_argument('--model', default=None, help="模型路径 (.pth 或包含 5 折权重的目录)")
    p.add_argument('--fold', type=int, default=None, help="仅在指定 Fold 的独立测试染色体上评估 (1 到 5)")
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    p.add_argument('--ecc-fasta', default=None)
    p.add_argument('--other-fasta', default=None)
    p.add_argument('--output-dir', default=str(METRICS_DIR))
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model) if args.model else (CV_MODEL_DIR if CV_MODEL_DIR.exists() else MODEL_DIR / "best_model.pth")
    model, ecc_class_index = load_model(model_path, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ecc_fa = Path(args.ecc_fasta) if args.ecc_fasta else PROCESSED_DATA_DIR / "eccDNA.fa"
    other_fa = Path(args.other_fasta) if args.other_fasta else PROCESSED_DATA_DIR / "otherDNA.fa"

    # 如果指定了 --fold，构建专属的 Held-out 测试集
    if args.fold is not None:
        fold_idx = int(args.fold) - 1
        logger.info(f"Evaluating exclusively on Held-out Fold {args.fold} Test Chromosomes: {CHROM_FOLDS[fold_idx]}...")
        _, _, loader, _, _, _, _ = build_chromosome_cv_loaders(
            test_fold=fold_idx, batch_size=args.batch_size, num_workers=args.num_workers,
            ecc_path=ecc_fa, other_path=other_fa
        )
        tag = f"_fold{args.fold}"
    else:
        logger.info("Evaluating on Full Dataset...")
        loader, _ = build_full_loader(batch_size=args.batch_size, num_workers=args.num_workers, ecc_path=ecc_fa, other_path=other_fa)
        tag = "_full"

    y_true, y_prob = collect_predictions(model, loader, device, ecc_class_index)
    y_pred = (y_prob >= args.threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    _p, _r, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(_r, _p)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=[0, 1], zero_division=0)
    
    print("\n" + "=" * 64)
    print(f"EVALUATION METRICS REPORT {tag.upper()}")
    print("=" * 64)
    print(f"Total Samples : {y_true.size} (eccDNA: {int((y_true==1).sum())}, otherDNA: {int((y_true==0).sum())})")
    print(f"Accuracy      : {acc:.4f}")
    print(f"AUC-ROC       : {auc_roc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"Specificity   : {spec:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"MCC           : {mcc:.4f}")
    print(f"Confusion Mtx : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 64 + "\n")

    report_path = out_dir / f"evaluation_report{tag}.txt"
    report_path.write_text(f"ACC: {acc:.4f}\nAUC: {auc_roc:.4f}\nF1: {f1:.4f}\nRecall: {rec:.4f}\nPrecision: {prec:.4f}\nMCC: {mcc:.4f}\n\n{report_txt}")
    
    if not args.no_plot:
        out_png = out_dir / f"metrics_plot{tag}.png"
        plot_metrics(y_true, y_prob, y_pred, cm, report_txt, args.threshold, auc_roc, pr_auc, out_png)


if __name__ == '__main__':
    main()