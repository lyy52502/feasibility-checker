"""
evaluate_advanced.py
====================
Generates advanced evaluation figures for thesis Chapter 5:

  Figure 5:  Confusion matrix + classification report
             (precision, recall, F1 for feasibility checker)
  Figure 5B: Precision / Recall / F1 bar chart
  Figure 6:  Constraint satisfaction of generated alternatives
  Figure 7:  t-SNE latent space visualisation
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from feasibility_checker import FeasibilityChecker

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

THRESHOLD = 0.35   # align with Figure 2


# ── Load data helpers ──────────────────────────────────────────────────────────

def load_all(checker):
    df   = pd.read_csv("features.csv")
    with open("column_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    cond_cols = [c for c in meta['condition_cols'] if c in df.columns]
    tgt_cols  = [c for c in meta['target_cols']    if c in df.columns]
    C = torch.tensor(df[cond_cols].values, dtype=torch.float32)
    X = torch.tensor(df[tgt_cols].values,  dtype=torch.float32)
    return C, X, cond_cols, tgt_cols, df


def get_errors_and_latents(checker, C, X):
    """Run all samples through the model, return errors and latent vectors."""
    checker.model.eval()
    errors, latents, feas_scores = [], [], []
    with torch.no_grad():
        for i in range(len(C)):
            c = C[i:i+1]
            x = X[i:i+1]
            x_recon, mu, logvar, feas = checker.model(x, c)
            err = torch.nn.functional.mse_loss(
                x_recon, x, reduction='mean').item()
            errors.append(err)
            latents.append(mu.squeeze(0).numpy())
            feas_scores.append(feas.item())
    return np.array(errors), np.array(latents), np.array(feas_scores)


def make_infeasible_samples(C, X, n=200, seed=42):
    """
    Synthetically create infeasible configurations by randomising
    target features — these will never match real historical patterns.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(C), size=n, replace=True)
    C_inf = C[idx].clone()
    # Randomise X with values far outside the [0,1] normal range
    X_inf = torch.tensor(
        rng.uniform(1.5, 5.0, size=(n, X.shape[1])),
        dtype=torch.float32
    )
    return C_inf, X_inf


# ── Figure 5 — Confusion Matrix ────────────────────────────────────────────────

def fig5_confusion_matrix(checker, C, X):
    print("  Building Figure 5: Confusion matrix...")

    # Real historical orders = feasible ground truth
    errors_real, _, _ = get_errors_and_latents(checker, C, X)

    # Synthetic infeasible orders
    C_inf, X_inf = make_infeasible_samples(C, X, n=300)
    errors_inf, _, _ = get_errors_and_latents(checker, C_inf, X_inf)

    # Ground truth labels:  1 = feasible, 0 = infeasible
    y_true = np.concatenate([
        np.ones(len(errors_real)),
        np.zeros(len(errors_inf))
    ])

    # Predicted labels based on threshold
    y_pred = np.concatenate([
        (errors_real < THRESHOLD).astype(int),
        (errors_inf  < THRESHOLD).astype(int)
    ])

    # ── Confusion matrix plot ──────────────────────────────────────
    cm  = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Not Feasible', 'Feasible']
    )
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title("Figure 5a: Confusion Matrix\n"
                      f"(threshold = {THRESHOLD})")

    # ── Classification metrics bar chart ──────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=['Not Feasible', 'Feasible'],
        output_dict=True
    )

    metrics   = ['precision', 'recall', 'f1-score']
    feasible  = [report['Feasible'][m]     for m in metrics]
    infeasible= [report['Not Feasible'][m] for m in metrics]
    x         = np.arange(len(metrics))
    width     = 0.35

    axes[1].bar(x - width/2, feasible,   width, label='Feasible',
                color='steelblue', alpha=0.85)
    axes[1].bar(x + width/2, infeasible, width, label='Not Feasible',
                color='tomato',    alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Figure 5b: Precision / Recall / F1\n"
                      "Feasibility Checker Performance")
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in axes[1].patches:
        h = bar.get_height()
        axes[1].annotate(f'{h:.2f}',
                         xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("fig5_confusion_and_f1.png")
    plt.close()

    # ── Print report ───────────────────────────────────────────────
    print(f"\n  Classification Report (threshold = {THRESHOLD}):")
    print("  " + "-" * 50)
    rpt = classification_report(
        y_true, y_pred,
        target_names=['Not Feasible', 'Feasible']
    )
    for line in rpt.splitlines():
        print("  " + line)

    acc = (y_pred == y_true).mean()
    print(f"\n  Overall Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Feasible in data : {int(y_true.sum())} / {len(y_true)}")
    print("  ✓ Saved fig5_confusion_and_f1.png")

    return report, acc


# ── Figure 6 — Constraint Satisfaction ────────────────────────────────────────

def fig6_constraint_satisfaction(checker, C, X, tgt_cols, df_raw):
    print("\n  Building Figure 6: Constraint satisfaction...")

    # Use the same infeasible example from run_example.py
    with open("column_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    cond_cols = [c for c in meta['condition_cols'] if c in df_raw.columns]

    mill = dict(zip(cond_cols, C[0].numpy()))

    infeasible_roll = {
        'TotalNoOfZones': 999, 'NoOf52mmZones': 500,
        'NoOf26mmZones': 500, 'BearingCentreDistance': 9999,
        'LongJournalLength': 9999, 'RaSTR': 99.0, 'RaStrip': 99.0,
        'LongJournals': 1, 'HighTempCon': 1,
        'RollTypeID': 6, 'RollDiameterID': 5, 'BearingTypeID': 10,
        'BearingHouseID': 6, 'BearingDiameterID': 7, 'SealingTypeID': 3,
        'ShaftMaterialID': 3, 'SignalTransUnitID': 45,
        'RollSurfaceID': 12, 'RollRepairID': 2,
        'StripCooling': 'Kerosene', 'GearBox': 'Yes',
    }

    result = checker.check(mill, infeasible_roll, n_alternatives=3)
    alts   = result.get('alternatives', [])

    if not alts:
        print("  ⚠ No alternatives returned — skipping Figure 6")
        return

    # For each alternative, compute how many columns are within
    # the 5th–95th percentile of training data (= "within normal range")
    X_np = X.numpy()

    # Compute training data percentiles per column
    p05 = np.percentile(X_np, 5,  axis=0)
    p95 = np.percentile(X_np, 95, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.22
    x         = np.arange(len(tgt_cols))
    colours   = ['steelblue', 'darkorange', 'seagreen']

    for alt_i, alt in enumerate(alts):
        in_range = []
        for j, col in enumerate(tgt_cols):
            val = alt.get(col, None)
            if val is None:
                in_range.append(0)
                continue
            # Re-normalise for comparison
            if col in checker.label_encoders:
                enc = checker.label_encoders[col]
                try:
                    encoded = list(enc.classes_).index(str(val))
                    norm_val = encoded / max(len(enc.classes_) - 1, 1)
                except ValueError:
                    norm_val = 0.5
            else:
                try:
                    scale  = checker.target_scaler.data_range_[j]
                    minval = checker.target_scaler.data_min_[j]
                    norm_val = (float(val) - minval) / (scale + 1e-9)
                except Exception:
                    norm_val = float(val)

            in_range.append(
                1 if p05[j] <= norm_val <= p95[j] else 0
            )

        satisfaction_pct = sum(in_range) / len(tgt_cols) * 100
        oid = alt.get('OrderID', f'Alt {alt_i+1}')

        # Stacked bar: green = in range, red = out of range
        in_count  = sum(in_range)
        out_count = len(in_range) - in_count

        ax.bar(alt_i, in_count,
               label=f'OrderID {oid}: {satisfaction_pct:.0f}% satisfied',
               color=colours[alt_i], alpha=0.85)
        ax.bar(alt_i, out_count, bottom=in_count,
               color='lightcoral', alpha=0.5)

        ax.text(alt_i, len(tgt_cols) + 0.5,
                f'{satisfaction_pct:.0f}%',
                ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(alts)))
    ax.set_xticklabels(
        [f"Alt {i+1}\n(OrderID={a.get('OrderID','?')})"
         for i, a in enumerate(alts)]
    )
    ax.set_ylim(0, len(tgt_cols) + 2)
    ax.set_ylabel("Number of Target Columns")
    ax.set_title("Figure 6: Constraint Satisfaction of Suggested Alternatives\n"
                 "(columns within 5th–95th percentile of training data)")

    in_patch  = mpatches.Patch(color='steelblue', label='Within normal range')
    out_patch = mpatches.Patch(color='lightcoral', alpha=0.5,
                               label='Outside normal range')
    ax.legend(handles=[in_patch, out_patch], loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig6_constraint_satisfaction.png")
    plt.close()
    print("  ✓ Saved fig6_constraint_satisfaction.png")


# ── Figure 7 — t-SNE Latent Space ─────────────────────────────────────────────

def fig7_tsne_latent_space(checker, C, X, df_raw):
    print("\n  Building Figure 7: t-SNE latent space (takes ~2 min)...")

    errors, latents, _ = get_errors_and_latents(checker, C, X)
    feasible_mask      = errors < THRESHOLD

    # ── Run t-SNE ──────────────────────────────────────────────────
    tsne    = TSNE(n_components=2, perplexity=30,
                  random_state=42, n_iter=1000)
    latents_2d = tsne.fit_transform(latents)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: colour by feasibility
    axes[0].scatter(
        latents_2d[feasible_mask,  0],
        latents_2d[feasible_mask,  1],
        c='steelblue', alpha=0.5, s=12,
        label=f'Feasible ({feasible_mask.sum()})'
    )
    axes[0].scatter(
        latents_2d[~feasible_mask, 0],
        latents_2d[~feasible_mask, 1],
        c='tomato', alpha=0.7, s=20,
        label=f'Above threshold ({(~feasible_mask).sum()})'
    )
    axes[0].set_title("Figure 7a: Latent Space — Feasibility")
    axes[0].set_xlabel("t-SNE dimension 1")
    axes[0].set_ylabel("t-SNE dimension 2")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Panel B: colour by reconstruction error (continuous)
    sc = axes[1].scatter(
        latents_2d[:, 0], latents_2d[:, 1],
        c=np.clip(errors, 0, 0.5),
        cmap='RdYlGn_r', alpha=0.6, s=12
    )
    plt.colorbar(sc, ax=axes[1], label='Reconstruction Error')
    axes[1].set_title("Figure 7b: Latent Space — Reconstruction Error")
    axes[1].set_xlabel("t-SNE dimension 1")
    axes[1].set_ylabel("t-SNE dimension 2")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle("Figure 7: CVAE Latent Space Visualisation (t-SNE)\n"
                 "Each point = one historical order",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("fig7_tsne_latent_space.png")
    plt.close()
    print("  ✓ Saved fig7_tsne_latent_space.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Advanced Evaluation — Thesis Figures 5, 6, 7")
    print("=" * 60 + "\n")

    print("  Loading model...")
    checker = FeasibilityChecker("model.pt")
    C, X, cond_cols, tgt_cols, df_raw = load_all(checker)
    print(f"  ✓ {len(C)} samples loaded\n")

    report, acc = fig5_confusion_matrix(checker, C, X)
    fig6_constraint_satisfaction(checker, C, X, tgt_cols, df_raw)
    fig7_tsne_latent_space(checker, C, X, df_raw)

    # ── Final thesis metrics summary ───────────────────────────────
    print("\n" + "=" * 60)
    print("  THESIS METRICS SUMMARY — copy to Chapter 5")
    print("=" * 60)

    feas_p  = report['Feasible']['precision']
    feas_r  = report['Feasible']['recall']
    feas_f1 = report['Feasible']['f1-score']
    inf_p   = report['Not Feasible']['precision']
    inf_r   = report['Not Feasible']['recall']
    inf_f1  = report['Not Feasible']['f1-score']

    print(f"""
  Feasibility Checker Performance:
  ─────────────────────────────────────────────────────
  Class          Precision   Recall    F1-Score
  Feasible        {feas_p:.3f}      {feas_r:.3f}     {feas_f1:.3f}
  Not Feasible    {inf_p:.3f}      {inf_r:.3f}     {inf_f1:.3f}
  Overall Acc.    {acc:.3f}

  Key findings:
  - The CVAE feasibility checker achieved {acc*100:.1f}% accuracy
  - Feasible configurations: F1 = {feas_f1:.3f}
  - Infeasible configurations: F1 = {inf_f1:.3f}
  - Decision boundary: reconstruction error threshold = {THRESHOLD}
    """)

    print("  Figures saved:")
    print("  fig5_confusion_and_f1.png     → Chapter 5.3")
    print("  fig6_constraint_satisfaction.png → Chapter 5.4")
    print("  fig7_tsne_latent_space.png    → Chapter 4.3")


if __name__ == "__main__":
    main()