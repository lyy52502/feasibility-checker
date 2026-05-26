"""
visualise.py
============
Generates all thesis figures in ABB brand style.

ABB Palette:
  Red       #FF000F   — alerts, thresholds, NOT feasible
  DarkGray  #333333   — primary data series, text
  MidGray   #696969   — secondary series, grid
  LightGray #E8E8E8   — backgrounds, fills
  NearBlack #1A1A1A   — titles, axis labels

Figures:
  Fig 01  Training & validation loss curve
  Fig 02  Reconstruction error distribution + threshold
  Fig 03  Feasibility head vs reconstruction error (scatter)
  Fig 04  Per-column RMSE bar chart  (5-fold CV)
  Fig 05  Confusion matrix  +  Precision / Recall / F1 bar chart
  Fig 06  Constraint satisfaction of suggested alternatives
  Fig 07  t-SNE latent space (2-panel: feasibility + error)
  Fig 08  Predicted vs actual scatter  (top 6 numeric columns)
  Fig 09  Training experiment comparison  (3 runs)
  Fig 10  Condition–target correlation heatmap
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from feasibility_checker import FeasibilityChecker


# ══════════════════════════════════════════════════════════════════════════════
#  ABB STYLE SETUP
# ═════════════════════════════════════════════════════��════════════════════════

ABB = {
    'red':        '#FF000F',
    'dark_gray':  '#333333',
    'mid_gray':   '#696969',
    'light_gray': '#E8E8E8',
    'near_black': '#1A1A1A',
    'white':      '#FFFFFF',
    'red_light':  '#FF6B6B',
    'red_dark':   '#CC0000',
    'grid':       '#D5D5D5',
}


def set_abb_style():
    plt.rcParams.update({
        'font.family':          'sans-serif',
        'font.sans-serif':      ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':            10,
        'axes.titlesize':       12,
        'axes.titleweight':     'bold',
        'axes.titlepad':        14,
        'axes.labelsize':       10,
        'axes.labelcolor':      ABB['near_black'],
        'axes.facecolor':       ABB['white'],
        'axes.edgecolor':       ABB['dark_gray'],
        'axes.linewidth':       0.8,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.grid':            True,
        'grid.color':           ABB['grid'],
        'grid.linewidth':       0.5,
        'grid.linestyle':       '-',
        'grid.alpha':           1.0,
        'xtick.color':          ABB['dark_gray'],
        'ytick.color':          ABB['dark_gray'],
        'xtick.labelsize':      9,
        'ytick.labelsize':      9,
        'xtick.direction':      'out',
        'ytick.direction':      'out',
        'xtick.major.size':     4,
        'ytick.major.size':     4,
        'legend.fontsize':      9,
        'legend.frameon':       True,
        'legend.framealpha':    0.95,
        'legend.edgecolor':     ABB['light_gray'],
        'legend.fancybox':      False,
        'figure.facecolor':     ABB['white'],
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.facecolor':    ABB['white'],
        'savefig.pad_inches':   0.15,
        'lines.linewidth':      1.8,
        'lines.solid_capstyle': 'round',
    })


def add_abb_branding(fig, subtitle=None):
    """Red bottom bar + optional subtitle — consistent across all figures."""
    fig.add_artist(
        plt.Line2D([0.0, 1.0], [0.0, 0.0],
                   transform=fig.transFigure,
                   color=ABB['red'], linewidth=3.5, clip_on=False)
    )
    if subtitle:
        fig.text(0.98, 0.01, subtitle,
                 ha='right', va='bottom', fontsize=7.5,
                 color=ABB['mid_gray'], style='italic')


def style_axis(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, color=ABB['near_black'], pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=ABB['near_black'], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=ABB['near_black'], labelpad=8)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(ABB['dark_gray'])
        ax.spines[spine].set_linewidth(0.8)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_features(checker):
    df = pd.read_csv("features.csv")
    with open("column_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    cond_cols = [c for c in meta['condition_cols'] if c in df.columns]
    tgt_cols  = [c for c in meta['target_cols']    if c in df.columns]
    C = torch.tensor(df[cond_cols].values, dtype=torch.float32)
    X = torch.tensor(df[tgt_cols].values,  dtype=torch.float32)
    return C, X, cond_cols, tgt_cols, df


def compute_errors_latents(checker, C, X):
    checker.model.eval()
    errors, latents, feas_scores = [], [], []
    with torch.no_grad():
        for i in range(len(C)):
            c, x = C[i:i+1], X[i:i+1]
            xr, mu, _, feas = checker.model(x, c)
            errors.append(
                torch.nn.functional.mse_loss(xr, x, reduction='mean').item())
            latents.append(mu.squeeze(0).numpy())
            feas_scores.append(feas.item())
    return np.array(errors), np.array(latents), np.array(feas_scores)


def make_infeasible_samples(C, X, n=300, seed=42):
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(C), size=n, replace=True)
    C_inf = C[idx].clone()
    X_inf = torch.tensor(
        rng.uniform(1.5, 6.0, size=(n, X.shape[1])),
        dtype=torch.float32)
    return C_inf, X_inf


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Training Curve
# ══════════════════════════════════════════════════════════════════════════════

def fig01_training_curve():
    if not pd.io.common.file_exists("training_history.csv"):
        print("  ⚠  training_history.csv not found — skipping Fig 01"); return

    df = pd.read_csv("training_history.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(df['epoch'], df['train'],
            color=ABB['dark_gray'], lw=1.8, label='Training loss', zorder=3)
    ax.plot(df['epoch'], df['val'],
            color=ABB['red'], lw=1.8, ls='--', label='Validation loss', zorder=3)
    ax.fill_between(df['epoch'], df['train'], df['val'],
                    alpha=0.06, color=ABB['red'], zorder=1)

    best_idx   = df['val'].idxmin()
    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_val   = df.loc[best_idx, 'val']
    ax.axvline(best_epoch, color=ABB['red'], lw=1.0, ls=':', alpha=0.7)
    ax.annotate(
        f'Best val: {best_val:.4f}\n(epoch {best_epoch})',
        xy=(best_epoch, best_val),
        xytext=(best_epoch - len(df) * 0.2, best_val + df['val'].max() * 0.08),
        fontsize=8.5, color=ABB['red_dark'],
        arrowprops=dict(arrowstyle='->', color=ABB['red_dark'],
                        lw=1.2, connectionstyle='arc3,rad=0.2'),
        bbox=dict(boxstyle='round,pad=0.3', fc=ABB['white'],
                  ec=ABB['light_gray'], alpha=0.95))

    style_axis(ax, 'CVAE Training and Validation Loss',
               'Epoch', 'Loss  (MSE + β·KL)')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    add_abb_branding(fig, 'Fig 01 — Training convergence')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig01_training_curve.png"); plt.close()
    print("  ✓ Saved fig01_training_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Reconstruction Error Distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig02_error_distribution(errors, threshold):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    clip_max = min(errors.max(), threshold * 6)
    bins     = np.linspace(0, clip_max, 55)
    below    = errors[errors <  threshold]
    above    = errors[(errors >= threshold) & (errors <= clip_max)]
    pct      = len(below) / len(errors) * 100

    ax.hist(below, bins=bins, color=ABB['dark_gray'], alpha=0.85,
            label=f'Feasible  ({len(below):,} orders — {pct:.1f}%)', zorder=3)
    ax.hist(above, bins=bins, color=ABB['red_light'], alpha=0.80,
            label=f'Above threshold  ({len(above):,} orders)', zorder=3)
    ax.axvline(threshold, color=ABB['red'], lw=2.0, ls='--', zorder=4,
               label=f'Threshold = {threshold}')

    ymax = ax.get_ylim()[1]
    ax.annotate(
        f'{pct:.1f}% of historical\norders ≤ threshold',
        xy=(threshold * 0.45, ymax * 0.78),
        fontsize=8.5, color=ABB['dark_gray'], ha='center',
        bbox=dict(boxstyle='round,pad=0.4', fc=ABB['light_gray'],
                  ec='none', alpha=0.9))

    style_axis(ax, 'Reconstruction Error Distribution — All Historical Orders',
               'Reconstruction Error (MSE)', 'Number of Orders')
    ax.legend()

    add_abb_branding(fig, 'Fig 02 — Threshold calibration')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig02_error_distribution.png"); plt.close()
    print("  ✓ Saved fig02_error_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Feasibility Scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig03_feasibility_scatter(errors, feas_scores, threshold):
    feasible = errors < threshold
    fig, ax  = plt.subplots(figsize=(9, 4.5))

    ax.scatter(errors[feasible],  feas_scores[feasible],
               color=ABB['dark_gray'], alpha=0.40, s=14, lw=0,
               label=f'Feasible  ({feasible.sum()})', zorder=3)
    ax.scatter(errors[~feasible], feas_scores[~feasible],
               color=ABB['red'], alpha=0.65, s=18, lw=0,
               label=f'Above threshold  ({(~feasible).sum()})', zorder=4)
    ax.axvline(threshold, color=ABB['red'], lw=1.8, ls='--', zorder=5,
               label=f'Threshold = {threshold}')
    ax.axhline(0.5, color=ABB['mid_gray'], lw=1.0, ls=':', zorder=2,
               label='Sigmoid = 0.5')

    style_axis(ax, 'Feasibility Head Output vs Reconstruction Error',
               'Reconstruction Error (MSE)',
               'Feasibility Head Output (sigmoid)')
    ax.set_xlim(-0.01, threshold * 5)
    ax.set_ylim(0.44, 1.06)
    ax.legend(loc='lower right')

    add_abb_branding(fig, 'Fig 03 — Feasibility decision boundary')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig03_feasibility_scatter.png"); plt.close()
    print("  ✓ Saved fig03_feasibility_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Per-Column RMSE
# ══════════════════════════════════════════════════════════════════════════════

def fig04_per_column_rmse():
    if not pd.io.common.file_exists("evaluation_results.csv"):
        print("  ⚠  evaluation_results.csv not found — run evaluate.py first")
        return

    df = pd.read_csv("evaluation_results.csv")

    # Only keep numerical variables
    numerical_cols = [
        'BearingCentreDistance',
        'LongJournalLength',
        'RaSTR',
        'RaStrip',
        'TotalNoOfZones',
        'NoOf52mmZones',
        'NoOf26mmZones'
    ]

    df = (df[df['Column'].isin(numerical_cols)]
            .sort_values("RMSE (norm)", ascending=True)
            .reset_index(drop=True))

    colours = [ABB['red'] if v < 0.10
               else ABB['mid_gray'] if v < 0.20
               else ABB['dark_gray']
               for v in df['RMSE (norm)']]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df['Column'], df['RMSE (norm)'],
                   color=colours, height=0.65, edgecolor='none', zorder=3)

    ax.axvline(0.10, color=ABB['dark_gray'], lw=1.0, ls='--', alpha=0.6)
    ax.axvline(0.20, color=ABB['mid_gray'],  lw=1.0, ls='--', alpha=0.6)

    for bar, val in zip(bars, df['RMSE (norm)']):
        ax.text(val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left',
                fontsize=7.5, color=ABB['near_black'])

    ymax = len(df) - 0.3
    for x, lbl in [(0.10, '0.10'), (0.20, '0.20')]:
        ax.text(x, ymax, lbl, fontsize=7.5, ha='center', va='bottom',
                color=ABB['mid_gray'])

    good = mpatches.Patch(color=ABB['red'], label='Good  (< 0.10)')
    fair = mpatches.Patch(color=ABB['mid_gray'],  label='Fair   (0.10 – 0.20)')
    poor = mpatches.Patch(color=ABB['dark_gray'],       label='Poor  (> 0.20)')
    ax.legend(handles=[good, fair, poor], loc='lower right')

    style_axis(ax,
               'Per-Column RMSE — Target Numerical Variables\n'
               '(5-fold cross-validation, normalised scale)',
               'RMSE (normalised)', '')
    ax.set_xlim(0, df['RMSE (norm)'].max() * 1.20)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    add_abb_branding(fig, 'Fig 04 — Prediction accuracy per output variable')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig04_per_column_rmse.png"); plt.close()
    print("  ✓ Saved fig04_per_column_rmse.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Confusion Matrix + F1 Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def fig05_confusion_f1(checker, C, X, threshold):
    print("  Computing Figure 05 (confusion matrix)...")

    errors_real, _, _ = compute_errors_latents(checker, C, X)
    C_inf, X_inf      = make_infeasible_samples(C, X, n=300)
    errors_inf, _, _  = compute_errors_latents(checker, C_inf, X_inf)

    y_true = np.concatenate([np.ones(len(errors_real)),
                              np.zeros(len(errors_inf))])
    y_pred = np.concatenate([(errors_real < threshold).astype(int),
                              (errors_inf  < threshold).astype(int)])

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=['Not Feasible', 'Feasible'],
        output_dict=True)
    acc    = (y_pred == y_true).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: Confusion matrix ──────────────────────────────────
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Not Feasible', 'Feasible'])
    disp.plot(ax=axes[0], colorbar=False,
              cmap=plt.cm.Greys)
    # Re-colour text for readability
    for text in axes[0].texts:
        text.set_color(ABB['near_black'])
        text.set_fontsize(14)
        text.set_fontweight('bold')
    axes[0].set_title(f'Confusion Matrix\n(threshold = {threshold})',
                      color=ABB['near_black'], pad=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', color=ABB['near_black'])
    axes[0].set_ylabel('True Label',      color=ABB['near_black'])

    # ── Panel B: Precision / Recall / F1 ──────────────────────────
    metrics   = ['precision', 'recall', 'f1-score']
    labels    = ['Precision', 'Recall', 'F1-Score']
    feasible  = [report['Feasible'][m]     for m in metrics]
    infeasible= [report['Not Feasible'][m] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35

    b1 = axes[1].bar(x - w/2, feasible,   w, color=ABB['dark_gray'],
                     alpha=0.9, label='Feasible',     edgecolor='none')
    b2 = axes[1].bar(x + w/2, infeasible, w, color=ABB['red'],
                     alpha=0.9, label='Not Feasible', edgecolor='none')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.01,
                         f'{h:.2f}', ha='center', va='bottom',
                         fontsize=9, color=ABB['near_black'])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.20)
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Precision / Recall / F1\nOverall Accuracy: {acc:.1%}',
                      color=ABB['near_black'], pad=14, fontweight='bold')
    axes[1].legend()
    axes[1].axhline(1.0, color=ABB['light_gray'], lw=0.8, ls='--')

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    add_abb_branding(fig, 'Fig 05 — Classifier performance')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig05_confusion_f1.png"); plt.close()
    print("  ✓ Saved fig05_confusion_f1.png")

    # Print report
    print(f"\n  Accuracy: {acc:.1%}")
    print(f"  Feasible   — P: {report['Feasible']['precision']:.3f}  "
          f"R: {report['Feasible']['recall']:.3f}  "
          f"F1: {report['Feasible']['f1-score']:.3f}")
    print(f"  Infeasible — P: {report['Not Feasible']['precision']:.3f}  "
          f"R: {report['Not Feasible']['recall']:.3f}  "
          f"F1: {report['Not Feasible']['f1-score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Constraint Satisfaction of Alternatives
# ══════════════════════════════════════════════════════════════════════════════

def fig06_constraint_satisfaction(checker, C, X, tgt_cols):
    print("  Computing Figure 06 (constraint satisfaction)...")

    # Use the first mill condition row as the query condition
    mill   = dict(zip(checker.condition_cols, C[0].numpy()))
    X_np   = X.numpy()
    p05    = np.percentile(X_np,  5, axis=0)
    p95    = np.percentile(X_np, 95, axis=0)

    infeasible_roll = {
        'TotalNoOfZones': 999, 'NoOf52mmZones': 500, 'NoOf26mmZones': 500,
        'BearingCentreDistance': 9999, 'LongJournalLength': 9999,
        'RaSTR': 99.0, 'RaStrip': 99.0,
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
        print("  ⚠  No alternatives returned — skipping Fig 06"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: Stacked bar — in / out of range per alternative ──
    n_cols   = len(tgt_cols)
    bar_cols = [ABB['dark_gray'], ABB['mid_gray'], ABB['red_light']]

    in_counts, out_counts, oids = [], [], []
    for alt in alts:
        in_n = 0
        for j, col in enumerate(tgt_cols):
            val = alt.get(col, None)
            if val is None:
                continue
            if col in checker.label_encoders:
                enc = checker.label_encoders[col]
                try:
                    nv = list(enc.classes_).index(str(val)) / max(len(enc.classes_)-1,1)
                except ValueError:
                    nv = 0.5
            else:
                try:
                    sc  = checker.target_scaler
                    nv  = (float(val) - sc.data_min_[j]) / (sc.data_range_[j] + 1e-9)
                except Exception:
                    nv = float(val)
            if p05[j] <= nv <= p95[j]:
                in_n += 1
        in_counts.append(in_n)
        out_counts.append(n_cols - in_n)
        oids.append(alt.get('OrderID', '?'))

    x = np.arange(len(alts))
    for i, (inc, outc, col) in enumerate(zip(in_counts, out_counts, bar_cols)):
        pct = inc / n_cols * 100
        axes[0].bar(i, inc,  color=col,         alpha=0.90,
                    edgecolor='none', label=f'OrderID {oids[i]}: {pct:.0f}%')
        axes[0].bar(i, outc, bottom=inc,
                    color=ABB['light_gray'], alpha=0.70, edgecolor='none')
        axes[0].text(i, n_cols + 0.6, f'{pct:.0f}%',
                     ha='center', fontsize=12, fontweight='bold',
                     color=ABB['near_black'])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f'Alt {i+1}\nOrderID={oid}' for i, oid in enumerate(oids)],
        fontsize=9)
    axes[0].set_ylim(0, n_cols + 2.5)
    axes[0].set_ylabel('Number of Target Columns  (out of 21)')
    axes[0].set_title('Constraint Satisfaction per Alternative',
                      color=ABB['near_black'], pad=14, fontweight='bold')

    in_p  = mpatches.Patch(color=ABB['dark_gray'], label='Within normal range')
    out_p = mpatches.Patch(color=ABB['light_gray'], alpha=0.7,
                           label='Outside normal range')
    axes[0].legend(handles=[in_p, out_p], loc='lower right')

    # ── Panel B: Radar-style column-by-column comparison ──────────
    # Show for best alternative vs proposed infeasible
    best_alt = alts[0]
    cfg      = {k: v for k, v in best_alt.items()
                if k not in ('distance', 'OrderID')}

    # Normalise both to [0,1] for radar
    proposed_norm, hist_norm = [], []
    for j, col in enumerate(tgt_cols):
        # proposed
        raw_p = infeasible_roll.get(col, 0)
        if col in checker.label_encoders:
            enc = checker.label_encoders[col]
            try:
                pv = list(enc.classes_).index(str(raw_p)) / max(len(enc.classes_)-1,1)
            except ValueError:
                pv = 0.5
        else:
            try:
                sc = checker.target_scaler
                pv = min((float(raw_p) - sc.data_min_[j])/(sc.data_range_[j]+1e-9), 3.0)
            except Exception:
                pv = float(raw_p)
        proposed_norm.append(pv)

        # historical
        raw_h = cfg.get(col, 0)
        if col in checker.label_encoders:
            enc = checker.label_encoders[col]
            try:
                hv = list(enc.classes_).index(str(raw_h)) / max(len(enc.classes_)-1,1)
            except ValueError:
                hv = 0.5
        else:
            try:
                sc = checker.target_scaler
                hv = (float(raw_h) - sc.data_min_[j])/(sc.data_range_[j]+1e-9)
            except Exception:
                hv = float(raw_h)
        hist_norm.append(np.clip(hv, 0, 1))

    short_names = [c.replace('ID','').replace('NoOf','#') for c in tgt_cols]
    x2  = np.arange(len(tgt_cols))
    w2  = 0.38

    axes[1].bar(x2 - w2/2,
                np.clip(proposed_norm, 0, 1.5), w2,
                color=ABB['red'],       alpha=0.80,
                edgecolor='none', label='Proposed (infeasible)')
    axes[1].bar(x2 + w2/2,
                hist_norm, w2,
                color=ABB['dark_gray'], alpha=0.80,
                edgecolor='none', label=f'Alt 1  (OrderID {oids[0]})')
    axes[1].axhline(1.0, color=ABB['mid_gray'], lw=0.8, ls='--',
                    label='Normal range ceiling')

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(short_names, rotation=60,
                             ha='right', fontsize=7.5)
    axes[1].set_ylabel('Normalised Value')
    axes[1].set_title('Proposed vs Best Alternative\n(all 21 target variables)',
                      color=ABB['near_black'], pad=14, fontweight='bold')
    axes[1].legend(fontsize=8)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    add_abb_branding(fig, 'Fig 06 — Requirement 4: alternative suggestion quality')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig06_constraint_satisfaction.png"); plt.close()
    print("  ✓ Saved fig06_constraint_satisfaction.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — t-SNE Latent Space
# ══════════════════════════════════════════════════════════════════════════════

def fig07_tsne(errors, latents, threshold):
    print("  Running t-SNE for Figure 07 (takes ~2 min)...")
    feasible   = errors < threshold
    tsne       = TSNE(n_components=2, perplexity=30,
                      random_state=42, n_iter=1000, verbose=0)
    lat2d      = tsne.fit_transform(latents)

    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: feasibility coloured ─────────────────────────────
    axes[0].scatter(lat2d[feasible,  0], lat2d[feasible,  1],
                    color=ABB['dark_gray'], alpha=0.40, s=12, lw=0,
                    label=f'Feasible  ({feasible.sum()})', zorder=3)
    axes[0].scatter(lat2d[~feasible, 0], lat2d[~feasible, 1],
                    color=ABB['red'], alpha=0.75, s=20, lw=0,
                    label=f'Above threshold  ({(~feasible).sum()})', zorder=4)
    axes[0].set_title('Latent Space — Feasibility Labels',
                      color=ABB['near_black'], pad=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # ── Panel B: continuous error heatmap ─────────────────────────
    sc = axes[1].scatter(lat2d[:, 0], lat2d[:, 1],
                         c=np.clip(errors, 0, threshold * 2),
                         cmap='RdGy_r', alpha=0.6, s=12, lw=0,
                         vmin=0, vmax=threshold * 2)
    cb = plt.colorbar(sc, ax=axes[1])
    cb.set_label('Reconstruction Error', color=ABB['near_black'])
    cb.ax.yaxis.set_tick_params(color=ABB['dark_gray'])
    axes[1].set_title('Latent Space — Reconstruction Error',
                      color=ABB['near_black'], pad=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('Figure 7: CVAE Latent Space Visualisation  (t-SNE projection)\n'
                 'Each point represents one historical order',
                 fontsize=12, fontweight='bold', color=ABB['near_black'], y=1.02)

    add_abb_branding(fig, 'Fig 07 ��� Latent space structure')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig07_tsne.png"); plt.close()
    print("  ✓ Saved fig07_tsne.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Predicted vs Actual (top 6 numeric columns)
# ══════════════════════════════════════════════════════════════════════════════

def fig08_predicted_vs_actual(checker, C, X, tgt_cols):
    print("  Computing Figure 08 (predicted vs actual)...")

    checker.model.eval()
    X_recon_list = []
    with torch.no_grad():
        for i in range(len(C)):
            xr, *_ = checker.model(X[i:i+1], C[i:i+1])
            X_recon_list.append(xr.squeeze(0).numpy())

    X_true = X.numpy()
    X_pred = np.array(X_recon_list)

    # Pick 6 best-performing numeric columns (lowest RMSE)
    rmses     = np.sqrt(((X_true - X_pred)**2).mean(axis=0))
    label_enc = set(checker.label_encoders.keys())
    numeric   = [i for i, c in enumerate(tgt_cols) if c not in label_enc]
    top6_idx  = sorted(numeric, key=lambda i: rmses[i])[:6]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes      = axes.flatten()

    for ax, idx in zip(axes, top6_idx):
        col  = tgt_cols[idx]
        y_t  = X_true[:, idx]
        y_p  = X_pred[:, idx]
        rmse = rmses[idx]

        # Scatter
        ax.scatter(y_t, y_p, color=ABB['dark_gray'],
                   alpha=0.35, s=10, lw=0, zorder=3)

        # Perfect prediction line
        lo = min(y_t.min(), y_p.min()) - 0.02
        hi = max(y_t.max(), y_p.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color=ABB['red'],
                lw=1.5, ls='--', zorder=4, label='Perfect fit')

        ax.set_title(col, fontweight='bold',
                     color=ABB['near_black'], pad=10)
        ax.set_xlabel('Actual (normalised)')
        ax.set_ylabel('Predicted (normalised)')

        # RMSE badge
        ax.text(0.97, 0.05, f'RMSE = {rmse:.3f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, color=ABB['red_dark'],
                bbox=dict(boxstyle='round,pad=0.3',
                          fc=ABB['white'], ec=ABB['light_gray']))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('Figure 8: Predicted vs Actual Values — Top 6 Numeric Columns\n'
                 '(normalised scale, diagonal = perfect prediction)',
                 fontsize=12, fontweight='bold',
                 color=ABB['near_black'], y=1.01)

    add_abb_branding(fig, 'Fig 08 — Reconstruction quality per column')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig08_predicted_vs_actual.png"); plt.close()
    print("  ✓ Saved fig08_predicted_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — Training Experiment Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig09_experiment_comparison():
    # Hard-coded from the 3 runs you have already completed
    experiments = [
        {'label': 'Run 1\nBaseline',
         'epochs': 200, 'latent': 32, 'beta': 1.0,
         'lr': 1e-3,  'val_loss': 0.4404,
         'train_loss': 0.3756},
        {'label': 'Run 2\nLarger model',
         'epochs': 500, 'latent': 64, 'beta': 0.5,
         'lr': 1e-3,  'val_loss': 0.2410,
         'train_loss': 0.2078},
        {'label': 'Run 3\nTuned LR',
         'epochs': 600, 'latent': 64, 'beta': 0.5,
         'lr': 5e-4,  'val_loss': 0.1928,
         'train_loss': 0.1510},
    ]

    df  = pd.DataFrame(experiments)
    x   = np.arange(len(df))
    w   = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: Train vs Val loss per run ────────────────────────
    b1 = axes[0].bar(x - w/2, df['train_loss'], w,
                     color=ABB['dark_gray'], alpha=0.9,
                     edgecolor='none', label='Train loss')
    b2 = axes[0].bar(x + w/2, df['val_loss'],   w,
                     color=ABB['red'],       alpha=0.9,
                     edgecolor='none', label='Val loss  (best)')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.005,
                         f'{h:.4f}', ha='center', va='bottom',
                         fontsize=8, color=ABB['near_black'])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['label'], fontsize=9)
    axes[0].set_ylabel('Loss (MSE + β·KL)')
    axes[0].set_title('Train vs Validation Loss\nper Experiment',
                      color=ABB['near_black'], pad=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, df['val_loss'].max() * 1.25)

    # Add improvement arrows
    for i in range(1, len(df)):
        imp = (df.loc[i-1,'val_loss'] - df.loc[i,'val_loss']) \
               / df.loc[i-1,'val_loss'] * 100
        axes[0].annotate(
            f'−{imp:.0f}%',
            xy   =(i + w/2,      df.loc[i,'val_loss']),
            xytext=(i + w/2 + 0.05, df.loc[i,'val_loss'] + 0.03),
            fontsize=8, color=ABB['red_dark'], fontweight='bold')

    # ── Panel B: Hyperparameter overview table ────────────────────
    axes[1].axis('off')
    col_labels = ['Run', 'Epochs', 'Latent\nDim', 'Beta',
                  'Init LR', 'Best Val\nLoss', 'Improvement']
    rows = []
    for i, exp in enumerate(experiments):
        if i == 0:
            imp_str = '—  (baseline)'
        else:
            prev = experiments[i-1]['val_loss']
            imp  = (prev - exp['val_loss']) / prev * 100
            imp_str = f'↓ {imp:.1f}%'
        rows.append([
            exp['label'].replace('\n', ' '),
            str(exp['epochs']),
            str(exp['latent']),
            str(exp['beta']),
            str(exp['lr']),
            f"{exp['val_loss']:.4f}",
            imp_str,
        ])

    table = axes[1].table(
        cellText   = rows,
        colLabels  = col_labels,
        cellLoc    = 'center',
        loc        = 'center',
        bbox       = [0, 0.1, 1, 0.85])

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor(ABB['near_black'])
        table[(0, j)].get_text().set_color(ABB['white'])
        table[(0, j)].get_text().set_fontweight('bold')

    # Style best run row (Run 3)
    for j in range(len(col_labels)):
        table[(3, j)].set_facecolor(ABB['light_gray'])
        table[(3, j)].get_text().set_fontweight('bold')

    axes[1].set_title('Hyperparameter Experiment Summary',
                      color=ABB['near_black'], pad=14,
                      fontweight='bold', y=1.0)

    for ax in [axes[0]]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    add_abb_branding(fig, 'Fig 09 — Hyperparameter search results')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig09_experiment_comparison.png"); plt.close()
    print("  ✓ Saved fig09_experiment_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — Condition–Target Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig10_correlation_heatmap(C, X, cond_cols, tgt_cols):
    print("  Computing Figure 10 (correlation heatmap)...")

    C_np  = C.numpy()
    X_np  = X.numpy()
    corr  = np.corrcoef(
        np.hstack([C_np, X_np]), rowvar=False
    )[:len(cond_cols), len(cond_cols):]   # shape: (cond, target)

    # Shorten labels
    def shorten(name, max_len=12):
        return name if len(name) <= max_len else name[:max_len-1] + '…'

    row_labels = [shorten(c) for c in cond_cols]
    col_labels = [shorten(c) for c in tgt_cols]

    # Use only top 15 condition rows by max abs correlation (readability)
    max_abs = np.abs(corr).max(axis=1)
    top15   = np.argsort(max_abs)[-15:][::-1]
    corr_sub     = corr[top15, :]
    row_labels_sub = [row_labels[i] for i in top15]

    fig, ax = plt.subplots(figsize=(13, 7))

    im = ax.imshow(corr_sub, cmap='RdGy', aspect='auto',
                   vmin=-1, vmax=1, interpolation='nearest')

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Pearson Correlation', color=ABB['near_black'])
    cb.ax.yaxis.set_tick_params(color=ABB['dark_gray'])

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(row_labels_sub)))
    ax.set_yticklabels(row_labels_sub, fontsize=8)

    # Annotate cells
    for i in range(len(row_labels_sub)):
        for j in range(len(col_labels)):
            val  = corr_sub[i, j]
            text_col = ABB['white'] if abs(val) > 0.55 else ABB['near_black']
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=6.5, color=text_col)

    ax.set_title('Figure 10: Condition–Target Correlation Heatmap\n'
                 '(Top 15 condition features by maximum absolute correlation)',
                 color=ABB['near_black'], pad=14, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    add_abb_branding(fig, 'Fig 10 — Feature correlation analysis')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("fig10_correlation_heatmap.png"); plt.close()
    print("  ✓ Saved fig10_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_abb_style()

    print("=" * 60)
    print("  Generating all thesis figures  (ABB style)")
    print("=" * 60 + "\n")

    print("  Loading model and data...")
    checker = FeasibilityChecker("model.pt")
    C, X, cond_cols, tgt_cols, df_raw = load_features(checker)
    threshold = checker.THRESHOLD
    print(f"  ✓ {len(C)} samples | threshold = {threshold}\n")

    print("  Pre-computing errors and latent vectors...")
    errors, latents, feas_scores = compute_errors_latents(checker, C, X)
    print(f"  ✓ Done  (errors range: {errors.min():.4f} – {errors.max():.2f})\n")

    fig01_training_curve()
    fig02_error_distribution(errors, threshold)
    fig03_feasibility_scatter(errors, feas_scores, threshold)
    fig04_per_column_rmse()
    fig05_confusion_f1(checker, C, X, threshold)
    fig06_constraint_satisfaction(checker, C, X, tgt_cols)
    fig07_tsne(errors, latents, threshold)
    fig08_predicted_vs_actual(checker, C, X, tgt_cols)
    fig09_experiment_comparison()
    fig10_correlation_heatmap(C, X, cond_cols, tgt_cols)

    print("\n" + "=" * 60)
    print("  All 10 figures saved")
    print("=" * 60)
    print("""
  File                               Thesis chapter
  ────────────────────────────────────────────────────────
  fig01_training_curve.png         → Ch 4.2  Model Training
  fig02_error_distribution.png     → Ch 5.1  Threshold Selection
  fig03_feasibility_scatter.png    → Ch 5.2  Feasibility Checker
  fig04_per_column_rmse.png        → Ch 5.3  Prediction Accuracy
  fig05_confusion_f1.png           → Ch 5.4  Classifier Metrics
  fig06_constraint_satisfaction.png→ Ch 5.5  Req 4 Validation
  fig07_tsne.png                   → Ch 4.3  Latent Space
  fig08_predicted_vs_actual.png    → Ch 5.3  Reconstruction Quality
  fig09_experiment_comparison.png  → Ch 4.2  Hyperparameter Tuning
  fig10_correlation_heatmap.png    → Ch 3.4  Feature Analysis
    """)


if __name__ == "__main__":
    main()