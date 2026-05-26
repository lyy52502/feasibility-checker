"""
Step 3: Train the CVAE model.
Reads features.csv, trains for N epochs, saves model.pt
"""

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import CVAE, cvae_loss

# ── Config ──────────────────────────────────────────────────────────
EPOCHS       = 300
BATCH_SIZE   = 64
LEARNING_RATE = 5e-4
LATENT_DIM   = 64
BETA         = 0.5    # weight of KL loss; increase to enforce disentanglement
SEED         = 42
# ────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


def load_data():
    df = pd.read_csv("features.csv")
    with open("column_metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    condition_cols = [c for c in meta['condition_cols'] if c in df.columns]
    target_cols    = [c for c in meta['target_cols']    if c in df.columns]

    print(f"Condition columns ({len(condition_cols)}): {condition_cols}")
    print(f"Target columns    ({len(target_cols)}):    {target_cols}")

    C = torch.tensor(df[condition_cols].values, dtype=torch.float32)
    X = torch.tensor(df[target_cols].values,    dtype=torch.float32)

    return C, X, condition_cols, target_cols


def train():
    C, X, cond_cols, tgt_cols = load_data()

    dataset  = TensorDataset(C, X)
    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = CVAE(
        condition_dim = C.shape[1],
        target_dim    = X.shape[1],
        latent_dim    = LATENT_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=30,   # wait longer before reducing
    factor=0.7,    # reduce more gently (0.7x instead of 0.5x)
    min_lr=1e-5,   # never go below this — stops dead LR problem
)

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Train: {n_train} samples | Val: {n_val} samples\n")

    best_val_loss = float('inf')
    history = []
    prev_lr = LEARNING_RATE  # add this line

    for epoch in range(1, EPOCHS + 1):
        # ── Training ─────────────���────────────────────────────────
        model.train()
        train_losses = []
        for c_batch, x_batch in train_loader:
            optimizer.zero_grad()
            x_recon, mu, logvar, feasibility = model(x_batch, c_batch)
            loss, recon, kl, feas = cvae_loss(
                x_recon, x_batch, mu, logvar, feasibility, BETA
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for c_batch, x_batch in val_loader:
                x_recon, mu, logvar, feasibility = model(x_batch, c_batch)
                loss, *_ = cvae_loss(
                    x_recon, x_batch, mu, logvar, feasibility, BETA
                )
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})

        # Print every 20 epochs
        if epoch % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            lr_msg = f"  ← LR reduced to {current_lr:.6f}" if current_lr != prev_lr else ""
            print(f"Epoch {epoch:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{lr_msg}")
            prev_lr = current_lr

        # ── Save best model ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                temp_path = "model_temp.pt"
                final_path = "model.pt"

                torch.save({
                    'model_state': model.state_dict(),
                    'condition_cols': cond_cols,
                    'target_cols': tgt_cols,
                    'condition_dim': C.shape[1],
                    'target_dim': X.shape[1],
                    'latent_dim': LATENT_DIM,
                }, temp_path)

                import os
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)

            except Exception as e:
                print(f"  ⚠ Could not save model at epoch {epoch}: {e}")

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")
    print("✓ Model saved to model.pt")

    # Save training history
    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    print("✓ History saved to training_history.csv")


if __name__ == "__main__":
    train()