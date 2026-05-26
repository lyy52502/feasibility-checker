import torch
import pickle
import numpy as np
import pandas as pd
from model import CVAE
import torch.nn.functional as F


class FeasibilityChecker:
    def __init__(self,
                 model_path="model.pt",
                 preprocessors_path="preprocessors.pkl",
                 features_csv="features.csv"):

        # ---------------- MODEL ----------------
        checkpoint = torch.load(model_path, map_location="cpu")

        self.model = CVAE(
            condition_dim=checkpoint["condition_dim"],
            target_dim=checkpoint["target_dim"],
            latent_dim=checkpoint["latent_dim"],
        )

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.condition_cols = checkpoint["condition_cols"]
        self.target_cols = checkpoint["target_cols"]

        # ---------------- PREPROCESSORS ----------------
        with open(preprocessors_path, "rb") as f:
            prep = pickle.load(f)

        self.scalers = prep["scalers"]
        self.label_encoders = prep["label_encoders"]

        # ---------------- TRAIN DATA ----------------
        self.training_df = pd.read_csv(features_csv)

        self.training_df = self.training_df.fillna(0)

        # defaults
        self.default_conditions = {
            col: self.training_df[col].median()
            for col in self.condition_cols
            if col in self.training_df.columns
        }

        self.default_targets = {
            col: self.training_df[col].median()
            for col in self.target_cols
            if col in self.training_df.columns
        }

        # ---------------- LATENT SPACE ----------------
        C_all = torch.tensor(
            self.training_df[self.condition_cols].values,
            dtype=torch.float32
        )

        X_all = torch.tensor(
            self.training_df[self.target_cols].values,
            dtype=torch.float32
        )

        with torch.no_grad():
            mu, _ = self.model.encode(X_all, C_all)

        self.training_latents = mu

    # =========================================================
    # SAFETY: convert numpy/torch → python types
    # =========================================================
    def to_native(self, obj):
        if isinstance(obj, dict):
            return {k: self.to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.to_native(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if hasattr(obj, "item"):  # torch scalar
            return obj.item()
        return obj

    # =========================================================
    # CONDITION PREP
    # =========================================================
    def _prepare_condition(self, condition_dict):

        row = []

        for col in self.condition_cols:

            val = condition_dict.get(col, self.default_conditions.get(col, 0))

            if val is None:
                val = 0.0

            # scale numeric
            if col in self.scalers:
                try:
                    df = pd.DataFrame([{col: val}])
                    val = self.scalers[col].transform(df)[0][0]
                except Exception:
                    val = 0.0

            row.append(float(val))

        return torch.tensor([row], dtype=torch.float32)

    # =========================================================
    # TARGET PREP
    # =========================================================
    def _prepare_target(self, config_dict):

        row = []

        for col in self.target_cols:

            val = config_dict.get(col, self.default_targets.get(col, 0))

            if val is None:
                val = 0.0

            # numeric scaling
            if col in self.scalers:
                try:
                    df = pd.DataFrame([{col: val}])
                    val = self.scalers[col].transform(df)[0][0]
                except Exception:
                    val = 0.0

            # categorical encoding
            elif col in self.label_encoders:
                try:
                    enc = self.label_encoders[col]
                    val_str = str(val)

                    if val_str not in enc.classes_:
                        val_str = enc.classes_[0]

                    val = enc.transform([val_str])[0]

                except Exception:
                    val = 0.0

            row.append(float(val))

        return torch.tensor([row], dtype=torch.float32)

    # =========================================================
    # MAIN CHECK FUNCTION
    # =========================================================
    def check(self, condition_dict, config_dict, n_alternatives=3):

        c = self._prepare_condition(condition_dict)
        x = self._prepare_target(config_dict)

        with torch.no_grad():
            x_recon, mu, logvar, feasibility_prob = self.model(x, c)

        recon_error = F.mse_loss(x_recon, x, reduction="mean").item()

        is_feasible = (
            feasibility_prob.item() > 0.5   # ignore reconstruction error for now
        )

        result = {
            "feasible": bool(is_feasible),
            "confidence": float(feasibility_prob.item()),
            "reconstruction_error": float(round(recon_error, 4)),
            "alternatives": []
        }

        if not is_feasible:
            result["alternatives"] = self._find_alternatives(mu, n_alternatives)

        return self.to_native(result)

    # =========================================================
    # ALTERNATIVES
    # =========================================================
    def _find_alternatives(self, query_z, k):

        distances = torch.cdist(query_z, self.training_latents).squeeze(0)
        top_k = distances.topk(k, largest=False).indices.tolist()

        alternatives = []

        for idx in top_k:

            row = self.training_df.iloc[idx]

            alt = {}

            for col in self.target_cols:

                raw_val = row[col]

                # categorical decode
                if col in self.label_encoders:
                    try:
                        enc = self.label_encoders[col]
                        int_idx = int(round(float(raw_val)))
                        int_idx = max(0, min(int_idx, len(enc.classes_) - 1))
                        alt[col] = str(enc.classes_[int_idx])
                    except:
                        alt[col] = str(raw_val)

                else:
                    alt[col] = float(raw_val) if isinstance(
                        raw_val, (np.floating, np.integer)
                    ) else raw_val

            alt["distance"] = float(distances[idx].item())
            alt["OrderID"] = int(row["OrderID"])

            alternatives.append(alt)

        return alternatives