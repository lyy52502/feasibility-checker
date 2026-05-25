"""
Step 4: Use the trained CVAE to:
  (a) Check if a proposed configuration is feasible
  (b) Suggest valid alternatives if it is not

Usage:
    from feasibility_checker import FeasibilityChecker
    checker = FeasibilityChecker()
    result  = checker.check(condition_dict, config_dict)
"""

import torch
import pickle
import numpy as np
import pandas as pd
from model import CVAE


class FeasibilityChecker:
    def __init__(self,
                 model_path="model.pt",
                 preprocessors_path="preprocessors.pkl",
                 features_csv="features.csv"):
        

        # ── Load model ────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = CVAE(
            condition_dim = checkpoint['condition_dim'],
            target_dim    = checkpoint['target_dim'],
            latent_dim    = checkpoint['latent_dim'],
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.condition_cols = checkpoint['condition_cols']
        self.target_cols    = checkpoint['target_cols']

        # ── Load preprocessors ────────────────────────────────────
        with open(preprocessors_path, "rb") as f:
            prep = pickle.load(f)
        self.scalers        = prep['scalers']
        self.label_encoders = prep['label_encoders']

        # ── Load training data (for alternative retrieval) ────────
        self.training_df = pd.read_csv(features_csv)
        # ── Default values from training data ─────────────────
        self.default_conditions = {}
        self.default_targets = {}

        for col in self.condition_cols:
            if col in self.training_df.columns:
                self.default_conditions[col] = (
                    self.training_df[col].median()
                )

        for col in self.target_cols:
            if col in self.training_df.columns:
                self.default_targets[col] = (
                    self.training_df[col].median()
                )

        # Pre-compute latent representations of all training configs
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
        self.training_latents = mu  # shape: (N, latent_dim)

    # ── Feasibility threshold ─────────────────────────────────────
    # Reconstruction error above this → flagged as infeasible.
    # You calibrate this in your thesis evaluation chapter.
    THRESHOLD = 0.35

    def _prepare_condition(self, condition_dict: dict) -> torch.Tensor:
        """
        Convert WPF interface input into full model condition vector.
        Missing fields are filled using historical defaults.
        """

        row = []

        for col in self.condition_cols:

            # Use interface value if provided
            if col in condition_dict:
                val = condition_dict[col]

            # Otherwise use historical default
            else:
                val = self.default_conditions.get(col, 0)

            # Scale numeric values
            if col in self.scalers:
                try:
                    val = self.scalers[col].transform([[val]])[0][0]
                except Exception:
                    val = 0.0

            row.append(float(val))

        return torch.tensor([row], dtype=torch.float32)

    def _prepare_target(self, config_dict: dict) -> torch.Tensor:

        row = []
        
        for col in self.target_cols:
            val = config_dict.get(
                    col,
                    self.default_targets.get(col, 0)
                )
            if val is None:
                val = 0.0
            if col in self.scalers:
                try:
                    val = self.scalers[col].transform([[val]])[0][0]
                except Exception:
                    val = 0.0

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



    def check(self, condition_dict: dict, config_dict: dict,
              n_alternatives: int = 3) -> dict:
        """
        Main method.

        Args:
            condition_dict: mill requirements
                e.g. {'NoOfRolls': 1, 'RPM': 1500, 'MaxTension': 50.0, ...}
            config_dict: proposed roll configuration
                e.g. {'TotalNoOfZones': 34, 'RollTypeID': 2, ...}
            n_alternatives: how many alternatives to return if infeasible

        Returns:
            {
              'feasible': True/False,
              'confidence': 0.0-1.0,
              'reconstruction_error': float,
              'alternatives': [...] or []
            }
        """
        c = self._prepare_condition(condition_dict)
        x = self._prepare_target(config_dict)
        print("\n=== MODEL INPUT DEBUG ===")

        print("\nCondition input:")
        for i, col in enumerate(self.condition_cols):
            print(col, "=", c[0][i].item())

        print("\nTarget input:")
        for i, col in enumerate(self.target_cols):
            print(col, "=", x[0][i].item())


        with torch.no_grad():
            x_recon, mu, logvar, feasibility_prob = self.model(x, c)

        recon_error = torch.nn.functional.mse_loss(
            x_recon, x, reduction='mean'
        ).item()

        is_feasible = (
            recon_error < self.THRESHOLD and
            feasibility_prob.item() > 0.5
        )

        result = {
            'feasible': bool(is_feasible),
            'confidence': float(feasibility_prob.item()),
            'reconstruction_error': round(recon_error, 4),
            'alternatives': [],
        }

        # ── If not feasible: find closest valid alternatives ──────
        if not is_feasible:
            result['alternatives'] = self._find_alternatives(
                mu, c, n_alternatives
            )

        return result

    def _find_alternatives(self, query_z, c, k: int) -> list:
        """
        Find k nearest valid configs in latent space.
        Returns human-readable (denormalised) values.
        """
        distances = torch.cdist(
            query_z,                      # shape (1, latent_dim)
            self.training_latents         # shape (N, latent_dim)
        ).squeeze(0)

        top_k_idx = distances.topk(k, largest=False).indices.tolist()

        alternatives = []
        for idx in top_k_idx:
            row = self.training_df.iloc[idx]
            alt = {}

            for col in self.target_cols:
                raw_val = row[col]

                # ── Decode categoricals back to label strings ──────
                if col in self.label_encoders:
                    try:
                        enc     = self.label_encoders[col]
                        int_idx = int(round(float(raw_val)))
                        int_idx = max(0, min(int_idx, len(enc.classes_) - 1))
                        alt[col] = enc.classes_[int_idx]
                    except Exception:
                        alt[col] = str(raw_val)

                # ── Denormalise numeric values back to real scale ──
                elif col in self.scalers:
                    try:
                        real_val = self.scalers[col].inverse_transform(
                            [[float(raw_val)]]
                        )[0][0]
                        alt[col] = round(float(real_val), 2)
                    except Exception:
                        alt[col] = round(float(raw_val), 4)

                else:
                    alt[col] = raw_val

            alt['distance'] = round(distances[idx].item(), 4)
            alt['OrderID']  = int(row['OrderID'])
            alternatives.append(alt)

        return alternatives

    def generate_configs(self, condition_dict: dict,
                         n_samples: int = 5) -> list:
        """
        Forward-generation mode:
        Given mill conditions, generate n_samples candidate configs.
        Use this to support CAD automation.
        """
        c = self._prepare_condition(condition_dict)
        configs, feasibilities = self.model.generate(c, n_samples)

        results = []
        for i in range(n_samples):
            cfg = {}
            for j, col in enumerate(self.target_cols):
                cfg[col] = round(configs[i, j].item(), 4)
            cfg['feasibility_score'] = round(feasibilities[i].item(), 4)
            results.append(cfg)

        # Sort by feasibility score descending
        return sorted(results, key=lambda x: x['feasibility_score'],
                      reverse=True)
    
    