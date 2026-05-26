"""
run_example.py
==============
Demonstrates the feasibility checker with two real examples:
  1. A FEASIBLE configuration   (realistic values)
  2. A NON-FEASIBLE configuration (unrealistic values)

Thesis requirements covered:
  - Requirement 3: ML-based feasibility checker  →  check()
  - Requirement 4: Alternative suggestions       →  alternatives list

FeasibilityChecker.check() returns:
  {
    'feasible'             : bool   – True if recon_error < THRESHOLD
                                      AND feasibility_prob > 0.5
    'confidence'           : float  – raw sigmoid output of the
                                      feasibility head (0 to 1)
    'reconstruction_error' : float  – MSE between input and
                                      reconstruction (lower = better)
    'alternatives'         : list   – nearest historical configs
                                      (only populated when NOT feasible)
  }
"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import sys
import numpy as np
from feasibility_checker import FeasibilityChecker


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def decode_categorical(checker, col, encoded_val):
    """Convert an encoded integer back to its original label string.
    If already a string (already decoded), return as-is.
    """
    if col in checker.label_encoders:
        # Already decoded to a string — return directly
        if isinstance(encoded_val, str):
            return encoded_val
        try:
            classes = checker.label_encoders[col].classes_
            idx = int(round(float(encoded_val)))
            idx = max(0, min(idx, len(classes) - 1))
            return classes[idx]
        except (ValueError, TypeError):
            return str(encoded_val)
    return encoded_val


def denormalize_numeric(checker, col, norm_val):
    """Convert a value back to original scale.
    If already denormalised (large number), return as-is.
    """
    # Already denormalised by _find_alternatives — return directly
    if isinstance(norm_val, (int, float)):
        # Heuristic: if value is clearly outside [0,1], already real scale
        if abs(float(norm_val)) > 1.5:
            return round(float(norm_val), 2)
    try:
        idx   = checker.target_cols.index(col)
        dummy = np.zeros((1, len(checker.target_cols)), dtype=np.float32)
        dummy[0, idx] = float(norm_val)
        restored = checker.target_scaler.inverse_transform(dummy)
        return round(float(restored[0, idx]), 2)
    except Exception:
        return round(float(norm_val), 4)


def format_predicted(checker, col, val):
    """Human-readable string for any predicted column value."""
    if col in checker.label_encoders:
        return decode_categorical(checker, col, val)
    return denormalize_numeric(checker, col, val)


def error_interpretation(error: float, threshold: float) -> str:
    """Plain-English interpretation of reconstruction error."""
    ratio = error / threshold
    if ratio < 0.5:
        return "Very low error — strongly matches historical patterns"
    elif ratio < 1.0:
        return "Low error — consistent with historical designs"
    elif ratio < 5.0:
        return "Moderate error — unusual but not extreme"
    elif ratio < 50.0:
        return "High error — far from any historical design"
    else:
        return "Extreme error — physically impossible values detected"


def print_result(checker, result: dict, proposed: dict):
    """Pretty-print one feasibility check result."""

    feasible   = result["feasible"]
    conf       = result["confidence"]
    recon_err  = result["reconstruction_error"]
    alts       = result.get("alternatives", [])
    threshold  = checker.THRESHOLD

    # ── Verdict banner ─────────────────────────────────────────────
    verdict    = "✓  FEASIBLE" if feasible else "✗  NOT FEASIBLE"
    interp     = error_interpretation(recon_err, threshold)

    print(f"  Verdict              : {verdict}")
    print(f"  Reconstruction error : {recon_err:.4f}  "
          f"(threshold = {threshold})")
    print(f"  Feasibility head     : {conf:.6f}  "
          f"({'> 0.5 ✓' if conf > 0.5 else '≤ 0.5 ✗'})")
    print(f"  Interpretation       : {interp}")
    print(f"  Alternatives found   : {len(alts)}")

    # ── Proposed vs best alternative (closest historical match) ────
    if alts:
        best      = alts[0]
        best_oid  = best.get("OrderID", "?")
        best_dist = best.get("distance", "?")
        cfg       = {k: v for k, v in best.items()
                     if k not in ("distance", "dist", "OrderID")}

        dist_str = f"{best_dist:.4f}" if isinstance(best_dist, float) else str(best_dist)

        print(f"\n  Proposed vs Closest Historical Match  "
              f"(OrderID={best_oid}, latent dist={dist_str}):")
        print(f"\n  {'Column':<26} {'Proposed':>18}  {'Historical':>18}")
        print("  " + "-" * 66)

        for col in proposed:
            prop_val = proposed[col]
            hist_raw = cfg.get(col, "—")

            hist_str = (str(format_predicted(checker, col, hist_raw))
                        if hist_raw != "—" else "—")
            prop_str = (f"{prop_val:.2f}"
                        if isinstance(prop_val, float) else str(prop_val))

            # Flag large differences for numeric columns
            flag = ""
            if (hist_raw != "—"
                    and col not in checker.label_encoders
                    and col not in ("LongJournals", "HighTempCon")):
                try:
                    hist_num = float(hist_raw)
                    prop_num = float(prop_val)
                    max_val  = max(abs(prop_num), abs(hist_num), 1e-6)
                    if abs(prop_num - hist_num) / max_val > 0.3:
                        flag = "  ⚠"
                except (TypeError, ValueError):
                    pass

            print(f"  {col:<26} {prop_str:>18}  {hist_str:>18}{flag}")

    return alts


def print_alternatives(checker, alts: list, title: str = "Alternatives"):
    """Print Requirement 4 — alternative configurations."""
    print(f"\n  {title}:")

    if not alts:
        print("  (none — only generated for non-feasible configurations)")
        return

    for i, alt in enumerate(alts, 1):
        dist = alt.get("distance", alt.get("dist", "?"))
        oid  = alt.get("OrderID",  "?")
        cfg  = {k: v for k, v in alt.items()
                if k not in ("distance", "dist", "OrderID")}

        dist_str = f"{dist:.4f}" if isinstance(dist, float) else str(dist)

        print(f"\n  ── Alternative {i}  "
              f"(historical OrderID = {oid}, "
              f"latent distance = {dist_str}) " + "─" * 8)
        print(f"  {'Column':<26} {'Value':>18}")
        print("  " + "-" * 48)

        for col, val in cfg.items():
            readable = format_predicted(checker, col, val)
            print(f"  {col:<26} {str(readable):>18}")


# ── Mill conditions (INPUT — what the engineer provides) ──────────────────────

MILL_CONDITIONS = {
    'NoOfRolls':          1,
    'NoOfSystems':        1,
    'MillType1ID':        2,
    'MillType2ID':        4,
    'MillType3ID':        1,
    'System1ID':          4,
    'System2ID':          2,
    'ClientInterfaceID':  1,
    'HMIID':              1,
    'SWReleaseID':        3,
    'MSSCommID':          0,
    'RollPositionID':     1,
    'RPM':                800,
    'StripSpeed':         20.0,
    'MaxTension':         150.0,
    'MinTension':         30.0,
    'MaxWrapAngle':       5.0,
    'MinWrapAngle':       1.0,
    'MaxStress':          300.0,
    'MinStress':          50.0,
    'MaxTW':              0.8,
    'MinTW':              0.1,
    'MaxStripTemp':       60,
    'MaxWidth':           1600.0,
    'MinWidth':           600.0,
    'MaxThickness':       3.0,
    'MinThickness':       0.2,
    'MaxEqLoad':          500,
    'MinEqLoad':          100,
    'BendingSystem':      1,
    'CoolingSystem':      1,
    'FlatnessLogger':     1,
    'MSS':                0,
    'MST':                1,
    'TSS':                0,
    'MTG':                0,
    'DoubleFoil':         0,
}

# ── Example 1: Realistic → expected FEASIBLE ──────────────────────────────────

PROPOSED_ROLL_FEASIBLE = {
    'TotalNoOfZones':        32,
    'NoOf52mmZones':         16,
    'NoOf26mmZones':         16,
    'BearingCentreDistance': 450,
    'LongJournalLength':     200,
    'RaSTR':                 0.4,
    'RaStrip':               0.3,
    'LongJournals':          0,
    'HighTempCon':           0,
    'RollTypeID':            1,
    'RollDiameterID':        3,
    'BearingTypeID':         2,
    'BearingHouseID':        1,
    'BearingDiameterID':     3,
    'SealingTypeID':         1,
    'ShaftMaterialID':       1,
    'SignalTransUnitID':     5,
    'RollSurfaceID':         1,
    'RollRepairID':          1,
    'StripCooling':          'Wateremulsion',
    'GearBox':               'No',
}

# ── Example 2: Unrealistic → expected NOT FEASIBLE ────────────────────────────

PROPOSED_ROLL_INFEASIBLE = {
    'TotalNoOfZones':        99,
    'NoOf52mmZones':         50,
    'NoOf26mmZones':         50,
    'BearingCentreDistance': 99,
    'LongJournalLength':     99,
    'RaSTR':                 99.0,
    'RaStrip':               99.0,
    'LongJournals':          1,
    'HighTempCon':           1,
    'RollTypeID':            6,
    'RollDiameterID':        5,
    'BearingTypeID':         10,
    'BearingHouseID':        6,
    'BearingDiameterID':     7,
    'SealingTypeID':         3,
    'ShaftMaterialID':       3,
    'SignalTransUnitID':     45,
    'RollSurfaceID':         12,
    'RollRepairID':          2,
    'StripCooling':          'Kerosene',
    'GearBox':               'Yes',
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():

    # Load model
    print_section("Loading Trained CVAE Model")
    try:
        checker = FeasibilityChecker("model.pt")
        print(f"  ✓ model.pt loaded successfully")
        print(f"  Condition columns : {len(checker.condition_cols)}")
        print(f"  Target columns    : {len(checker.target_cols)}")
        print(f"  THRESHOLD         : {checker.THRESHOLD}")
        print()
        print("  NOTE: The checker uses TWO criteria for feasibility:")
        print("    1. reconstruction_error < THRESHOLD")
        print("    2. feasibility_head output > 0.5")
        print("  Both must pass for a config to be marked FEASIBLE.")
        print()
        print("  If Example 1 shows NOT FEASIBLE with a low error,")
        print("  the threshold needs to be raised. See comment at top.")
    except FileNotFoundError:
        print("  ✗ model.pt not found — run train.py first")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        sys.exit(1)

    # ── Example 1 ─────────────────────────────────────────────────
    print_section("Example 1 — Realistic Configuration  (Expected: FEASIBLE)")
    print("\n  Mill  : 4-High, 1600mm width, standard process")
    print("  Roll  : 32 zones, Wateremulsion cooling, standard bearings\n")

    try:
        result1 = checker.check(MILL_CONDITIONS, PROPOSED_ROLL_FEASIBLE)
        alts1   = print_result(checker, result1, PROPOSED_ROLL_FEASIBLE)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Example 2 ─────────────────────────────────────────────────
    print_section("Example 2 — Unrealistic Configuration  (Expected: NOT FEASIBLE)")
    print("\n  Mill  : same mill")
    print("  Roll  : 999 zones, extreme dimensions — clearly wrong\n")

    try:
        result2 = checker.check(MILL_CONDITIONS, PROPOSED_ROLL_INFEASIBLE)
        alts2   = print_result(checker, result2, PROPOSED_ROLL_INFEASIBLE)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Requirement 4: Alternatives ───────────────────────────────
    print_section("Requirement 4 — Alternative Configurations")
    print("\n  Alternatives are the closest historical roll designs")
    print("  retrieved from the training data via nearest-neighbour")
    print("  search in the CVAE latent space.\n")

    print_alternatives(checker, alts2,
                       "For the NON-FEASIBLE config (Requirement 4)")
    print_alternatives(checker, alts1,
                       "For the FEASIBLE config (no alternatives — as expected)")

    # ── Summary ───────────────────────────────────────────────────
    print_section("Summary — Thesis Requirements 3 & 4")

    e1  = result1["reconstruction_error"]
    e2  = result2["reconstruction_error"]
    f1  = result1["feasible"]
    f2  = result2["feasible"]
    c1  = result1["confidence"]
    c2  = result2["confidence"]
    thr = checker.THRESHOLD

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  Requirement 3: ML-based Feasibility Checker                │
  └─────────────────────────────────────────────────────────────┘

  Architecture : Conditional Variational Autoencoder (CVAE)
  Input        : 37 mill condition features  (condition vector c)
  Output       : 21 roll configuration features reconstructed
  Feasibility  : dual criterion —
                   recon_error < {thr}  AND  sigmoid_head > 0.5

  {'Example':<14} {'Recon Error':>13} {'Threshold':>11} {'Head':>8}  {'Result'}
  {'─'*62}
  {'Realistic':<14} {e1:>13.4f} {thr:>11}  {c1:>7.4f}  {'FEASIBLE ✓' if f1 else 'NOT FEASIBLE ✗ ← check threshold'}
  {'Unrealistic':<14} {e2:>13.4f} {thr:>11}  {c2:>7.4f}  {'FEASIBLE ✓' if f2 else 'NOT FEASIBLE ✗'}

  ┌─────────────────────────────────────────────────────────────┐
  │  Requirement 4: Alternative Suggestions                     │
  └─────────────────────────────────────────────────────────────┘

  When a configuration is non-feasible, the system encodes it
  into the latent space and retrieves the {len(alts2)} nearest historical
  order(s) as concrete, proven alternatives the engineer can use.

  Alternatives for non-feasible config : {len(alts2)}
  Alternatives for feasible config     : {len(alts1)} (none expected)
    """)

    # ── Threshold advice ──────────────────────────────────────────
    if not f1 and e1 < thr * 5:
        print_section("⚠  Action Required — Threshold Too Strict")
        print(f"""
  Example 1 (realistic config) was rejected despite a low
  reconstruction error of {e1:.4f}.

  The current THRESHOLD = {thr} is too strict for this model.

  Recommended fix in feasibility_checker.py:
    self.THRESHOLD = {max(e1 * 2, 0.3):.2f}   # ~2× the realistic error

  After changing the threshold, re-run this script.
        """)


if __name__ == "__main__":
    main()