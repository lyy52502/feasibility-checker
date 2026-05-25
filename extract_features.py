"""
Step 1: Extract and join relevant tables from SQL Server.
Produces: raw_data.csv
Run this first before any ML work.
"""

import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

SQL_CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    f"Server={os.getenv('SQL_SERVER')};"
    f"Database={os.getenv('SQL_DATABASE')};"
    f"Uid={os.getenv('SQL_USERNAME')};"
    f"Pwd={{{os.getenv('SQL_PASSWORD')}}};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

# ------------------------------------------------------------------ #
# This query joins all 5 core tables by OrderID.
# LEFT JOIN means: keep the Order row even if RollData/Mechanical
# data is missing for that order (we handle NULLs in Step 2).
# ------------------------------------------------------------------ #
QUERY = """
SELECT
    -- ── Identity ──────────────────────────────────────────────────
    o.OrderID,

    -- ── Mill context (from Order) ─────────────────────────────────
    -- These describe WHAT KIND OF MILL the roll goes into.
    -- They are the "conditions" for the CVAE.
    o.NoOfRolls,
    o.NoOfSystems,
    o.MillType1ID,
    o.MillType2ID,
    o.MillType3ID,
    o.System1ID,
    o.System2ID,
    o.ClientInterfaceID,
    o.HMIID,
    o.SWReleaseID,
    o.MSSCommID,
    o.BendingSystem,
    o.CoolingSystem,
    o.FlatnessLogger,
    o.MSS,
    o.MST,
    o.TSS,
    o.MTG,

    -- ── Mechanical operating envelope (from MechanicalData) ───────
    -- These describe HOW the mill operates.
    -- Also part of the "conditions" for the CVAE.
    m.RPM,
    m.StripSpeed,
    m.MaxTension,
    m.MinTension,
    m.MaxWrapAngle,
    m.MinWrapAngle,
    m.MaxStress,
    m.MinStress,
    m.MaxTW,
    m.MinTW,
    m.MaxStripTemp,
    m.MaxWidth,
    m.MinWidth,
    m.MaxThickness,
    m.MinThickness,
    m.MaxEqLoad,
    m.MinEqLoad,
    m.DoubleFoil,
    m.RollPositionID,

    -- ── Roll configuration (from RollData) ────────────────────────
    -- These are what the model PREDICTS / GENERATES.
    -- They are the "target" for the CVAE.
    r.TotalNoOfZones,
    r.NoOf52mmZones,
    r.NoOf26mmZones,
    r.BearingCentreDistance,
    r.LongJournalLength,
    r.LongJournals,
    r.HighTempCon,
    r.RaSTR,
    r.RaStrip,
    r.RollTypeID,
    r.RollDiameterID,
    r.BearingTypeID,
    r.BearingHouseID,
    r.BearingDiameterID,
    r.SealingTypeID,
    r.ShaftMaterialID,
    r.SignalTransUnitID,
    r.RollSurfaceID,
    r.RollRepairID,
    r.StripCooling,
    r.GearBox,

    -- ── Cooling (from CoolingData) ─────────────────────────────────
    -- Only 81 rows have this - will be NaN for most orders.
    c.NumberOfZones        AS Cooling_NumberOfZones,
    c.MaxWorkrollDiameter  AS Cooling_MaxWorkrollDiameter,
    c.MaxStripThickness    AS Cooling_MaxStripThickness,
    c.MaxRPMOnWorkroll     AS Cooling_MaxRPMOnWorkroll,
    c.TotalCoolingAmount   AS Cooling_TotalCoolingAmount,
    c.PressureRegulator    AS Cooling_PressureRegulator,

    -- ── Drive system (from DriveSystemOrder) ──────────────────────
    d.DriveSystemTypeID,
    d.DriveSystemID,
    d.Voltage,
    d.Power

FROM dbo.[Order] o

-- Each order can have multiple RollData rows (one per roll position).
-- We take the FIRST one per order to keep the dataset flat.
-- For a more advanced thesis: keep all and use sequence modelling.
LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY OrderID ORDER BY RollDataID) AS rn
    FROM dbo.RollData
) r ON o.OrderID = r.OrderID AND r.rn = 1

LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY OrderID ORDER BY MechanicalDataID) AS rn
    FROM dbo.MechanicalData
) m ON o.OrderID = m.OrderID AND m.rn = 1

LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY OrderID ORDER BY CoolingDataID) AS rn
    FROM dbo.CoolingData
) c ON o.OrderID = c.OrderID AND c.rn = 1

LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY OrderID ORDER BY DriveSystemID) AS rn
    FROM dbo.DriveSystemOrder
) d ON o.OrderID = d.OrderID AND d.rn = 1

-- Only use confirmed, visible reference orders (not hidden test records)
WHERE o.HideInRefList = 0
ORDER BY o.OrderID
"""

def main():
    print("Connecting to SQL Server...")
    conn = pyodbc.connect(SQL_CONN_STR, timeout=10)
    print("✓ Connected\n")

    print("Running extraction query (this may take a few seconds)...")
    df = pd.read_sql(QUERY, conn)
    conn.close()

    print(f"✓ Extracted {len(df)} rows × {len(df.columns)} columns")
    print(f"\nColumn list:\n{list(df.columns)}\n")

    # Save raw (unprocessed) data
    df.to_csv("raw_data.csv", index=False)
    print("✓ Saved to raw_data.csv")

    # Quick summary of missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nColumns with missing values:")
    print(missing.to_string())

if __name__ == "__main__":
    main()