"""
Generates ALTER TABLE SQL statements to make all NOT NULL columns in the Order table
nullable, and drops any DEFAULT constraints that could violate FK rules.

Run this script, then execute the output SQL in SSMS or Azure Data Studio.
"""

import pyodbc
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

TABLE = "Order"

def main():
    conn = pyodbc.connect(SQL_CONN_STR, timeout=10)
    cursor = conn.cursor()
    print(f"Connected. Analysing [{TABLE}] table...\n")

    # Get all columns with their type info and nullability
    cursor.execute("""
        SELECT
            c.COLUMN_NAME,
            c.DATA_TYPE,
            c.CHARACTER_MAXIMUM_LENGTH,
            c.NUMERIC_PRECISION,
            c.NUMERIC_SCALE,
            c.IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_NAME = ?
        ORDER BY c.ORDINAL_POSITION
    """, TABLE)
    columns = cursor.fetchall()

    # Get DEFAULT constraint names for the table
    cursor.execute("""
        SELECT
            col.name AS column_name,
            dc.name AS constraint_name,
            dc.definition AS default_value
        FROM sys.default_constraints dc
        JOIN sys.columns col ON dc.parent_object_id = col.object_id
                             AND dc.parent_column_id = col.column_id
        JOIN sys.tables t ON dc.parent_object_id = t.object_id
        WHERE t.name = ?
    """, TABLE)
    defaults = {row.column_name: (row.constraint_name, row.default_value) for row in cursor.fetchall()}

    conn.close()

    not_null_cols = [c for c in columns if c.IS_NULLABLE == 'NO']

    if not not_null_cols:
        print("All columns already allow NULL. No changes needed.")
        return

    print(f"Found {len(not_null_cols)} NOT NULL column(s) (excluding identity/PK — handle manually if needed).\n")
    print("=" * 70)
    print("-- Run the following SQL in SSMS / Azure Data Studio:")
    print("=" * 70)
    print()

    for col in not_null_cols:
        col_name = col.COLUMN_NAME
        data_type = col.DATA_TYPE.upper()

        # Build the full type string
        if data_type in ('NVARCHAR', 'VARCHAR', 'CHAR', 'NCHAR'):
            max_len = col.CHARACTER_MAXIMUM_LENGTH
            length = 'MAX' if max_len == -1 else str(max_len) if max_len else '255'
            type_str = f"{data_type}({length})"
        elif data_type in ('DECIMAL', 'NUMERIC'):
            type_str = f"{data_type}({col.NUMERIC_PRECISION},{col.NUMERIC_SCALE})"
        else:
            type_str = data_type  # INT, BIT, FLOAT, DATE, etc.

        # Drop DEFAULT constraint first if one exists
        if col_name in defaults:
            constraint_name, default_val = defaults[col_name]
            print(f"-- Drop DEFAULT {default_val} on [{col_name}]")
            print(f"ALTER TABLE [{TABLE}] DROP CONSTRAINT [{constraint_name}];")
            print()

        print(f"ALTER TABLE [{TABLE}] ALTER COLUMN [{col_name}] {type_str} NULL;")
        print()

    print("=" * 70)
    print("-- Done. All listed columns will accept NULL after running the above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
