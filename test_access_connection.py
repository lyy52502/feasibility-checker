import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Update this path to your Access database file
ACCESS_DB_PATH = os.getenv('ACCESS_DB_PATH')


def main():
    """Test Access database connection"""
    print(f"Attempting to connect to Access database: {ACCESS_DB_PATH}")
    print("-" * 80)

    try:
        # Connection string for Access database
        access_conn_str = (
            f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};"
            f"DBQ={ACCESS_DB_PATH};"
        )
        
        conn = pyodbc.connect(access_conn_str)
        cursor = conn.cursor()
        
        print("? Connection successful!\n")
        
        # List all tables
        print("Available tables:")
        print("-" * 80)
        table_count = 0
        for table_info in cursor.tables(tableType='TABLE'):
            table_name = table_info.table_name
            # Skip system tables
            if not table_name.startswith('MSys'):
                table_count += 1
                print(f"  {table_count}. {table_name}")
        
        print(f"\nTotal tables found: {table_count}")

        # Test reading relationships
        print(f"\n{'='*80}")
        print("Access relationships (MSysRelationships)")
        print("="*80)

        try:
            cursor_rel = conn.cursor()
            cursor_rel.execute("""
                SELECT szRelationship, szObject, szColumn, szReferencedObject, szReferencedColumn
                FROM MSysRelationships
            """)
            relationships = cursor_rel.fetchall()

            print(f"Total relationships found: {len(relationships)}")
            print("-" * 80)
            for i, row in enumerate(relationships[:10], 1):
                print(
                    f"{i}. {row[1]}.{row[2]} -> {row[3]}.{row[4]} "
                    f"(rel: {row[0]})"
                )

            if len(relationships) > 10:
                print(f"... showing first 10 of {len(relationships)}")
        except pyodbc.Error as e:
            print(f"? Error reading MSysRelationships: {e}")
            print("  You may need permissions to read system tables or enable system tables in Access.")
        
        # Test reading from Order table
        print(f"\n{'='*80}")
        print(f"Sample data from table: Order")
        print("="*80)
        
        try:
            # Get column info and data
            cursor2 = conn.cursor()
            cursor2.execute(f"SELECT TOP 5 * FROM [Order]")
            
            # Print column names
            columns = [column[0] for column in cursor2.description]
            print(f"Columns: {', '.join(columns)}")
            
            # Print rows
            rows = cursor2.fetchall()
            print(f"Row count (sample): {len(rows)}")
            print("-" * 80)
            for i, row in enumerate(rows, 1):
                print(f"Row {i}: {row}")
            
            # Get total row count
            cursor3 = conn.cursor()
            cursor3.execute(f"SELECT COUNT(*) FROM [Order]")
            total_rows = cursor3.fetchone()[0]
            print(f"\nTotal rows in table: {total_rows:,}")
            
        except pyodbc.Error as e:
            print(f"? Error reading from Order table: {e}")
            print("  Table 'Order' may not exist in this database")
        
        conn.close()
        print("\n? Connection closed successfully")
        
    except FileNotFoundError:
        print(f"? Error: Access database file not found at: {ACCESS_DB_PATH}")
        print("  Please update ACCESS_DB_PATH to point to your .accdb or .mdb file")
    except pyodbc.Error as e:
        print(f"? Database Error: {e}")
        print("\nPossible issues:")
        print("  1. Microsoft Access Database Engine not installed")
        print("     Download from: https://www.microsoft.com/en-us/download/details.aspx?id=54920")
        print("  2. Path to database file is incorrect")
        print("  3. Database file is locked/in use by another application")
        print("  4. Bit version mismatch (use 64-bit driver for 64-bit Python)")
    except Exception as e:
        print(f"? Unexpected Error: {e}")


if __name__ == "__main__":
    main()
