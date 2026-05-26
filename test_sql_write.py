import pyodbc
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# SQL Server connection configuration
SQL_CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    f"Server={os.getenv('SQL_SERVER')};"
    f"Database={os.getenv('SQL_DATABASE')};"
    f"Uid={os.getenv('SQL_USERNAME')};"
    f"Pwd={{{os.getenv('SQL_PASSWORD')}}};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;" 
)


def main():
    """Test writing to SQL Server database"""
    print(f"Attempting to connect to SQL Server database: {os.getenv('SQL_DATABASE')}")
    print("-" * 80)

    try:
        conn = pyodbc.connect(SQL_CONN_STR, timeout=10)
        cursor = conn.cursor()
        
        print("✓ Connection successful!\n")
        
        # Check if test table exists
        print("Checking if test table exists...")
        print("-" * 80)
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'TestTable'
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("✓ Test table found!\n")
        else:
            print("✗ Test table not found!")
            print("Please run 'create_test_table.sql' first with an elevated account.\n")
            conn.close()
            return
        
        # Insert test data
        print("Inserting test data...")
        print("-" * 80)
        
        test_data = [
            ('First Test', 100, datetime.now(), 'This is the first test record'),
            ('Second Test', 200, datetime.now(), 'This is the second test record'),
            ('Third Test', 300, datetime.now(), 'This is the third test record'),
        ]
        
        for name, value, date, desc in test_data:
            cursor.execute("""
                INSERT INTO dbo.TestTable (TestName, TestValue, TestDate, Description)
                VALUES (?, ?, ?, ?)
            """, name, value, date, desc)
            print(f"  ✓ Inserted: {name} - Value: {value}")
        
        conn.commit()
        print(f"\n✓ Successfully inserted {len(test_data)} rows!\n")
        
        # Read back the data to verify
        print("Verifying data...")
        print("-" * 80)
        cursor.execute("SELECT * FROM dbo.TestTable ORDER BY ID")
        
        rows = cursor.fetchall()
        print(f"\nTotal rows in TestTable: {len(rows)}\n")
        
        # Print column headers
        columns = [column[0] for column in cursor.description]
        header = " | ".join(f"{col:15}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # Print data rows
        for row in rows:
            row_data = " | ".join(f"{str(val)[:15]:15}" for val in row)
            print(row_data)
        
        print("\n" + "=" * 80)
        print("✓ WRITE TEST SUCCESSFUL!")
        print("=" * 80)
        
        # Clean up - ask user if they want to keep the test table
        print("\nNote: Test table 'dbo.TestTable' has been created.")
        print("You may want to drop it manually when done testing:")
        print("  DROP TABLE dbo.TestTable")
        
        conn.close()
        print("\n✓ Connection closed")
        
    except pyodbc.Error as e:
        print(f"✗ Database error occurred: {e}")
        if hasattr(e, 'args') and len(e.args) > 1:
            print(f"  SQL State: {e.args[0]}")
            print(f"  Error Message: {e.args[1]}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
