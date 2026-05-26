
import pyodbc
import os
from dotenv import load_dotenv

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
    """Test SQL Server database connection"""
    print(f"Attempting to connect to SQL Server database: Stressometer Reference Database")
    print("-" * 80)

    try:
        conn = pyodbc.connect(SQL_CONN_STR, timeout=10)
        cursor = conn.cursor()
        
        print("? Connection successful!\n")
        
        # List all tables
        print("Available tables:")
        print("-" * 80)
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        tables = cursor.fetchall()
        table_count = 0
        for table in tables:
            table_count += 1
            print(f"  {table_count}. {table[0]}")
        
        print(f"\nTotal tables found: {table_count}")
        
        # Test reading from Order table
        print(f"\n{'='*80}")
        print(f"Sample data from table: Order")
        print("="*80)
        
        try:
            # Get column info and data
            cursor2 = conn.cursor()
            cursor2.execute("SELECT TOP 5 * FROM [dbo].[Order]")
            
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
            cursor3.execute("SELECT COUNT(*) FROM [dbo].[Order]")
            total_rows = cursor3.fetchone()[0]
            print(f"\nTotal rows in table: {total_rows:,}")
            
        except pyodbc.Error as e:
            print(f"? Error reading from Order table: {e}")
            print("  Table 'Order' may not exist in this database")
        
        conn.close()
        print("\n? Connection closed successfully")
        
    except pyodbc.Error as e:
        print(f"? Database Error: {e}")
        print("\nPossible issues:")
        print("  1. SQL Server is not accessible")
        print("  2. Incorrect server name or port")
        print("  3. Invalid credentials")
        print("  4. Network/firewall blocking connection")
        print("  5. ODBC Driver 17 for SQL Server not installed")
    except Exception as e:
        print(f"? Unexpected Error: {e}")


if __name__ == "__main__":
    main()
