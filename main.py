import test_access_connection
import test_sql_connection
import database_comparer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Comment/uncomment what you want to test:
    
    # Test Access database connection
    # test_access_connection.main()
    
    # Test SQL Server database connection
    # test_sql_connection.main()
    
    # Run full database comparison
    ACCESS_DB_PATH = os.getenv('ACCESS_DB_PATH')
    SQL_CONN_STR = (
        "Driver={ODBC Driver 17 for SQL Server};"
        f"Server={os.getenv('SQL_SERVER')};"
        f"Database={os.getenv('SQL_DATABASE')};"
        f"Uid={os.getenv('SQL_USERNAME')};"
        f"Pwd={{{os.getenv('SQL_PASSWORD')}}};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    comparer = database_comparer.ComprehensiveDatabaseComparer(ACCESS_DB_PATH, SQL_CONN_STR)
    comparer.run_full_comparison()

    