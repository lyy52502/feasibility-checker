# Database Comparison Tool

Comprehensive tool to compare Access database (.accdb/.mdb) with SQL Server database, specifically designed to detect data loss and structural differences.

## Features

### 1. **Data Loss Detection** (Primary Focus)
- Identifies missing tables in SQL Server
- Detects missing columns in SQL Server
- Compares row counts to find missing data
- Highlights critical data loss warnings

### 2. **Schema Comparison**
- Compares table structures
- Identifies column differences
- Detects data type mismatches
- Shows column ordering differences

### 3. **Relationship Comparison**
- Compares foreign key relationships
- Identifies missing relationships in SQL Server
- Shows constraint differences

### 4. **Primary Key Comparison**
- Verifies primary keys match between databases
- Identifies missing primary keys

### 5. **Index Comparison**
- Compares indexes on tables
- Shows missing indexes

### 6. **Detailed Reporting**
- Console output with color-coded warnings
- Text report file with all findings
- JSON report for programmatic access

## Installationpython extract_features.py

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Microsoft Access Database Engine installed:
   - Download from: https://www.microsoft.com/en-us/download/details.aspx?id=54920
   - Install the 64-bit version if using 64-bit Python

## Configuration

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Edit the `.env` file and update with your credentials:

```env
# SQL Server Configuration
SQL_SERVER=your-server-name,1433
SQL_DATABASE=your-database-name
SQL_USERNAME=your-username
SQL_PASSWORD=your-password

# Access Database Configuration
ACCESS_DB_PATH=C:\path\to\your\database.accdb
```

**Important:** Never commit the `.env` file to version control as it contains sensitive credentials.

## Usage

Run the comparison:
```bash
python main.py
```

Or test individual connections:
```bash
# Test Access database connection
python test_access_connection.py

# Test SQL Server connection
python test_sql_connection.py
```

## Output

The tool generates:

1. **Console Output** - Real-time progress and findings with severity indicators:
   - ? Success/Match
   - ? Warning
   - ? Error

2. **database_comparison_report.txt** - Detailed text report with:
   - Data loss warnings (CRITICAL/HIGH priority)
   - Table comparison
   - Schema differences
   - Data comparison
   - Relationship differences
   - Primary key comparison

3. **database_comparison_report.json** - Machine-readable JSON with all findings

## Understanding the Output

### Severity Levels

- **CRITICAL**: Data loss detected (missing rows, missing tables)
- **HIGH**: Potential data loss (missing columns)
- **WARNING**: Structural differences (relationships, indexes)

### Example Output

```
??? DATA LOSS WARNINGS (3) ???
  [CRITICAL] Missing tables in SQL: Products, Orders
  [HIGH] Table 'Customers' - Missing columns in SQL: LegacyID, Notes
  [CRITICAL] Table 'Invoices' - 1,523 rows missing in SQL
```

## Common Issues

1. **Access Driver Not Found**
   - Install Microsoft Access Database Engine (see Installation)
   - Ensure 64-bit version matches your Python installation

2. **SQL Server Connection Failed**
   - Verify server name and port
   - Check credentials
   - Ensure SQL Server allows remote connections

3. **Permission Denied**
   - Ensure you have read access to Access database
   - Verify SQL Server user has sufficient permissions

## Customization

You can modify the comparison to:

- Compare only specific tables
- Add detailed row-by-row comparison
- Export differences to Excel
- Add custom validation rules

## Next Steps

After running the comparison:

1. Review data loss warnings (CRITICAL priority)
2. Verify missing tables and columns
3. Investigate row count discrepancies
4. Check relationship integrity
5. Plan migration/sync strategy

## Support

For issues or questions about the comparison results, review the detailed report files.
