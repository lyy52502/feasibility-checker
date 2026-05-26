import pyodbc
import pandas as pd
from datetime import datetime
from collections import defaultdict
import json

class ComprehensiveDatabaseComparer:
    def __init__(self, access_db_path, sql_conn_str):
        """
        Comprehensive database comparison tool to detect data loss and differences
        
        :param access_db_path: Path to .accdb or .mdb file
        :param sql_conn_str: SQL Server connection string
        """
        self.access_db_path = access_db_path
        self.sql_conn_str = sql_conn_str
        self.access_conn = None
        self.sql_conn = None
        self.comparison_results = {
            'tables': {},
            'schemas': {},
            'data': {},
            'relationships': {},
            'primary_keys': {},
            'indexes': {},
            'data_loss_warnings': []
        }
        
    def connect_databases(self):
        """Connect to both Access and SQL Server databases"""
        try:
            access_conn_str = (
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};"
                f"DBQ={self.access_db_path};"
            )
            self.access_conn = pyodbc.connect(access_conn_str)
            print(f"? Connected to Access database: {self.access_db_path}")
        except Exception as e:
            print(f"? Failed to connect to Access database: {e}")
            return False
            
        try:
            self.sql_conn = pyodbc.connect(self.sql_conn_str, timeout=10)
            print("? Connected to SQL Server database")
        except Exception as e:
            print(f"? Failed to connect to SQL Server: {e}")
            return False
            
        return True
    
    def get_tables(self, connection, db_type):
        """Get list of tables from database"""
        cursor = connection.cursor()
        tables = []
        
        if db_type == "access":
            for table_info in cursor.tables(tableType='TABLE'):
                table_name = table_info.table_name
                if not table_name.startswith('MSys'):  # Skip system tables
                    tables.append(table_name)
        else:  # SQL Server
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)
            tables = [row[0] for row in cursor.fetchall()]
        
        return sorted(tables)
    
    def get_table_schema(self, connection, table_name, db_type):
        """Get detailed schema for a table"""
        cursor = connection.cursor()
        schema = []
        
        try:
            columns = cursor.columns(table=table_name)
            for col in columns:
                schema.append({
                    'column_name': col.column_name,
                    'data_type': col.type_name,
                    'nullable': col.nullable,
                    'column_size': col.column_size,
                    'ordinal_position': col.ordinal_position
                })
        except UnicodeDecodeError:
            # Fallback: use SELECT query to get column info from cursor.description
            try:
                cursor.execute(f"SELECT * FROM [{table_name}] WHERE 1=0")
                for i, desc in enumerate(cursor.description):
                    schema.append({
                        'column_name': desc[0],
                        'data_type': str(desc[1]),
                        'nullable': True,
                        'column_size': desc[3] if desc[3] else 0,
                        'ordinal_position': i + 1
                    })
            except Exception:
                pass
        return sorted(schema, key=lambda x: x['ordinal_position'])
    
    def get_primary_keys(self, connection, table_name, db_type):
        """Get primary key information"""
        cursor = connection.cursor()
        pk_columns = []
        
        try:
            if db_type == "access":
                try:
                    for row in cursor.primaryKeys(table_name):
                        pk_columns.append(row.column_name)
                except UnicodeDecodeError:
                    # Fallback: skip primary keys if encoding error
                    pass
            else:  # SQL Server
                cursor.execute("""
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
                    AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, table_name)
                pk_columns = [row[0] for row in cursor.fetchall()]
        except:
            pass
            
        return pk_columns
    
    def get_foreign_keys(self, connection, db_type):
        """Get foreign key relationships"""
        cursor = connection.cursor()
        relationships = []
        
        try:
            if db_type == "access":
                # Access foreign keys through exported relationships table
                try:
                    cursor.execute("""
                        SELECT szRelationship, szObject, szColumn, szReferencedObject, szReferencedColumn
                        FROM RelDump
                    """)
                except Exception:
                    # Fallback to system table if RelDump is not available
                    cursor.execute("""
                        SELECT szRelationship, szObject, szColumn, szReferencedObject, szReferencedColumn
                        FROM MSysRelationships
                    """)

                for row in cursor.fetchall():
                    relationships.append({
                        'constraint_name': row[0],
                        'table': row[1],
                        'column': row[2],
                        'referenced_table': row[3],
                        'referenced_column': row[4]
                    })
            else:  # SQL Server
                cursor.execute("""
                    SELECT 
                        fk.name AS constraint_name,
                        tp.name AS table_name,
                        cp.name AS column_name,
                        tr.name AS referenced_table,
                        cr.name AS referenced_column
                    FROM sys.foreign_keys fk
                    INNER JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
                    INNER JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
                    INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                    INNER JOIN sys.columns cp ON fkc.parent_column_id = cp.column_id AND fkc.parent_object_id = cp.object_id
                    INNER JOIN sys.columns cr ON fkc.referenced_column_id = cr.column_id AND fkc.referenced_object_id = cr.object_id
                    ORDER BY tp.name, fk.name
                """)
                for row in cursor.fetchall():
                    relationships.append({
                        'constraint_name': row[0],
                        'table': row[1],
                        'column': row[2],
                        'referenced_table': row[3],
                        'referenced_column': row[4]
                    })
        except Exception as e:
            print(f"? Could not retrieve foreign keys for {db_type}: {e}")
            
        return relationships
    
    def get_indexes(self, connection, table_name, db_type):
        """Get index information for a table"""
        cursor = connection.cursor()
        indexes = []
        
        try:
            if db_type == "access":
                try:
                    cursor.execute(f"SELECT * FROM [{table_name}]")
                    # Access indexes through statistics
                    for row in cursor.statistics(table_name):
                        if row.index_name:
                            indexes.append({
                                'index_name': row.index_name,
                                'column_name': row.column_name,
                                'non_unique': row.non_unique
                            })
                except UnicodeDecodeError:
                    # Fallback: skip indexes if encoding error
                    pass
            else:  # SQL Server
                cursor.execute("""
                    SELECT 
                        i.name AS index_name,
                        c.name AS column_name,
                        i.is_unique
                    FROM sys.indexes i
                    INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                    INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                    WHERE i.object_id = OBJECT_ID(?)
                    AND i.is_primary_key = 0
                    ORDER BY i.name, ic.key_ordinal
                """, table_name)
                for row in cursor.fetchall():
                    indexes.append({
                        'index_name': row[0],
                        'column_name': row[1],
                        'non_unique': not row[2]
                    })
        except Exception as e:
            print(f"? Could not retrieve indexes for {table_name}: {e}")
            
        return indexes
    
    def compare_tables(self):
        """Compare table lists"""
        print("\n" + "="*80)
        print("TABLE COMPARISON")
        print("="*80)
        
        access_tables = self.get_tables(self.access_conn, "access")
        sql_tables = self.get_tables(self.sql_conn, "sql")
        
        print(f"\nAccess DB Tables ({len(access_tables)}): {', '.join(access_tables) if access_tables else 'None'}")
        print(f"SQL Server Tables ({len(sql_tables)}): {', '.join(sql_tables) if sql_tables else 'None'}")
        
        access_only = set(access_tables) - set(sql_tables)
        sql_only = set(sql_tables) - set(access_tables)
        common_tables = set(access_tables) & set(sql_tables)
        
        if access_only:
            print(f"\n? POTENTIAL DATA LOSS - Tables only in Access ({len(access_only)}): {', '.join(access_only)}")
            self.comparison_results['data_loss_warnings'].append({
                'type': 'missing_tables_in_sql',
                'tables': list(access_only),
                'severity': 'HIGH'
            })
        
        if sql_only:
            print(f"\n? Tables only in SQL Server ({len(sql_only)}): {', '.join(sql_only)}")
        
        if common_tables:
            print(f"\n? Common tables ({len(common_tables)}): {', '.join(common_tables)}")
        
        self.comparison_results['tables'] = {
            'access': access_tables,
            'sql': sql_tables,
            'access_only': list(access_only),
            'sql_only': list(sql_only),
            'common': list(common_tables)
        }
        
        return common_tables
    
    def compare_schemas(self, common_tables):
        """Compare schemas for common tables"""
        print("\n" + "="*80)
        print("SCHEMA COMPARISON")
        print("="*80)
        
        for table in common_tables:
            print(f"\n?? Table: {table}")
            
            access_schema = self.get_table_schema(self.access_conn, table, "access")
            sql_schema = self.get_table_schema(self.sql_conn, table, "sql")
            
            access_cols = {col['column_name']: col for col in access_schema}
            sql_cols = {col['column_name']: col for col in sql_schema}
            
            access_col_names = set(access_cols.keys())
            sql_col_names = set(sql_cols.keys())
            
            missing_in_sql = access_col_names - sql_col_names
            missing_in_access = sql_col_names - access_col_names
            common_cols = access_col_names & sql_col_names
            
            if missing_in_sql:
                print(f"  ? POTENTIAL DATA LOSS - Columns in Access but missing in SQL ({len(missing_in_sql)}): {', '.join(missing_in_sql)}")
                self.comparison_results['data_loss_warnings'].append({
                    'type': 'missing_columns_in_sql',
                    'table': table,
                    'columns': list(missing_in_sql),
                    'severity': 'HIGH'
                })
            
            if missing_in_access:
                print(f"  ? Columns in SQL but missing in Access ({len(missing_in_access)}): {', '.join(missing_in_access)}")
            
            # Check data type differences
            type_mismatches = []
            for col_name in common_cols:
                access_type = access_cols[col_name]['data_type']
                sql_type = sql_cols[col_name]['data_type']
                if access_type != sql_type:
                    type_mismatches.append({
                        'column': col_name,
                        'access_type': access_type,
                        'sql_type': sql_type
                    })
            
            if type_mismatches:
                print(f"  ? Data type mismatches ({len(type_mismatches)}):")
                for mismatch in type_mismatches:
                    print(f"    - {mismatch['column']}: Access({mismatch['access_type']}) ? SQL({mismatch['sql_type']})")
            
            if not missing_in_sql and not missing_in_access and not type_mismatches:
                print(f"  ? Schemas match perfectly")
            
            self.comparison_results['schemas'][table] = {
                'access_columns': list(access_col_names),
                'sql_columns': list(sql_col_names),
                'missing_in_sql': list(missing_in_sql),
                'missing_in_access': list(missing_in_access),
                'type_mismatches': type_mismatches
            }
    
    def compare_data(self, common_tables):
        """Compare data between tables to detect data loss"""
        print("\n" + "="*80)
        print("DATA COMPARISON - CHECKING FOR DATA LOSS")
        print("="*80)
        
        for table in common_tables:
            print(f"\n?? Table: {table}")
            
            try:
                # Get row counts
                access_cursor = self.access_conn.cursor()
                access_cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                access_count = access_cursor.fetchone()[0]
                
                sql_cursor = self.sql_conn.cursor()
                sql_cursor.execute(f"SELECT COUNT(*) FROM [dbo].[{table}]")
                sql_count = sql_cursor.fetchone()[0]
                
                print(f"  Access rows: {access_count:,}")
                print(f"  SQL rows: {sql_count:,}")
                
                data_info = {
                    'access_rows': access_count,
                    'sql_rows': sql_count,
                    'difference': abs(access_count - sql_count),
                    'data_loss': False
                }
                
                if access_count != sql_count:
                    diff = access_count - sql_count
                    if diff > 0:
                        print(f"  ? POTENTIAL DATA LOSS - {diff:,} rows missing in SQL Server!")
                        self.comparison_results['data_loss_warnings'].append({
                            'type': 'missing_rows',
                            'table': table,
                            'missing_count': diff,
                            'severity': 'CRITICAL'
                        })
                        data_info['data_loss'] = True
                    else:
                        print(f"  ? SQL Server has {abs(diff):,} more rows than Access")
                else:
                    print(f"  ? Row counts match")
                
                # Get sample data to verify
                access_cursor.execute(f"SELECT TOP 5 * FROM [{table}]")
                access_sample = access_cursor.fetchall()
                access_col_names = [col[0] for col in access_cursor.description]
                
                sql_cursor.execute(f"SELECT TOP 5 * FROM [dbo].[{table}]")
                sql_sample = sql_cursor.fetchall()
                sql_col_names = [col[0] for col in sql_cursor.description]
                
                data_info['access_columns'] = access_col_names
                data_info['sql_columns'] = sql_col_names
                data_info['sample_access_rows'] = len(access_sample)
                data_info['sample_sql_rows'] = len(sql_sample)
                
                self.comparison_results['data'][table] = data_info
                
            except Exception as e:
                print(f"  ? Error comparing data: {e}")
                self.comparison_results['data'][table] = {'error': str(e)}
    
    def compare_primary_keys(self, common_tables):
        """Compare primary keys"""
        print("\n" + "="*80)
        print("PRIMARY KEY COMPARISON")
        print("="*80)
        
        for table in common_tables:
            print(f"\n?? Table: {table}")
            
            access_pks = self.get_primary_keys(self.access_conn, table, "access")
            sql_pks = self.get_primary_keys(self.sql_conn, table, "sql")
            
            if access_pks:
                print(f"  Access PK: {', '.join(access_pks)}")
            else:
                print(f"  Access PK: None")
                
            if sql_pks:
                print(f"  SQL PK: {', '.join(sql_pks)}")
            else:
                print(f"  SQL PK: None")
            
            if set(access_pks) == set(sql_pks):
                if access_pks:
                    print(f"  ? Primary keys match")
            else:
                print(f"  ? Primary key mismatch!")
                if access_pks and not sql_pks:
                    print(f"    WARNING: Primary key exists in Access but missing in SQL")
            
            self.comparison_results['primary_keys'][table] = {
                'access': access_pks,
                'sql': sql_pks,
                'match': set(access_pks) == set(sql_pks)
            }
    
    def compare_relationships(self):
        """Compare foreign key relationships"""
        print("\n" + "="*80)
        print("RELATIONSHIP COMPARISON")
        print("="*80)
        
        access_fks = self.get_foreign_keys(self.access_conn, "access")
        sql_fks = self.get_foreign_keys(self.sql_conn, "sql")
        
        print(f"\nAccess relationships: {len(access_fks)}")
        for fk in access_fks:
            print(f"  {fk['table']}.{fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
        
        print(f"\nSQL Server relationships: {len(sql_fks)}")
        for fk in sql_fks:
            print(f"  {fk['table']}.{fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
        
        # Compare relationships by table.column -> ref_table.ref_column
        access_rel_set = {f"{fk['table']}.{fk['column']}->{fk['referenced_table']}.{fk['referenced_column']}" for fk in access_fks}
        sql_rel_set = {f"{fk['table']}.{fk['column']}->{fk['referenced_table']}.{fk['referenced_column']}" for fk in sql_fks}
        
        missing_in_sql = access_rel_set - sql_rel_set
        missing_in_access = sql_rel_set - access_rel_set
        
        if missing_in_sql:
            print(f"\n? Relationships in Access but missing in SQL ({len(missing_in_sql)}):")
            for rel in missing_in_sql:
                print(f"  {rel}")
        
        if missing_in_access:
            print(f"\n? Relationships in SQL but missing in Access ({len(missing_in_access)}):")
            for rel in missing_in_access:
                print(f"  {rel}")
        
        if not missing_in_sql and not missing_in_access and access_fks:
            print(f"\n? All relationships match")
        
        self.comparison_results['relationships'] = {
            'access': access_fks,
            'sql': sql_fks,
            'missing_in_sql': list(missing_in_sql),
            'missing_in_access': list(missing_in_access)
        }
    
    def compare_indexes(self, common_tables):
        """Compare indexes"""
        print("\n" + "="*80)
        print("INDEX COMPARISON")
        print("="*80)
        
        for table in common_tables:
            print(f"\n?? Table: {table}")
            
            access_indexes = self.get_indexes(self.access_conn, table, "access")
            sql_indexes = self.get_indexes(self.sql_conn, table, "sql")
            
            access_idx_set = {f"{idx['index_name']}:{idx['column_name']}" for idx in access_indexes}
            sql_idx_set = {f"{idx['index_name']}:{idx['column_name']}" for idx in sql_indexes}
            
            if access_indexes:
                print(f"  Access indexes: {len(access_idx_set)}")
            if sql_indexes:
                print(f"  SQL indexes: {len(sql_idx_set)}")
            
            if access_idx_set == sql_idx_set and access_indexes:
                print(f"  ? Indexes match")
            elif access_indexes or sql_indexes:
                missing_in_sql = access_idx_set - sql_idx_set
                if missing_in_sql:
                    print(f"  ? Indexes in Access but missing in SQL: {len(missing_in_sql)}")
            
            self.comparison_results['indexes'][table] = {
                'access': access_indexes,
                'sql': sql_indexes
            }
    
    def generate_summary_report(self):
        """Generate summary of all findings"""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY REPORT")
        print("="*80)
        
        # Data loss warnings
        if self.comparison_results['data_loss_warnings']:
            print(f"\n??? DATA LOSS WARNINGS ({len(self.comparison_results['data_loss_warnings'])}) ???")
            for warning in self.comparison_results['data_loss_warnings']:
                if warning['type'] == 'missing_tables_in_sql':
                    print(f"  [CRITICAL] Missing tables in SQL: {', '.join(warning['tables'])}")
                elif warning['type'] == 'missing_columns_in_sql':
                    print(f"  [HIGH] Table '{warning['table']}' - Missing columns in SQL: {', '.join(warning['columns'])}")
                elif warning['type'] == 'missing_rows':
                    print(f"  [CRITICAL] Table '{warning['table']}' - {warning['missing_count']:,} rows missing in SQL")
        else:
            print("\n??? NO DATA LOSS DETECTED ???")
        
        # Summary statistics
        print(f"\n?? STATISTICS:")
        print(f"  Total tables in Access: {len(self.comparison_results['tables']['access'])}")
        print(f"  Total tables in SQL: {len(self.comparison_results['tables']['sql'])}")
        print(f"  Common tables: {len(self.comparison_results['tables']['common'])}")
        
        total_access_rows = sum(
            data.get('access_rows', 0) 
            for data in self.comparison_results['data'].values() 
            if isinstance(data, dict) and 'access_rows' in data
        )
        total_sql_rows = sum(
            data.get('sql_rows', 0) 
            for data in self.comparison_results['data'].values() 
            if isinstance(data, dict) and 'sql_rows' in data
        )
        
        print(f"  Total rows in Access: {total_access_rows:,}")
        print(f"  Total rows in SQL: {total_sql_rows:,}")
        if total_access_rows > total_sql_rows:
            print(f"  ? Total row difference: {total_access_rows - total_sql_rows:,} rows missing in SQL")
        elif total_sql_rows > total_access_rows:
            print(f"  Total row difference: {total_sql_rows - total_access_rows:,} extra rows in SQL")
        
        print(f"\n?? RELATIONSHIPS:")
        print(f"  Access foreign keys: {len(self.comparison_results['relationships']['access'])}")
        print(f"  SQL foreign keys: {len(self.comparison_results['relationships']['sql'])}")
        print(f"  Missing in SQL: {len(self.comparison_results['relationships']['missing_in_sql'])}")
    
    def export_detailed_report(self, output_file="database_comparison_report.txt"):
        """Export detailed comparison report to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE DATABASE COMPARISON REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Access Database: {self.access_db_path}\n")
            f.write(f"SQL Server Database: (configured connection)\n\n")
            
            # Data Loss Warnings
            f.write("="*80 + "\n")
            f.write("DATA LOSS WARNINGS\n")
            f.write("="*80 + "\n")
            if self.comparison_results['data_loss_warnings']:
                for warning in self.comparison_results['data_loss_warnings']:
                    f.write(f"\n[{warning['severity']}] {warning['type']}\n")
                    if 'tables' in warning:
                        f.write(f"  Tables: {', '.join(warning['tables'])}\n")
                    if 'table' in warning:
                        f.write(f"  Table: {warning['table']}\n")
                    if 'columns' in warning:
                        f.write(f"  Columns: {', '.join(warning['columns'])}\n")
                    if 'missing_count' in warning:
                        f.write(f"  Missing rows: {warning['missing_count']:,}\n")
            else:
                f.write("\nNo data loss detected.\n")
            
            # Tables
            f.write("\n" + "="*80 + "\n")
            f.write("TABLE COMPARISON\n")
            f.write("="*80 + "\n")
            f.write(f"Access tables: {', '.join(self.comparison_results['tables']['access'])}\n")
            f.write(f"SQL tables: {', '.join(self.comparison_results['tables']['sql'])}\n")
            if self.comparison_results['tables']['access_only']:
                f.write(f"Only in Access: {', '.join(self.comparison_results['tables']['access_only'])}\n")
            if self.comparison_results['tables']['sql_only']:
                f.write(f"Only in SQL: {', '.join(self.comparison_results['tables']['sql_only'])}\n")
            
            # Schema details
            f.write("\n" + "="*80 + "\n")
            f.write("SCHEMA COMPARISON\n")
            f.write("="*80 + "\n")
            for table, schema_info in self.comparison_results['schemas'].items():
                f.write(f"\nTable: {table}\n")
                if schema_info['missing_in_sql']:
                    f.write(f"  Missing in SQL: {', '.join(schema_info['missing_in_sql'])}\n")
                if schema_info['missing_in_access']:
                    f.write(f"  Missing in Access: {', '.join(schema_info['missing_in_access'])}\n")
                if schema_info['type_mismatches']:
                    f.write(f"  Type mismatches:\n")
                    for mismatch in schema_info['type_mismatches']:
                        f.write(f"    {mismatch['column']}: {mismatch['access_type']} -> {mismatch['sql_type']}\n")
            
            # Data comparison
            f.write("\n" + "="*80 + "\n")
            f.write("DATA COMPARISON\n")
            f.write("="*80 + "\n")
            for table, data_info in self.comparison_results['data'].items():
                if isinstance(data_info, dict) and 'access_rows' in data_info:
                    f.write(f"\nTable: {table}\n")
                    f.write(f"  Access rows: {data_info['access_rows']:,}\n")
                    f.write(f"  SQL rows: {data_info['sql_rows']:,}\n")
                    f.write(f"  Difference: {data_info['difference']:,}\n")
                    if data_info['data_loss']:
                        f.write(f"  ? DATA LOSS DETECTED\n")
            
            # Relationships
            f.write("\n" + "="*80 + "\n")
            f.write("RELATIONSHIP COMPARISON\n")
            f.write("="*80 + "\n")
            f.write(f"\nAccess relationships ({len(self.comparison_results['relationships']['access'])}):\n")
            for fk in self.comparison_results['relationships']['access']:
                f.write(f"  {fk['table']}.{fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}\n")
            f.write(f"\nSQL relationships ({len(self.comparison_results['relationships']['sql'])}):\n")
            for fk in self.comparison_results['relationships']['sql']:
                f.write(f"  {fk['table']}.{fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}\n")
            if self.comparison_results['relationships']['missing_in_sql']:
                f.write(f"\nMissing in SQL:\n")
                for rel in self.comparison_results['relationships']['missing_in_sql']:
                    f.write(f"  {rel}\n")
            
            # Primary Keys
            f.write("\n" + "="*80 + "\n")
            f.write("PRIMARY KEY COMPARISON\n")
            f.write("="*80 + "\n")
            for table, pk_info in self.comparison_results['primary_keys'].items():
                f.write(f"\nTable: {table}\n")
                f.write(f"  Access: {', '.join(pk_info['access']) if pk_info['access'] else 'None'}\n")
                f.write(f"  SQL: {', '.join(pk_info['sql']) if pk_info['sql'] else 'None'}\n")
                f.write(f"  Match: {pk_info['match']}\n")
        
        print(f"\n? Detailed report saved to: {output_file}")
        
        # Also save JSON version
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        print(f"? JSON report saved to: {json_file}")
    
    def run_full_comparison(self):
        """Run complete database comparison"""
        if not self.connect_databases():
            return False
        
        try:
            common_tables = self.compare_tables()
            
            if common_tables:
                self.compare_schemas(common_tables)
                self.compare_data(common_tables)
                self.compare_primary_keys(common_tables)
                self.compare_indexes(common_tables)
            
            self.compare_relationships()
            self.generate_summary_report()
            self.export_detailed_report()
            
            return True
            
        except Exception as e:
            print(f"\n? Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.close_connections()
    
    def close_connections(self):
        """Close database connections"""
        if self.access_conn:
            try:
                self.access_conn.close()
                print("\n? Access connection closed")
            except:
                pass
        if self.sql_conn:
            try:
                self.sql_conn.close()
                print("? SQL Server connection closed")
            except:
                pass


def main():
    """Main entry point"""
    print("="*80)
    print("COMPREHENSIVE DATABASE COMPARISON TOOL")
    print("Checking for data loss and differences between Access and SQL Server")
    print("="*80 + "\n")
    
    # Configuration
    ACCESS_DB_PATH = r"C:\Users\SEDATAN\ABB\Digital Tools and Automation Team - General\Thesis work 2026\David Tanudin\Stressometer DB\StressFiles_Unsecure\StressDB2011 (2025-06-24)_Unsecure.accdb"
    
    SQL_CONN_STR = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=SE-S-RA00054,1433;"
        "Database=Stressometer Reference Database;"
        "Uid=StressometerAppUser;"
        "Pwd={xG#;G,vADDE;w{nT};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    
    # Create and run comparer
    comparer = ComprehensiveDatabaseComparer(ACCESS_DB_PATH, SQL_CONN_STR)
    success = comparer.run_full_comparison()
    
    if success:
        print("\n? Comparison completed successfully!")
    else:
        print("\n? Comparison failed!")


if __name__ == "__main__":
    main()
