-- SQL Script to create test table
-- Run this with an account that has CREATE TABLE permissions (e.g., db_ddladmin or db_owner)

-- Drop table if it exists
/* IF OBJECT_ID('dbo.TestTable', 'U') IS NOT NULL 
    DROP TABLE dbo.TestTable;

-- Create test table
CREATE TABLE dbo.TestTable (
    ID INT PRIMARY KEY IDENTITY(1,1),
    TestName NVARCHAR(100) NOT NULL,
    TestValue INT,
    TestDate DATETIME,
    Description NVARCHAR(255)
);

-- Verify table was created
SELECT 'Table created successfully' AS Status;

-- Show table structure
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    CHARACTER_MAXIMUM_LENGTH
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'TestTable'
ORDER BY ORDINAL_POSITION; */
