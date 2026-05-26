import pyodbc
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

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

# ---- Configure here ----
TABLE_NAME = "Order"
MAX_ROWS = 10        # set to None for all rows
# --------------------------

console = Console()


def get_table_content(table_name: str, max_rows: int | None = None):
    """Fetch and display all rows from the specified table using rich."""
    console.print(Panel(
        f"[bold cyan]Database:[/] {os.getenv('SQL_DATABASE')}  "
        f"[bold cyan]Server:[/] {os.getenv('SQL_SERVER')}  "
        f"[bold cyan]Table:[/] [yellow]{table_name}[/]",
        title="[bold white]SQL Table Viewer[/]",
        border_style="blue",
    ))

    with console.status("[bold green]Connecting...[/]"):
        conn = pyodbc.connect(SQL_CONN_STR, timeout=10)
        cursor = conn.cursor()

    # Validate table exists
    cursor.execute(
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
        table_name,
    )
    if cursor.fetchone()[0] == 0:
        console.print(f"[bold red]✗[/] Table '[yellow]{table_name}[/]' not found in the database.")
        conn.close()
        return

    # Fetch rows (optionally capped)
    with console.status("[bold green]Fetching rows...[/]"):
        top_clause = f"TOP {max_rows} " if max_rows is not None else ""
        cursor.execute(f"SELECT {top_clause}* FROM dbo.[{table_name}]")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

    cap_note = f" [dim](capped at {max_rows})[/dim]" if max_rows is not None else ""
    console.print(f"[green]✓[/] Connected  |  [bold]{len(columns)}[/] columns  |  [bold]{len(rows)}[/] rows{cap_note}\n")

    if not rows:
        console.print("[dim](no data in table)[/dim]")
    else:
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            row_styles=["", "dim"],   # alternating row shading
            highlight=True,
        )

        for col in columns:
            table.add_column(col, overflow="fold")

        for row in rows:
            table.add_row(*[str(val) if val is not None else "[dim]NULL[/dim]" for val in row])

        console.print(table)

    console.print(f"\n[bold green]✓ Done.[/] {len(rows)} row(s) retrieved from [yellow]{table_name}[/].")
    conn.close()


if __name__ == "__main__":
    get_table_content(TABLE_NAME, MAX_ROWS)
