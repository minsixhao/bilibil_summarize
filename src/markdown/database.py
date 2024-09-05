import sqlite3
from typing import List, Tuple, Optional
import logging

class Database:
    def __init__(self, db_path: str):
        """Initialize the database connection."""
        self.db_path = db_path

    def _connect(self):
        """Create a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def _execute_query(self, sql: str, params: Tuple = ()) -> Optional[List[Tuple]]:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"数据库操作错误: {e}")
            return None

    def create_table(self, create_table_sql: str) -> None:
        """Create a table based on the SQL statement provided."""
        self._execute_query(create_table_sql)

    def insert(self, table: str, columns: str, values: Tuple) -> None:
        """Insert a row into the specified table."""
        placeholders = ', '.join(['?'] * len(values))
        sql = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
        self._execute_query(sql, values)

    def update(self, table: str, set_statement: str, condition: str, values: Tuple) -> None:
        """Update rows in the specified table."""
        sql = f'UPDATE {table} SET {set_statement} WHERE {condition}'
        self._execute_query(sql, values)

    def delete(self, table: str, condition: str, values: Tuple) -> None:
        """Delete rows from the specified table."""
        sql = f'DELETE FROM {table} WHERE {condition}'
        self._execute_query(sql, values)

    def query(self, table: str, columns: str, condition: Optional[str] = None, values: Optional[Tuple] = None) -> List[Tuple]:
        """Query the database and return the results."""
        sql = f'SELECT {columns} FROM {table}'
        if condition:
            sql += f' WHERE {condition}'
        result = self._execute_query(sql, values or ())
        return result or []

# Example Usage
def main():
    DATABASE_PATH = '/bilibili_summarize/db/sqlite/bilibili.db'
    db = Database(DATABASE_PATH)

    # Create a table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS dynamic (
        id TEXT PRIMARY KEY,
        audio BLOB,
        content TEXT,
        summary TEXT,
        is_sent INTEGER DEFAULT 0,
        content_md TEXT,
        summary_md TEXT,
        refine_content_md TEXT
    );
    """
    db.create_table(create_table_sql)

    # Insert a new record
    db.insert('dynamic', 'id, content', ('1', 'Sample content'))

    # Select a record
    results = db.query('dynamic', 'content', 'id = ?', ('1',))
    print(f"Selected content: {results[0][0] if results else 'Not found'}")

    # Update a record
    db.update('dynamic', 'content = ?', 'id = ?', ('Updated content', '1'))

    # Delete a record
    db.delete('dynamic', 'id = ?', ('1',))

    # Query records
    result = db.query('dynamic', 'content')
    print(result)

if __name__ == '__main__':
    main()