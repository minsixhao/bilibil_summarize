import sqlite3

class Database:
    def __init__(self, db_path):
        """Initialize the database connection."""
        self.db_path = db_path

    def _connect(self):
        """Create a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def create_table(self, create_table_sql):
        """Create a table based on the SQL statement provided."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while creating the table: {e}")

    def insert(self, table, columns, values):
        """Insert a row into the specified table."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                placeholders = ', '.join(['?'] * len(values))
                sql = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
                cursor.execute(sql, values)
                conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while inserting data: {e}")

    def update(self, table, set_statement, condition, values):
        """Update rows in the specified table."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                sql = f'UPDATE {table} SET {set_statement} WHERE {condition}'
                cursor.execute(sql, values)
                conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while updating data: {e}")

    def delete(self, table, condition, values):
        """Delete rows from the specified table."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                sql = f'DELETE FROM {table} WHERE {condition}'
                cursor.execute(sql, values)
                conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while deleting data: {e}")

    def query(self, table, columns, condition=None, values=None):
        """Query the database and return the results."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                sql = f'SELECT {columns} FROM {table}'
                if condition:
                    sql += f' WHERE {condition}'
                cursor.execute(sql, values or ())
                return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"An error occurred while querying data: {e}")
            return []

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
        is_sent INTEGER DEFAULT 0
        content_md TEXT,
        summary_md TEXT,
        refine_content_md TEXT,
    );
    """
    db.create_table(create_table_sql)

    # Insert a new record
    db.insert('dynamic', 'id, content', (1, 'Sample content'))

    # Select a record
    results = db.query('dynamic', 'content', 'id = ?', (1,))
    print(f"Selected content: {results[0][0]}")

    # Update a record
    db.update('dynamic', 'content = ?', 'id = ?', ('Updated content', 1))

    # Delete a record
    db.delete('dynamic', 'id = ?', (1,))

    # Query records
    result = db.query('dynamic', 'content')
    print(result)

if __name__ == '__main__':
    DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
    db = Database(DATABASE_PATH)
    print(db)
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