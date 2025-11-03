# init_db.py
import sqlite3
from pathlib import Path

# Create database file
db_path = Path("database.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL
)
''')

# Create images table
cursor.execute('''
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    image_path TEXT,
    predicted_emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

# Insert a test user (so app doesn't crash)
cursor.execute('INSERT OR IGNORE INTO users (username) VALUES (?)', ('testuser',))

# Save and close
conn.commit()
conn.close()

print("database.db created successfully!")
print("Test user 'testuser' added.")
print("You can now run the web app.")