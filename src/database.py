"""
Societal Challenge Theme Discovery - Dynamic Database Module
Provides persistent SQLite storage for crowdsourced problem statements,
supporting live submission, CRUD operations, and automatic seeding.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "societal_challenges.db")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "problem_statements.csv")

class ChallengeDatabase:
    """Persistent SQLite database manager for problem statements."""
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a sqlite3 connection with dict-like row access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema if not present."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS problem_statements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ps_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    organization TEXT DEFAULT 'General / Crowdsourced',
                    department TEXT DEFAULT 'N/A',
                    category TEXT DEFAULT 'Software',
                    theme TEXT DEFAULT 'General',
                    is_crowdsourced INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'Open',
                    submitted_by TEXT DEFAULT 'Anonymous Citizen',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ps_id ON problem_statements(ps_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON problem_statements(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_crowdsourced ON problem_statements(is_crowdsourced);")
            conn.commit()

    def is_empty(self) -> bool:
        """Checks if the problem statements table has any records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM problem_statements;")
            count = cursor.fetchone()[0]
            return count == 0

    def seed_from_csv(self, csv_path: str = DEFAULT_CSV_PATH, force: bool = False) -> int:
        """Seeds the database from CSV if empty or forced."""
        if not force and not self.is_empty():
            return self.get_count()

        if not os.path.exists(csv_path):
            return 0

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Standardize column mappings
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ['problem statement id', 'id', 'ps number', 'index']:
                col_map[c] = 'ps_id'
            elif cl in ['problem statement title', 'title', 'headline']:
                col_map[c] = 'title'
            elif cl in ['description', 'desc', 'text']:
                col_map[c] = 'description'
            elif cl in ['organization', 'org', 'ministry']:
                col_map[c] = 'organization'
            elif cl in ['department', 'dept']:
                col_map[c] = 'department'
            elif cl in ['category', 'modality']:
                col_map[c] = 'category'
            elif cl in ['theme', 'domain bucket', 'domain']:
                col_map[c] = 'theme'

        df = df.rename(columns=col_map)
        
        # Ensure mandatory columns
        if 'ps_id' not in df.columns:
            df['ps_id'] = [f"SIH-{i+1:04d}" for i in range(len(df))]
        if 'title' not in df.columns:
            df['title'] = df['description'].apply(lambda x: " ".join(str(x).split()[:7]) + "...")
        if 'description' not in df.columns:
            df['description'] = df['title']
        if 'organization' not in df.columns:
            df['organization'] = 'Smart India Hackathon / Govt. of India'
        if 'department' not in df.columns:
            df['department'] = df['organization']
        if 'category' not in df.columns:
            df['category'] = 'Software'
        if 'theme' not in df.columns:
            df['theme'] = 'General'

        records_inserted = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if force:
                cursor.execute("DELETE FROM problem_statements WHERE is_crowdsourced = 0;")

            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO problem_statements 
                        (ps_id, title, description, organization, department, category, theme, is_crowdsourced, status, submitted_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'Open', 'SIH Official Repository')
                    """, (
                        str(row['ps_id']),
                        str(row['title']).strip(),
                        str(row['description']).strip(),
                        str(row.get('organization', 'General / Crowdsourced')).strip(),
                        str(row.get('department', 'N/A')).strip(),
                        str(row.get('category', 'Software')).strip(),
                        str(row.get('theme', 'General')).strip()
                    ))
                    records_inserted += 1
                except Exception:
                    continue
            conn.commit()

        return records_inserted

    def get_count(self) -> int:
        """Returns total records count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM problem_statements;")
            return cursor.fetchone()[0]

    def get_all_problems(self, include_crowdsourced: bool = True) -> pd.DataFrame:
        """Retrieves problem statements formatted for the preprocessing pipeline."""
        query = "SELECT * FROM problem_statements"
        if not include_crowdsourced:
            query += " WHERE is_crowdsourced = 0"
        query += " ORDER BY id ASC;"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return pd.DataFrame()

        # Map back to pipeline canonical column names
        df = df.rename(columns={
            'ps_id': 'Problem Statement ID',
            'title': 'Problem Statement Title',
            'description': 'Description',
            'organization': 'Organization',
            'department': 'Department',
            'category': 'Category',
            'theme': 'Theme'
        })
        return df

    def add_problem(
        self,
        title: str,
        description: str,
        organization: str = "Citizen Crowdsource",
        department: str = "Public Citizen",
        category: str = "Software",
        theme: str = "To Be Discovered",
        submitted_by: str = "Anonymous Citizen",
        status: str = "Under Review"
    ) -> Dict[str, Any]:
        """Dynamically inserts a new crowdsourced problem statement into the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Generate unique dynamic ID
            cursor.execute("SELECT MAX(id) FROM problem_statements;")
            max_id = cursor.fetchone()[0] or 0
            new_ps_id = f"CS-{max_id + 1:04d}"

            cursor.execute("""
                INSERT INTO problem_statements
                (ps_id, title, description, organization, department, category, theme, is_crowdsourced, status, submitted_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (
                new_ps_id,
                title.strip(),
                description.strip(),
                organization.strip() if organization else "Citizen Crowdsource",
                department.strip() if department else "Public Citizen",
                category.strip() if category else "Software",
                theme.strip() if theme else "To Be Discovered",
                status,
                submitted_by.strip() if submitted_by else "Anonymous Citizen"
            ))
            conn.commit()
            inserted_id = cursor.lastrowid

        return {
            'id': inserted_id,
            'ps_id': new_ps_id,
            'title': title,
            'status': status,
            'is_crowdsourced': 1
        }

    def delete_problem(self, ps_id: str) -> bool:
        """Deletes a problem statement by ps_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM problem_statements WHERE ps_id = ?;", (ps_id,))
            conn.commit()
            return cursor.rowcount > 0

    def reset_database(self, csv_path: str = DEFAULT_CSV_PATH) -> int:
        """Completely clears the database and re-seeds cleanly from the seed CSV."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM problem_statements;")
            conn.commit()
        return self.seed_from_csv(csv_path=csv_path, force=True)

    def get_stats(self) -> Dict[str, Any]:
        """Returns live statistics of the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM problem_statements;")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM problem_statements WHERE is_crowdsourced = 1;")
            crowdsourced = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT organization) FROM problem_statements;")
            orgs = cursor.fetchone()[0]
            cursor.execute("SELECT category, COUNT(*) FROM problem_statements GROUP BY category;")
            cat_rows = cursor.fetchall()
            categories = {r[0]: r[1] for r in cat_rows}
            cursor.execute("SELECT COUNT(*) FROM problem_statements WHERE status = 'Under Review';")
            under_review = cursor.fetchone()[0]

        return {
            'total_statements': total,
            'crowdsourced_count': crowdsourced,
            'seeded_count': total - crowdsourced,
            'organizations_count': orgs,
            'categories': categories,
            'under_review_count': under_review
        }
