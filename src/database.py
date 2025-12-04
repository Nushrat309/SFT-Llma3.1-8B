"""Experiment logging and database management"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any


class ExperimentDatabase:
    """Handles logging of experiments and generated responses"""
    
    def __init__(self, db_path: str = "outputs/experiment_logs/llama_experiments.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LLAMAExperiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                strategy_name TEXT,
                lora_config TEXT,
                train_loss REAL,
                val_loss REAL,
                metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS GeneratedResponses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                input_text TEXT,
                response_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES LLAMAExperiments(id)
            )
        """)
        self.conn.commit()

    def log_experiment(self, model_name: str, strategy_name: str, 
                      lora_config: Dict[str, Any], train_loss: float, 
                      val_loss: float, metrics: Dict[str, float]) -> int:
        """Log experiment details to database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO LLAMAExperiments 
            (model_name, strategy_name, lora_config, train_loss, val_loss, metrics) 
            VALUES (?, ?, ?, ?, ?, ?)""",
            (model_name, strategy_name, json.dumps(lora_config), 
             train_loss, val_loss, json.dumps(metrics))
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_response(self, exp_id: int, inp: str, resp: str):
        """Log generated response to database"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO GeneratedResponses (experiment_id, input_text, response_text) VALUES (?, ?, ?)",
            (exp_id, inp, resp)
        )
        self.conn.commit()

    def get_experiment(self, exp_id: int) -> Dict[str, Any]:
        """Retrieve experiment by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM LLAMAExperiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "model_name": row[1],
                "strategy_name": row[2],
                "lora_config": json.loads(row[3]),
                "train_loss": row[4],
                "val_loss": row[5],
                "metrics": json.loads(row[6]),
                "timestamp": row[7]
            }
        return None

    def get_responses(self, exp_id: int) -> list:
        """Retrieve all responses for an experiment"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT input_text, response_text FROM GeneratedResponses WHERE experiment_id = ?",
            (exp_id,)
        )
        return cursor.fetchall()

    def close(self):
        """Close database connection"""
        self.conn.close()
        