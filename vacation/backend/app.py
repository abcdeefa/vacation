from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import sqlite3

app = FastAPI()

# DB 초기화
conn = sqlite3.connect("study.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS study_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    concept TEXT,
    learned_at TEXT,
    next_review TEXT
)
""")
conn.commit()

# 데이터 모델
class StudyRecord(BaseModel):
    user_id: str
    concept: str
    learned_at: str  # YYYY-MM-DD
    next_review: str # YYYY-MM-DD

# 학습 기록 추가
@app.post("/add_record")
def add_record(record: StudyRecord):
    cursor.execute("""
        INSERT INTO study_records (user_id, concept, learned_at, next_review)
        VALUES (?, ?, ?, ?)
    """, (record.user_id, record.concept, record.learned_at, record.next_review))
    conn.commit()
    return {"message": "학습 기록 추가됨", "next_review": record.next_review}

# 오늘 복습 목록 조회
@app.get("/due_today/{user_id}")
def get_due_today(user_id: str):
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT concept, next_review FROM study_records WHERE user_id=? AND next_review=?
    """, (user_id, today))
    return {"due_today": cursor.fetchall()}
