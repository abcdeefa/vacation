import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [userId] = useState("user1");
  const [concept, setConcept] = useState("");
  const [learnedAt, setLearnedAt] = useState("");
  const [nextReview, setNextReview] = useState("");
  const [dueToday, setDueToday] = useState([]);

  const backendURL = "http://localhost:8000";

  // 오늘 복습 목록 가져오기
  const fetchDueToday = async () => {
    const res = await axios.get(`${backendURL}/due_today/${userId}`);
    setDueToday(res.data.due_today);
  };

  useEffect(() => {
    fetchDueToday();
  }, []);

  // 학습 기록 추가
  const addRecord = async () => {
    if (!concept || !learnedAt || !nextReview) {
      alert("개념, 학습 날짜, 복습 날짜를 모두 입력하세요.");
      return;
    }
    await axios.post(`${backendURL}/add_record`, {
      user_id: userId,
      concept,
      learned_at: learnedAt,
      next_review: nextReview
    });
    alert("학습 기록이 추가되었습니다.");
    fetchDueToday();
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>📚 학습 기록 관리</h1>

      <h2>학습 기록 추가</h2>
      <input placeholder="개념명" value={concept} onChange={e => setConcept(e.target.value)} />
      <input type="date" value={learnedAt} onChange={e => setLearnedAt(e.target.value)} />
      <input type="date" value={nextReview} onChange={e => setNextReview(e.target.value)} />
      <button onClick={addRecord}>추가</button>

      <h2>오늘 복습할 항목</h2>
      <ul>
        {dueToday.length > 0 ? (
          dueToday.map((item, idx) => (
            <li key={idx}>{item[0]} (복습일: {item[1]})</li>
          ))
        ) : (
          <li>오늘 복습 항목이 없습니다.</li>
        )}
      </ul>
      <button onClick={fetchDueToday}>새로고침</button>
    </div>
  );
}

export default App;
