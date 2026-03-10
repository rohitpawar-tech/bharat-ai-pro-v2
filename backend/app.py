from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import sqlite3
import json
import time
from ai_engine import generate_stream_response

app = Flask(__name__)
DB_NAME = "bharat_ai.db"

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Simple table for chat history
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_db_history(session_id, limit=10):
    """Fetches last N messages for context."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?", 
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    # Reverse to get chronological order
    return [dict(row) for row in reversed(rows)]

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
              (session_id, role, content))
    conn.commit()
    conn.close()

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/history/<session_id>')
def load_history(session_id):
    history = get_db_history(session_id, limit=50) # Load more for UI
    return jsonify(history)

# 1️⃣ Real-Time AI Streaming Endpoint
@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')

    # Save user message to DB
    save_message(session_id, 'user', user_message)

    # Fetch history for context (Memory)
    history = get_db_history(session_id, limit=10)
    # Remove the very last message (current user input) from history context 
    # because the AI engine builds the prompt with the current user input separately.
    if history and history[-1]['role'] == 'user':
        history = history[:-1]

    def generate():
        # Create a generator from the AI engine
        for chunk in generate_stream_response(history, user_message):
            # Send chunk to frontend
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)