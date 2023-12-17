from flask import Flask, render_template, request
from datetime import datetime
import sqlite3
import locale

locale.setlocale(locale.LC_TIME, 'ja_JP.UTF-8')

app = Flask(__name__)

@app.route('/')
def index():
    conn = sqlite3.connect('database.db')
    threads = []
    for thread_id, title in conn.execute("SELECT id, title FROM thread").fetchall():
        posts = []
        for body, timestamp in conn.execute("SELECT body, timestamp FROM post WHERE thread_id = ? ORDER BY id", (thread_id,)).fetchall():
            posts.append({"body": body.replace('\n', '<br>'), "timestamp": datetime.fromisoformat(timestamp)})
        threads.append({"title": title, "posts": posts})
    conn.close()
    return render_template('board.html', threads=threads)

if __name__ == "__main__":
    app.run()