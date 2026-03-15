"""
Bharat AI Pro v4.0 — Production Backend
Flask + SQLite + Ollama + Streaming + Auth + Chat History + File Upload
"""

import os
import time
import uuid
import base64
import logging
import sqlite3
import json
import hashlib
import re
from datetime import datetime, timedelta
from functools import wraps
from logging.handlers import RotatingFileHandler

import requests
from flask import (
    Flask, request, jsonify, Response,
    stream_with_context, render_template, session, send_file
)
from werkzeug.utils import secure_filename

# ─── Optional heavy deps (graceful degradation) ───────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_OK = True
except ImportError:
    try:
        import pypdf as _pypdf
        PdfReader = _pypdf.PdfReader
        PYPDF2_OK = True
    except ImportError:
        PYPDF2_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, 'bharat_ai.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMP_FRAMES   = os.path.join(BASE_DIR, 'temp_frames')
AVATAR_FOLDER = os.path.join(BASE_DIR, 'avatars')
LOG_FILE      = os.path.join(BASE_DIR, 'app.log')

for d in [UPLOAD_FOLDER, TEMP_FRAMES, AVATAR_FOLDER]:
    os.makedirs(d, exist_ok=True)

ALLOWED_MEDIA     = {'png','jpg','jpeg','gif','webp','mp4','mov','avi','mkv','webm'}
ALLOWED_DOCS      = {'pdf','txt','md','doc','docx'}
ALLOWED_AVATARS   = {'png','jpg','jpeg','gif','webp'}
MAX_UPLOAD_MB     = 50
MAX_DOC_MB        = 10
MAX_AVATAR_MB     = 5
MAX_CONTEXT_CHARS = 12000   # chars of history sent to Ollama
OLLAMA_BASE_URL   = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_TIMEOUT    = int(os.environ.get('OLLAMA_TIMEOUT', '120'))
VISION_MODELS     = {'llava', 'llava-phi3', 'llava:7b', 'llava:13b', 'moondream', 'bakllava'}
SECRET_KEY        = os.environ.get('SECRET_KEY', 'bharat-ai-secret-key-change-in-production-' + str(uuid.uuid4()))

app = Flask(__name__, template_folder='templates')
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bharat_ai')
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s'))
logger.addHandler(handler)

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            full_name   TEXT NOT NULL,
            username    TEXT NOT NULL UNIQUE,
            password    TEXT NOT NULL,
            birth_year  INTEGER NOT NULL,
            email       TEXT,
            mobile      TEXT,
            avatar_path TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chats (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT NOT NULL DEFAULT 'New Chat',
            model       TEXT,
            pinned      INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            chat_id     TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            model       TEXT,
            lang        TEXT DEFAULT 'english',
            persona     TEXT,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS user_settings (
            user_id         TEXT PRIMARY KEY,
            selected_model  TEXT DEFAULT 'llama3',
            theme           TEXT DEFAULT 'dark',
            accent_color    TEXT DEFAULT '#7c6fff',
            language        TEXT DEFAULT 'english',
            system_prompt   TEXT DEFAULT '',
            writing_style   TEXT DEFAULT 'default',
            web_search      INTEGER DEFAULT 0,
            auto_speak      INTEGER DEFAULT 0,
            sound_effects   INTEGER DEFAULT 0,
            stream_mode     INTEGER DEFAULT 1,
            temperature     REAL DEFAULT 0.7,
            max_tokens      INTEGER DEFAULT 2048,
            extra_json      TEXT DEFAULT '{}',
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS message_feedback (
            id          TEXT PRIMARY KEY,
            message_id  TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            rating_type TEXT,
            star_rating INTEGER,
            feedback    TEXT,
            created_at  TEXT NOT NULL,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS prompt_templates (
            id          TEXT PRIMARY KEY,
            user_id     TEXT,
            title       TEXT NOT NULL,
            emoji       TEXT DEFAULT '📋',
            content     TEXT NOT NULL,
            is_system   INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
        CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
        CREATE INDEX IF NOT EXISTS idx_chats_user    ON chats(user_id);
        """)
    logger.info("Database initialised at %s", DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def check_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed


def ok(data=None, **kwargs):
    payload = {'success': True}
    if data is not None:
        payload['data'] = data
    payload.update(kwargs)
    return jsonify(payload)


def err(msg: str, code: int = 400):
    logger.warning("API error %s: %s", code, msg)
    return jsonify({'success': False, 'error': msg}), code


def now_iso():
    return datetime.utcnow().isoformat()


def allowed_ext(filename: str, allowed_set: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return err('Authentication required', 401)
        return f(*args, **kwargs)
    return wrapper


def current_user_id():
    return session.get('user_id')


def get_user(user_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def safe_str(s, maxlen=500):
    if not s:
        return ''
    return str(s)[:maxlen]


# ══════════════════════════════════════════════════════════════════════════════
#  OLLAMA INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def ollama_is_online() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def build_prompt_context(chat_history: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """Trim old messages so total context stays within budget."""
    messages = []
    total = 0
    for msg in reversed(chat_history):
        chunk = msg.get('content', '')
        total += len(chunk)
        if total > max_chars:
            break
        messages.insert(0, {'role': msg['role'], 'content': chunk})
    return messages


def generate_ai_response(
    prompt: str,
    model: str = 'llama3',
    history: list = None,
    system_prompt: str = '',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    image_base64: str = None,
) -> str:
    """Non-streaming AI response. Returns full text string."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if history:
        messages.extend(build_prompt_context(history))

    user_msg: dict = {'role': 'user', 'content': prompt}
    if image_base64 and model in VISION_MODELS:
        user_msg['images'] = [image_base64]
    messages.append(user_msg)

    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
        }
    }
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        r.raise_for_status()
        return r.json()['message']['content']
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is offline. Please start Ollama and try again.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama took too long to respond. Try a shorter message or faster model.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {str(e)}")


def generate_stream_response(
    prompt: str,
    model: str = 'llama3',
    history: list = None,
    system_prompt: str = '',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    image_base64: str = None,
):
    """Generator that yields text chunks from Ollama streaming API."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if history:
        messages.extend(build_prompt_context(history))

    user_msg: dict = {'role': 'user', 'content': prompt}
    if image_base64 and model in VISION_MODELS:
        user_msg['images'] = [image_base64]
    messages.append(user_msg)

    payload = {
        'model': model,
        'messages': messages,
        'stream': True,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
        }
    }
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=OLLAMA_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get('message', {}).get('content', '')
                        if token:
                            yield token
                        if chunk.get('done'):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ **Error:** Ollama is offline. Please start Ollama (`ollama serve`) and try again."
    except requests.exceptions.Timeout:
        yield "\n\n⚠️ **Error:** Request timed out. Try a shorter prompt or a lighter model."
    except Exception as e:
        yield f"\n\n⚠️ **Error:** {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  FILE / DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_file(filepath: str, filename: str) -> str:
    ext = filename.rsplit('.', 1)[-1].lower()
    try:
        if ext in ('txt', 'md'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()[:20000]

        if ext == 'pdf':
            if not PYPDF2_OK:
                return '[PDF extraction unavailable — install PyPDF2]'
            reader = PdfReader(filepath)
            texts = []
            for page in reader.pages[:30]:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return '\n\n'.join(texts)[:20000]

        if ext in ('doc', 'docx'):
            if not DOCX_OK:
                return '[DOCX extraction unavailable — install python-docx]'
            doc = DocxDocument(filepath)
            return '\n'.join(p.text for p in doc.paragraphs if p.text)[:20000]

        return '[Unsupported document format]'
    except Exception as e:
        return f'[Extraction error: {str(e)}]'


def image_to_base64(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def extract_video_frames(video_path: str, count: int = 3) -> list:
    if not CV2_OK:
        return []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = [int(i * total / count) for i in range(count)]
    frames = []
    base = os.path.join(TEMP_FRAMES, str(uuid.uuid4()))
    for idx, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            p = f"{base}_f{idx}.jpg"
            cv2.imwrite(p, frame)
            frames.append(p)
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        d = request.get_json(force=True) or {}
        full_name  = safe_str(d.get('full_name', '')).strip()
        username   = safe_str(d.get('username', '')).strip().lower()
        password   = d.get('password', '')
        birth_year = d.get('birth_year')
        email      = safe_str(d.get('email', '')).strip()
        mobile     = safe_str(d.get('mobile', '')).strip()

        # Validation
        if not full_name:
            return err('Full name is required')
        if not username or len(username) < 3:
            return err('Username must be at least 3 characters')
        if not re.match(r'^[a-z0-9_\.]+$', username):
            return err('Username may only contain letters, numbers, underscores, dots')
        if not password or len(password) < 8:
            return err('Password must be at least 8 characters')
        if not birth_year:
            return err('Birth year is required')
        try:
            birth_year = int(birth_year)
            if birth_year < 1900 or birth_year > datetime.utcnow().year:
                raise ValueError()
        except ValueError:
            return err('Invalid birth year')
        if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return err('Invalid email address')

        with get_db() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE username=?", (username,)
            ).fetchone()
            if existing:
                return err('Username already taken')

            uid = 'u_' + str(uuid.uuid4())
            conn.execute(
                """INSERT INTO users
                   (id, full_name, username, password, birth_year, email, mobile, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (uid, full_name, username, hash_password(password),
                 birth_year, email or None, mobile or None, now_iso())
            )
            conn.execute(
                "INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)", (uid,)
            )

        session.permanent = True
        session['user_id'] = uid
        session['username'] = username

        logger.info("New user registered: %s (%s)", username, uid)
        return ok({'user_id': uid, 'username': username, 'full_name': full_name})

    except Exception as e:
        logger.exception("Register error")
        return err(str(e), 500)


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        d = request.get_json(force=True) or {}
        identifier = safe_str(d.get('username', '')).strip().lower()
        password   = d.get('password', '')
        birth_year = d.get('birth_year')

        if not identifier or not password:
            return err('Username and password are required')

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=? OR email=?",
                (identifier, identifier)
            ).fetchone()

        if not user:
            return err('Invalid credentials')
        if not check_password(password, user['password']):
            return err('Invalid credentials')
        if birth_year:
            try:
                if int(birth_year) != user['birth_year']:
                    return err('Invalid credentials')
            except (ValueError, TypeError):
                return err('Invalid birth year')

        session.permanent = True
        session['user_id'] = user['id']
        session['username'] = user['username']

        logger.info("Login: %s", user['username'])
        return ok({
            'user_id':   user['id'],
            'username':  user['username'],
            'full_name': user['full_name'],
            'email':     user['email'] or '',
            'mobile':    user['mobile'] or '',
            'birth_year': user['birth_year'],
            'avatar_path': user['avatar_path'] or '',
            'created_at': user['created_at'],
        })

    except Exception as e:
        logger.exception("Login error")
        return err(str(e), 500)


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return ok(message='Logged out')


@app.route('/api/auth/me', methods=['GET'])
@login_required
def auth_me():
    user = get_user(current_user_id())
    if not user:
        return err('User not found', 404)
    user.pop('password', None)
    return ok(user)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — PROFILE
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    user = get_user(current_user_id())
    if not user:
        return err('User not found', 404)
    user.pop('password', None)
    return ok(user)


@app.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    try:
        d    = request.get_json(force=True) or {}
        uid  = current_user_id()
        user = get_user(uid)
        if not user:
            return err('User not found', 404)

        full_name  = safe_str(d.get('full_name', user['full_name'])).strip() or user['full_name']
        email      = safe_str(d.get('email', user.get('email', ''))).strip()
        mobile     = safe_str(d.get('mobile', user.get('mobile', ''))).strip()
        birth_year = d.get('birth_year', user['birth_year'])

        try:
            birth_year = int(birth_year)
        except (ValueError, TypeError):
            birth_year = user['birth_year']

        if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
            return err('Invalid email address')

        with get_db() as conn:
            conn.execute(
                """UPDATE users
                   SET full_name=?, email=?, mobile=?, birth_year=?
                   WHERE id=?""",
                (full_name, email or None, mobile or None, birth_year, uid)
            )
        return ok(message='Profile updated')
    except Exception as e:
        logger.exception("Profile update error")
        return err(str(e), 500)


@app.route('/api/profile/avatar', methods=['POST'])
@login_required
def upload_avatar():
    try:
        uid = current_user_id()
        if 'avatar' not in request.files:
            return err('No file uploaded')
        f = request.files['avatar']
        if not f.filename or not allowed_ext(f.filename, ALLOWED_AVATARS):
            return err('Invalid image type')
        if request.content_length and request.content_length > MAX_AVATAR_MB * 1024 * 1024:
            return err(f'Image too large (max {MAX_AVATAR_MB}MB)')

        ext = f.filename.rsplit('.', 1)[-1].lower()
        fname = f"avatar_{uid}.{ext}"
        fpath = os.path.join(AVATAR_FOLDER, fname)
        f.save(fpath)

        with get_db() as conn:
            conn.execute("UPDATE users SET avatar_path=? WHERE id=?", (fname, uid))

        return ok({'avatar_path': fname, 'avatar_url': f'/api/avatars/{fname}'})
    except Exception as e:
        logger.exception("Avatar upload error")
        return err(str(e), 500)


@app.route('/api/avatars/<filename>')
def serve_avatar(filename):
    path = os.path.join(AVATAR_FOLDER, secure_filename(filename))
    if not os.path.exists(path):
        return err('Not found', 404)
    return send_file(path)


@app.route('/api/profile/change-password', methods=['POST'])
@login_required
def change_password():
    try:
        d   = request.get_json(force=True) or {}
        uid = current_user_id()
        cur = d.get('current_password', '')
        nw  = d.get('new_password', '')
        cf  = d.get('confirm_password', '')

        with get_db() as conn:
            user = conn.execute("SELECT password FROM users WHERE id=?", (uid,)).fetchone()
        if not user or not check_password(cur, user['password']):
            return err('Current password is incorrect')
        if len(nw) < 8:
            return err('New password must be at least 8 characters')
        if nw != cf:
            return err("Passwords don't match")

        with get_db() as conn:
            conn.execute("UPDATE users SET password=? WHERE id=?", (hash_password(nw), uid))
        return ok(message='Password changed')
    except Exception as e:
        return err(str(e), 500)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/settings', methods=['GET'])
@login_required
def get_settings():
    uid = current_user_id()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM user_settings WHERE user_id=?", (uid,)).fetchone()
    if not row:
        with get_db() as conn:
            conn.execute("INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)", (uid,))
        return ok({})
    s = dict(row)
    try:
        s['extra'] = json.loads(s.get('extra_json') or '{}')
    except Exception:
        s['extra'] = {}
    s.pop('extra_json', None)
    return ok(s)


@app.route('/api/settings', methods=['POST'])
@login_required
def save_settings():
    try:
        d   = request.get_json(force=True) or {}
        uid = current_user_id()
        fields = {
            'selected_model': safe_str(d.get('selected_model', 'llama3'), 50),
            'theme':          safe_str(d.get('theme', 'dark'), 20),
            'accent_color':   safe_str(d.get('accent_color', '#7c6fff'), 20),
            'language':       safe_str(d.get('language', 'english'), 20),
            'system_prompt':  safe_str(d.get('system_prompt', ''), 2000),
            'writing_style':  safe_str(d.get('writing_style', 'default'), 30),
            'web_search':     int(bool(d.get('web_search', False))),
            'auto_speak':     int(bool(d.get('auto_speak', False))),
            'sound_effects':  int(bool(d.get('sound_effects', False))),
            'stream_mode':    int(bool(d.get('stream_mode', True))),
            'temperature':    float(d.get('temperature', 0.7)),
            'max_tokens':     int(d.get('max_tokens', 2048)),
            'extra_json':     json.dumps(d.get('extra', {})),
        }
        with get_db() as conn:
            conn.execute("""
                INSERT INTO user_settings
                  (user_id, selected_model, theme, accent_color, language,
                   system_prompt, writing_style, web_search, auto_speak,
                   sound_effects, stream_mode, temperature, max_tokens, extra_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id) DO UPDATE SET
                  selected_model=excluded.selected_model, theme=excluded.theme,
                  accent_color=excluded.accent_color, language=excluded.language,
                  system_prompt=excluded.system_prompt, writing_style=excluded.writing_style,
                  web_search=excluded.web_search, auto_speak=excluded.auto_speak,
                  sound_effects=excluded.sound_effects, stream_mode=excluded.stream_mode,
                  temperature=excluded.temperature, max_tokens=excluded.max_tokens,
                  extra_json=excluded.extra_json
            """, (uid, *fields.values()))
        return ok(message='Settings saved')
    except Exception as e:
        logger.exception("Settings save error")
        return err(str(e), 500)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — CHATS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/chats', methods=['GET'])
@login_required
def list_chats():
    uid = current_user_id()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE user_id=? ORDER BY pinned DESC, updated_at DESC",
            (uid,)
        ).fetchall()
    return ok([dict(r) for r in rows])


@app.route('/api/chats', methods=['POST'])
@login_required
def create_chat():
    try:
        d     = request.get_json(force=True) or {}
        uid   = current_user_id()
        title = safe_str(d.get('title', 'New Conversation'), 120)
        model = safe_str(d.get('model', 'llama3'), 50)
        cid   = 'chat_' + str(uuid.uuid4())
        now   = now_iso()
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chats (id, user_id, title, model, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (cid, uid, title, model, now, now)
            )
        return ok({'chat_id': cid, 'title': title, 'model': model, 'created_at': now})
    except Exception as e:
        return err(str(e), 500)


@app.route('/api/chats/<chat_id>', methods=['GET'])
@login_required
def get_chat(chat_id):
    uid = current_user_id()
    with get_db() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id=? AND user_id=?", (chat_id, uid)
        ).fetchone()
    if not chat:
        return err('Chat not found', 404)
    with get_db() as conn:
        msgs = conn.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY timestamp ASC", (chat_id,)
        ).fetchall()
    return ok({'chat': dict(chat), 'messages': [dict(m) for m in msgs]})


@app.route('/api/chats/<chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    uid = current_user_id()
    with get_db() as conn:
        conn.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, uid))
    return ok(message='Chat deleted')


@app.route('/api/chats/<chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(chat_id):
    uid   = current_user_id()
    d     = request.get_json(force=True) or {}
    title = safe_str(d.get('title', ''), 120).strip()
    if not title:
        return err('Title is required')
    with get_db() as conn:
        conn.execute(
            "UPDATE chats SET title=?, updated_at=? WHERE id=? AND user_id=?",
            (title, now_iso(), chat_id, uid)
        )
    return ok(message='Renamed')


@app.route('/api/chats/<chat_id>/pin', methods=['POST'])
@login_required
def pin_chat(chat_id):
    uid = current_user_id()
    d   = request.get_json(force=True) or {}
    pin = int(bool(d.get('pinned', True)))
    with get_db() as conn:
        conn.execute(
            "UPDATE chats SET pinned=? WHERE id=? AND user_id=?",
            (pin, chat_id, uid)
        )
    return ok(message='Pinned' if pin else 'Unpinned')


@app.route('/api/chats/search', methods=['GET'])
@login_required
def search_chats():
    uid = current_user_id()
    q   = request.args.get('q', '').strip()
    if not q:
        return ok([])
    pattern = f'%{q}%'
    with get_db() as conn:
        rows = conn.execute("""
            SELECT DISTINCT c.id, c.title, c.updated_at,
                   m.content as snippet
            FROM chats c
            LEFT JOIN messages m ON m.chat_id = c.id
            WHERE c.user_id=?
              AND (c.title LIKE ? OR m.content LIKE ?)
            ORDER BY c.updated_at DESC
            LIMIT 20
        """, (uid, pattern, pattern)).fetchall()
    return ok([dict(r) for r in rows])


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
@login_required
def add_message(chat_id):
    try:
        uid = current_user_id()
        d   = request.get_json(force=True) or {}
        # Verify ownership
        with get_db() as conn:
            chat = conn.execute(
                "SELECT id FROM chats WHERE id=? AND user_id=?", (chat_id, uid)
            ).fetchone()
        if not chat:
            return err('Chat not found', 404)

        role    = d.get('role', 'user')
        content = safe_str(d.get('content', ''), 32000)
        model   = safe_str(d.get('model', ''), 50)
        lang    = safe_str(d.get('lang', 'english'), 30)
        persona = safe_str(d.get('persona', ''), 60)
        mid     = 'msg_' + str(uuid.uuid4())
        ts      = now_iso()

        with get_db() as conn:
            conn.execute(
                """INSERT INTO messages (id, chat_id, user_id, role, content, model, lang, persona, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (mid, chat_id, uid, role, content, model, lang, persona, ts)
            )
            conn.execute(
                "UPDATE chats SET updated_at=? WHERE id=?", (ts, chat_id)
            )
        return ok({'message_id': mid, 'timestamp': ts})
    except Exception as e:
        return err(str(e), 500)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — AI CHAT (STREAMING)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/chat-stream', methods=['POST'])
def chat_stream_route():
    """
    Main streaming chat endpoint.
    Reads recent DB history if chat_id provided; falls back to in-payload history.
    """
    try:
        d           = request.get_json(force=True) or {}
        message     = safe_str(d.get('message', ''), 8000).strip()
        model       = safe_str(d.get('model', 'llama3'), 50)
        chat_id     = d.get('chat_id')
        temperature = float(d.get('temperature', 0.7))
        max_tokens  = int(d.get('max_tokens', 2048))
        top_p       = float(d.get('top_p', 0.9))
        system_p    = safe_str(d.get('persona') or d.get('system_prompt', ''), 2000)
        uid         = session.get('user_id')

        if not message:
            return err('Message is required')

        # Load DB history
        history = []
        if chat_id and uid:
            with get_db() as conn:
                rows = conn.execute(
                    """SELECT role, content FROM messages
                       WHERE chat_id=? ORDER BY timestamp DESC LIMIT 30""",
                    (chat_id,)
                ).fetchall()
            history = [{'role': r['role'], 'content': r['content']} for r in reversed(rows)]

        def generate():
            for chunk in generate_stream_response(
                prompt=message,
                model=model,
                history=history,
                system_prompt=system_p,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            ):
                yield chunk

        return Response(stream_with_context(generate()), content_type='text/plain; charset=utf-8')

    except Exception as e:
        logger.exception("Stream route error")
        return err(str(e), 500)


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream_alias():
    """Alias for frontend compatibility."""
    return chat_stream_route()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — MEDIA / DOCUMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/analyze-media', methods=['POST'])
def analyze_media():
    """Handle image, video, and document uploads for AI analysis."""
    filepath = None
    try:
        if 'file' not in request.files:
            return err('No file uploaded')

        f            = request.files['file']
        user_message = safe_str(request.form.get('message', 'Analyze this media in detail.'), 2000)
        model        = safe_str(request.form.get('model', 'llava'), 50)
        session_id   = request.form.get('session_id', '')

        if not f.filename:
            return err('Empty filename')

        fname = secure_filename(f.filename)
        ext   = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
        mime  = f.mimetype or ''

        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{fname}")
        f.save(filepath)

        # ── IMAGE ──────────────────────────────────────────
        if mime.startswith('image/') or ext in ('png','jpg','jpeg','gif','webp'):
            img_b64 = image_to_base64(filepath)
            # Force vision model
            vis_model = model if model in VISION_MODELS else 'llava'
            response_text = generate_ai_response(
                prompt=user_message,
                model=vis_model,
                image_base64=img_b64,
            )
            return ok({
                'response': response_text,
                'file_name': fname,
                'file_type': mime or 'image',
                'mode': 'vision',
            })

        # ── VIDEO ──────────────────────────────────────────
        if mime.startswith('video/') or ext in ('mp4','mov','avi','mkv','webm'):
            if not CV2_OK:
                return ok({'response': '[Video analysis requires OpenCV — install opencv-python]', 'file_type': 'video'})
            frames = extract_video_frames(filepath, count=3)
            if not frames:
                return ok({'response': 'Could not extract frames from video.', 'file_type': 'video'})
            try:
                vis_model = model if model in VISION_MODELS else 'llava'
                frame_analyses = []
                for i, fp in enumerate(frames):
                    b64 = image_to_base64(fp)
                    analysis = generate_ai_response(
                        prompt=f'Frame {i+1}: Describe in detail — objects, people, actions, setting.',
                        model=vis_model,
                        image_base64=b64,
                    )
                    frame_analyses.append(f'Frame {i+1}: {analysis}')

                summary_prompt = (
                    f"User Query: {user_message}\n\n"
                    f"Key frame analyses:\n" + "\n".join(frame_analyses) +
                    "\n\nProvide a comprehensive answer based on these frames."
                )
                final = generate_ai_response(prompt=summary_prompt, model='llama3')
                return ok({'response': final, 'file_type': 'video', 'mode': 'video_analysis'})
            finally:
                for fp in frames:
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

        # ── DOCUMENT ───────────────────────────────────────
        if ext in ALLOWED_DOCS:
            doc_text = extract_text_from_file(filepath, fname)
            if not doc_text.strip():
                return ok({'response': 'Could not extract text from this document.', 'file_type': ext})
            truncated = doc_text[:MAX_CONTEXT_CHARS]
            prompt = (
                f"The user uploaded a document. Here is its content:\n\n"
                f"---\n{truncated}\n---\n\n"
                f"User's request: {user_message}"
            )
            response_text = generate_ai_response(
                prompt=prompt,
                model=model if model not in VISION_MODELS else 'llama3',
            )
            return ok({
                'response': response_text,
                'file_name': fname,
                'file_type': ext,
                'doc_length': len(doc_text),
                'mode': 'document',
            })

        return err(f'Unsupported file type: {ext}')

    except RuntimeError as e:
        return err(str(e), 503)
    except Exception as e:
        logger.exception("Media analysis error")
        return err(str(e), 500)
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


@app.route('/api/upload/document', methods=['POST'])
@login_required
def upload_document():
    """Upload a document and return extracted text (no AI call)."""
    try:
        if 'file' not in request.files:
            return err('No file')
        f    = request.files['file']
        fname = secure_filename(f.filename or 'upload')
        if not allowed_ext(fname, ALLOWED_DOCS):
            return err('Invalid document type')
        if request.content_length and request.content_length > MAX_DOC_MB * 1024 * 1024:
            return err(f'Document too large (max {MAX_DOC_MB}MB)')

        filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{fname}")
        f.save(filepath)
        try:
            text = extract_text_from_file(filepath, fname)
        finally:
            os.remove(filepath)

        return ok({'file_name': fname, 'text': text[:MAX_CONTEXT_CHARS], 'length': len(text)})
    except Exception as e:
        return err(str(e), 500)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/export/chat/<chat_id>', methods=['GET'])
@login_required
def export_chat(chat_id):
    uid  = current_user_id()
    fmt  = request.args.get('format', 'txt').lower()

    with get_db() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id=? AND user_id=?", (chat_id, uid)
        ).fetchone()
    if not chat:
        return err('Chat not found', 404)

    with get_db() as conn:
        msgs = conn.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY timestamp ASC", (chat_id,)
        ).fetchall()

    chat = dict(chat)
    msgs = [dict(m) for m in msgs]
    title = chat.get('title', 'Chat')

    if fmt == 'json':
        content = json.dumps({'chat': chat, 'messages': msgs}, indent=2, ensure_ascii=False)
        mime = 'application/json'
        ext  = 'json'

    elif fmt == 'md':
        lines = [f"# {title}\n", f"*Exported {datetime.utcnow().strftime('%Y-%m-%d')}*\n\n---\n"]
        for m in msgs:
            role = 'You' if m['role'] == 'user' else f"Bharat AI ({m.get('model','')})"
            lines.append(f"## {role}  \n*{m['timestamp']}*\n\n{m['content']}\n\n---\n")
        content = '\n'.join(lines)
        mime = 'text/markdown'
        ext  = 'md'

    else:  # txt
        lines = [f"=== {title} ===", f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}\n"]
        for m in msgs:
            role = 'YOU' if m['role'] == 'user' else 'BHARAT AI'
            lines.append(f"[{role}] {m['timestamp']}\n{m['content']}\n{'-'*60}")
        content = '\n'.join(lines)
        mime = 'text/plain'
        ext  = 'txt'

    safe_title = re.sub(r'[^\w\s-]', '', title)[:40].strip().replace(' ', '_')
    filename   = f"bharat_ai_{safe_title}.{ext}"

    return Response(
        content,
        mimetype=mime,
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/messages/<message_id>/feedback', methods=['POST'])
@login_required
def message_feedback(message_id):
    try:
        uid  = current_user_id()
        d    = request.get_json(force=True) or {}
        rt   = safe_str(d.get('rating_type', ''), 20)   # 'like' | 'dislike'
        star = d.get('star_rating')
        fb   = safe_str(d.get('feedback', ''), 1000)

        if star is not None:
            try:
                star = int(star)
                if not 1 <= star <= 5:
                    star = None
            except (ValueError, TypeError):
                star = None

        fid = 'fb_' + str(uuid.uuid4())
        with get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO message_feedback
                   (id, message_id, user_id, rating_type, star_rating, feedback, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (fid, message_id, uid, rt or None, star, fb or None, now_iso())
            )
        return ok(message='Feedback recorded')
    except Exception as e:
        return err(str(e), 500)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — USAGE STATS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/usage/stats', methods=['GET'])
@login_required
def usage_stats():
    uid = current_user_id()
    with get_db() as conn:
        total_chats = conn.execute(
            "SELECT COUNT(*) as c FROM chats WHERE user_id=?", (uid,)
        ).fetchone()['c']

        total_msgs = conn.execute(
            "SELECT COUNT(*) as c FROM messages WHERE user_id=?", (uid,)
        ).fetchone()['c']

        user_msgs = conn.execute(
            "SELECT COUNT(*) as c FROM messages WHERE user_id=? AND role='user'", (uid,)
        ).fetchone()['c']

        ai_msgs = conn.execute(
            "SELECT COUNT(*) as c FROM messages WHERE user_id=? AND role='ai'", (uid,)
        ).fetchone()['c']

        top_model = conn.execute(
            """SELECT model, COUNT(*) as cnt FROM messages
               WHERE user_id=? AND role='ai' AND model IS NOT NULL AND model!=''
               GROUP BY model ORDER BY cnt DESC LIMIT 1""",
            (uid,)
        ).fetchone()

        last_active = conn.execute(
            "SELECT MAX(updated_at) as la FROM chats WHERE user_id=?", (uid,)
        ).fetchone()['la']

        user_row = conn.execute(
            "SELECT created_at FROM users WHERE id=?", (uid,)
        ).fetchone()

    created = user_row['created_at'][:10] if user_row else str(datetime.utcnow().date())
    try:
        days_active = (datetime.utcnow().date() - datetime.fromisoformat(created).date()).days + 1
    except Exception:
        days_active = 1

    return ok({
        'total_chats':    total_chats,
        'total_messages': total_msgs,
        'user_messages':  user_msgs,
        'ai_messages':    ai_msgs,
        'most_used_model': top_model['model'] if top_model else 'llama3',
        'last_active':    last_active,
        'days_active':    days_active,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

BUILT_IN_TEMPLATES = [
    {'id': 'tpl_001', 'emoji': '📝', 'title': 'Summarize Text',       'content': 'Summarize the following text in 5 bullet points:\n\n[paste text here]'},
    {'id': 'tpl_002', 'emoji': '🐛', 'title': 'Debug Code',           'content': 'Debug this code and explain what is wrong:\n\n```\n[paste code here]\n```'},
    {'id': 'tpl_003', 'emoji': '📧', 'title': 'Write Email',          'content': 'Write a professional email for the following purpose:\n\n[describe purpose]'},
    {'id': 'tpl_004', 'emoji': '🧒', 'title': 'Explain Simply',       'content': 'Explain [topic] like I am 5 years old, using simple analogies and examples.'},
    {'id': 'tpl_005', 'emoji': '📈', 'title': 'Marketing Strategy',   'content': 'Create a detailed marketing strategy for:\n\nProduct: [name]\nAudience: [audience]\nBudget: [budget]'},
    {'id': 'tpl_006', 'emoji': '🌐', 'title': 'Translate & Pronounce','content': 'Translate the following text to [target language] and provide pronunciation tips:\n\n[text here]'},
    {'id': 'tpl_007', 'emoji': '💡', 'title': 'Idea Generator',       'content': 'Generate 10 creative ideas for:\n\n[topic or problem]'},
    {'id': 'tpl_008', 'emoji': '👀', 'title': 'Code Review',          'content': 'Review the following code for best practices, security issues, and performance:\n\n```\n[paste code]\n```'},
]


@app.route('/api/templates', methods=['GET'])
def get_templates():
    uid = session.get('user_id')
    templates = list(BUILT_IN_TEMPLATES)
    if uid:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_templates WHERE user_id=? OR is_system=1 ORDER BY created_at",
                (uid,)
            ).fetchall()
        templates += [dict(r) for r in rows if r['id'] not in {t['id'] for t in templates}]
    return ok(templates)


@app.route('/api/templates', methods=['POST'])
@login_required
def save_template():
    try:
        uid = current_user_id()
        d   = request.get_json(force=True) or {}
        tid = 'tpl_' + str(uuid.uuid4())[:8]
        with get_db() as conn:
            conn.execute(
                """INSERT INTO prompt_templates (id, user_id, title, emoji, content, is_system, created_at)
                   VALUES (?,?,?,?,?,0,?)""",
                (tid, uid,
                 safe_str(d.get('title', 'My Template'), 80),
                 safe_str(d.get('emoji', '📋'), 10),
                 safe_str(d.get('content', ''), 4000),
                 now_iso())
            )
        return ok({'template_id': tid})
    except Exception as e:
        return err(str(e), 500)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — STATUS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/status', methods=['GET'])
def status():
    online = ollama_is_online()
    models = []
    if online:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=4)
            if r.ok:
                models = [m['name'] for m in r.json().get('models', [])]
        except Exception:
            pass
    return ok({
        'ollama': online,
        'models': models,
        'server': 'Bharat AI Pro v4.0',
        'timestamp': now_iso(),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY ENDPOINTS (backwards compat with original frontend JS)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/history/<session_id>', methods=['GET'])
def legacy_history(session_id):
    return jsonify([])


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorised(e):
    return jsonify({'success': False, 'error': 'Unauthorised'}), 401

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': f'File too large (max {MAX_UPLOAD_MB}MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    logger.exception("Unhandled 500")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    online = ollama_is_online()
    print("\n" + "═" * 55)
    print("   Bharat AI Pro v4.0 — Production Backend")
    print("═" * 55)
    print(f"   URL     : http://127.0.0.1:5000/")
    print(f"   Database: {DB_PATH}")
    print(f"   Uploads : {UPLOAD_FOLDER}")
    print(f"   Ollama  : {'✓ Online' if online else '✗ Offline — run: ollama serve'}")
    if online:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            models = [m['name'] for m in r.json().get('models', [])]
            print(f"   Models  : {', '.join(models) if models else 'none pulled yet'}")
        except Exception:
            pass
    print("═" * 55 + "\n")
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=True)