# frontend/app.py
import streamlit as st
import requests
import uuid
import html
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# MUST BE FIRST - set page config
st.set_page_config(page_title="Document Q&A", layout="wide", page_icon="📚")

API_BASE_URL = "http://localhost:8000"

# Midnight AI Theme - Clean, dark, professional
def load_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

      /* ===== Midnight AI Color Palette ===== */
      :root {
        /* Backgrounds */
        --bg-app: #05070F;        /* App background - deepest dark */
        --bg-sidebar: #070B17;     /* Sidebar background */
        --bg-main: #0A1020;        /* Main content background */
        --bg-card: #0F172A;         /* Card background */
        --bg-hover: #16223B;        /* Hover state */
        
        /* Borders */
        --border-primary: #1F2A44;   /* Primary borders */
        --border-hover: #2E3B5E;     /* Hover borders */
        
        /* Text */
        --text-primary: #F1F5FF;     /* Primary text - bright white */
        --text-secondary: #9BA8C7;   /* Secondary text */
        --text-muted: #6B7A99;        /* Muted text */
        
        /* Accents */
        --accent-primary: #6366F1;    /* Primary accent - indigo */
        --accent-hover: #7C7FFF;      /* Hover state */
        --accent-secondary: #22D3EE;  /* Secondary accent - cyan */
        
        /* Chat */
        --chat-user: #6366F1;          /* User message bubble */
        --chat-assistant: #111827;     /* Assistant message bubble */
        
        /* Buttons */
        --btn-primary: #6366F1;        /* Primary button */
        --btn-hover: #7C7FFF;          /* Button hover */
        --btn-secondary: #1F2A44;      /* Secondary button */
        
        /* Alerts */
        --success-bg: #065F46;         /* Success background */
        --success-text: #6EE7B7;       /* Success text */
        --error-bg: #7F1D1D;           /* Error background */
        --error-text: #FCA5A5;          /* Error text */
        --info-bg: #0C4A6E;            /* Info background */
        --info-text: #7DD3FC;           /* Info text */
        
        /* Graphs */
        --graph-question: #6366F1;      /* Question markers */
        --graph-answer: #22D3EE;        /* Answer markers */
        --graph-grid: #1F2A44;           /* Grid lines */
        
        /* Shadows */
        --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.6);
        --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.7);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.8);
      }

      /* ===== Base Styles ===== */
      .stApp {
        background: var(--bg-app);
        font-family: 'Inter', sans-serif;
      }

      .main .block-container {
        max-width: 1400px !important;
        padding: 1.5rem 2rem !important;
        background: var(--bg-main);
        border-radius: 24px 24px 0 0;
        margin-top: 0;
      }

      /* ===== Sidebar Styles ===== */
      section[data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-primary);
      }

      section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem !important;
        height: 100vh !important;
        overflow-y: auto !important;
        display: flex;
        flex-direction: column;
      }

      /* Make sidebar content take available space */
      .sidebar-content {
        flex: 1;
      }

      /* Sidebar header */
      .sidebar-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(34, 211, 238, 0.05));
        padding: 1.5rem 1.2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-primary);
      }

      .sidebar-header h2 {
        color: var(--text-primary) !important;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
      }

      .sidebar-header p {
        color: var(--text-secondary) !important;
        font-size: 0.8rem;
        margin: 0;
      }

      .sidebar-header .session-badge {
        background: var(--bg-card);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
        display: inline-block;
        margin-top: 0.8rem;
        border: 1px solid var(--border-primary);
      }

      /* Bottom Navigation */
      .sidebar-nav {
        margin-top: auto;
        padding-top: 1rem;
        border-top: 1px solid var(--border-primary);
      }

      /* Custom navigation buttons */
      .nav-button {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 12px;
        color: var(--text-secondary) !important;
        text-decoration: none;
        transition: all 0.2s ease;
        cursor: pointer;
        border: 1px solid transparent;
      }

      .nav-button:hover {
        background: var(--bg-hover);
        border-color: var(--border-primary);
        color: var(--text-primary) !important;
      }

      .nav-button.active {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(34, 211, 238, 0.1));
        border-color: var(--accent-primary);
        color: var(--text-primary) !important;
      }

      .nav-button i {
        font-size: 1.2rem;
        color: var(--accent-primary);
      }

      /* ===== Cards ===== */
      .card {
        background: var(--bg-card);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
      }

      .card:hover {
        border-color: var(--border-hover);
        background: var(--bg-hover);
      }

      .card-title {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid var(--border-primary);
        padding-bottom: 0.75rem;
      }

      .card-title i {
        color: var(--accent-primary);
      }

      /* ===== Document Items ===== */
      .document-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .document-item:hover {
        background: var(--bg-hover);
        border-color: var(--accent-primary);
      }

      .document-name {
        color: var(--text-primary);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }

      .document-chunks {
        background: rgba(99, 102, 241, 0.15);
        color: var(--accent-secondary);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.3);
      }

      /* ===== Metrics ===== */
      .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin: 1rem 0;
      }

      .metric-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
      }

      .metric-value {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
      }

      .metric-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-top: 0.25rem;
      }

      /* ===== Chat Messages ===== */
      .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
      }

      .message-user {
        background: var(--chat-user);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-md);
      }

      .message-assistant {
        background: var(--chat-assistant);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 1rem;
        border: 1px solid var(--border-primary);
      }

      /* ===== Buttons ===== */
      .stButton > button {
        background: var(--btn-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
      }

      .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: var(--accent-primary) !important;
      }

      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        border: none !important;
        color: white !important;
      }

      .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
        transform: translateY(-1px);
      }

      /* ===== File Uploader ===== */
      .stFileUploader > div {
        border: 2px dashed var(--border-primary) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 1.5rem !important;
      }

      .stFileUploader > div:hover {
        border-color: var(--accent-primary) !important;
        background: rgba(99, 102, 241, 0.05) !important;
      }

      /* ===== Chat Input ===== */
      .stChatInput > div > div > input {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 16px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
      }

      .stChatInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
      }

      /* ===== Selectbox & Multiselect ===== */
      .stSelectbox > div > div,
      .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 12px !important;
      }

      /* Dropdown menus */
      div[data-baseweb="popover"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-primary) !important;
        box-shadow: var(--shadow-lg) !important;
      }

      div[data-baseweb="menu"] {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
      }

      div[data-baseweb="menu"] * {
        color: var(--text-primary) !important;
      }

      div[data-baseweb="menu"] li:hover {
        background: var(--bg-hover) !important;
      }

      /* Tags */
      span[data-baseweb="tag"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: var(--text-primary) !important;
        border-radius: 20px !important;
        padding: 0.25rem 0.75rem !important;
      }

      /* ===== Alerts ===== */
      .success-msg {
        background: var(--success-bg);
        color: var(--success-text);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(110, 231, 183, 0.2);
        margin: 1rem 0;
      }

      .error-msg {
        background: var(--error-bg);
        color: var(--error-text);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(252, 165, 165, 0.2);
        margin: 1rem 0;
      }

      .info-msg {
        background: var(--info-bg);
        color: var(--info-text);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(125, 211, 252, 0.2);
        margin: 1rem 0;
      }

      /* ===== Progress Bar ===== */
      .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
        border-radius: 10px !important;
      }

      /* ===== Expander ===== */
      .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
      }

      .streamlit-expanderHeader:hover {
        border-color: var(--accent-primary) !important;
      }

      /* ===== Headers ===== */
      h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
      }

      /* ===== Chat Title ===== */
      .chat-title {
        text-align: center;
        margin-bottom: 2rem;
      }

      .chat-title h1 {
        font-size: 2rem;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
      }

      .chat-title p {
        color: var(--text-secondary);
        font-size: 0.95rem;
      }

      /* Hide Streamlit branding */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = []
    if "available_sessions" not in st.session_state:
        st.session_state.available_sessions = []
    if "session_name" not in st.session_state:
        st.session_state.session_name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Chat"


@st.cache_data(ttl=5)
def cached_get_json(url: str):
    try:
        r = requests.get(url, timeout=30)
        if r and r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def safe_post(url: str, **kwargs):
    try:
        return requests.post(url, timeout=60, **kwargs)
    except Exception as e:
        st.markdown(f'<div class="error-msg">❌ Request failed: {e}</div>', unsafe_allow_html=True)
        return None


def safe_get(url: str, **kwargs):
    try:
        return requests.get(url, timeout=30, **kwargs)
    except Exception:
        return None


def format_timestamp(timestamp):
    if not timestamp:
        return "Unknown"
    
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = datetime.fromisoformat(str(timestamp))
        
        now = datetime.now()
        diff = now - dt
        
        if diff < timedelta(minutes=1):
            return "Just now"
        elif diff < timedelta(hours=1):
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes} min ago"
        elif diff < timedelta(days=1):
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff < timedelta(days=7):
            days = diff.days
            return f"{days} day{'s' if days > 1 else ''} ago"
        else:
            return dt.strftime("%b %d, %Y")
    except:
        return "Unknown"


def load_session(session_id: str, session_name: str = None):
    with st.spinner("Loading session..."):
        resp = safe_get(f"{API_BASE_URL}/conversation-history/{session_id}")
        if resp and resp.status_code == 200:
            data = resp.json()
            st.session_state.session_id = session_id
            if session_name:
                st.session_state.session_name = session_name
            st.session_state.conversation_history = data.get("history", [])
            st.markdown('<div class="success-msg">✅ Session loaded successfully!</div>', unsafe_allow_html=True)
            return True
        st.markdown('<div class="error-msg">❌ Failed to load session</div>', unsafe_allow_html=True)
        return False


def create_new_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    st.session_state.conversation_history = []
    st.markdown('<div class="success-msg">✨ New session created!</div>', unsafe_allow_html=True)
    st.rerun()


def load_available_sessions():
    data = cached_get_json(f"{API_BASE_URL}/sessions")
    if data and "sessions" in data:
        st.session_state.available_sessions = data["sessions"]
    else:
        st.session_state.available_sessions = [{
            "session_id": st.session_state.session_id,
            "name": st.session_state.session_name,
            "message_count": len(st.session_state.conversation_history),
            "last_updated": datetime.now().timestamp()
        }]
    st.session_state.last_refresh = datetime.now()


def submit_feedback(interaction_id: str, rating: int, message_index: int):
    if not interaction_id:
        st.markdown('<div class="error-msg">⚠️ Cannot submit feedback: No interaction ID</div>', unsafe_allow_html=True)
        return False
    
    with st.spinner("Submitting feedback..."):
        response = safe_post(
            f"{API_BASE_URL}/feedback",
            json={
                "interaction_id": interaction_id,
                "feedback_type": "rating",
                "feedback_data": {"rating": rating},
            },
        )
        
        if response and response.status_code == 200:
            st.session_state.conversation_history[message_index]['feedback_given'] = 'positive' if rating == 5 else 'negative'
            if rating == 5:
                st.markdown('<div class="success-msg">🎉 Thanks for the positive feedback!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-msg">👎 Feedback recorded. This will help improve future responses!</div>', unsafe_allow_html=True)
            return True
        else:
            st.markdown('<div class="error-msg">❌ Failed to submit feedback</div>', unsafe_allow_html=True)
            return False


def set_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()


def main():
    load_css()
    init_session()

    # Sidebar - Now with bottom navigation
    with st.sidebar:
        # Header at top
        st.markdown(f"""
        <div class="sidebar-header">
            <h2>Intelligent Document Q&A</h2>
            <p>Upload documents and ask questions with memory</p>
            <div class="session-badge">{html.escape(st.session_state.session_name[:20])}</div>
        </div>
        """, unsafe_allow_html=True)

        # Main sidebar content
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Document Management
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>📄</span> Document Management</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "docx", "txt", "html", "md"],
            help="Supported formats: PDF, DOCX, TXT, HTML, MD",
            label_visibility="collapsed"
        )

        if uploaded_file:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<span style='color: var(--text-secondary); font-size:0.85rem;'>{uploaded_file.name[:20]}...</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color: var(--accent-primary); font-size:0.75rem;'>{uploaded_file.size/1024:.1f}KB</span>", unsafe_allow_html=True)

        if st.button("📤 Process Document", use_container_width=True):
            if not uploaded_file:
                st.markdown('<div class="error-msg">❌ Please upload a file first.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = safe_post(f"{API_BASE_URL}/upload-document", files=files)

                    if resp and resp.status_code == 200:
                        result = resp.json()
                        doc_id = result["document_id"]

                        already = any(d["id"] == doc_id for d in st.session_state.uploaded_documents)
                        if not already:
                            st.session_state.uploaded_documents.append(
                                {"name": uploaded_file.name, "id": doc_id, "chunks": result["chunks_processed"]}
                            )

                        if doc_id not in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.append(doc_id)

                        st.markdown(f'<div class="success-msg">✅ Document processed! {result["chunks_processed"]} chunks indexed.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-msg">❌ Error processing document</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Uploaded Documents
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>📑</span> Uploaded Documents</div>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_documents:
            for doc in st.session_state.uploaded_documents:
                st.markdown(f"""
                <div class="document-item">
                    <span class="document-name">📄 {doc['name'][:15]}...</span>
                    <span class="document-chunks">{doc['chunks']}</span>
                </div>
                """, unsafe_allow_html=True)

            options = [f"{d['name']} ({d['id'][:6]}...)" for d in st.session_state.uploaded_documents]
            id_map = {f"{d['name']} ({d['id'][:6]}...)": d["id"] for d in st.session_state.uploaded_documents}

            selected_labels = st.multiselect(
                "Search in documents",
                options=options,
                default=[opt for opt in options if id_map[opt] in st.session_state.selected_doc_ids],
                label_visibility="collapsed"
            )
            st.session_state.selected_doc_ids = [id_map[lbl] for lbl in selected_labels]
            
            if len(st.session_state.selected_doc_ids) == 0:
                st.info("🌐 Searching all documents")
            elif len(st.session_state.selected_doc_ids) == 1:
                st.info("📄 Searching in 1 document")
            else:
                st.info(f"📚 Searching in {len(st.session_state.selected_doc_ids)} documents")
        else:
            st.markdown('<div class="info-msg">📭 No documents uploaded yet</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Session Management
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>🔄</span> Session Management</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                cached_get_json.clear()
                load_available_sessions()
                st.rerun()
        with col2:
            if st.button("➕ New", use_container_width=True):
                create_new_session()
        
        load_available_sessions()
        sessions = st.session_state.available_sessions or []
        sessions = sorted(sessions, key=lambda s: s.get("last_updated", 0), reverse=True)

        def fmt_session(sess_id: str) -> str:
            s = next((x for x in sessions if x.get("session_id") == sess_id), None)
            if not s:
                return sess_id[:6]
            name = s.get("name") or f"Session {sess_id[:6]}"
            name = name.strip()
            if len(name) > 15:
                name = name[:15] + "…"
            cnt = s.get("message_count", 0)
            time_str = format_timestamp(s.get("last_updated"))
            return f"{name} · {cnt} msg"

        session_ids = [s["session_id"] for s in sessions if s.get("session_id")]

        if session_ids:
            current_index = session_ids.index(st.session_state.session_id) if st.session_state.session_id in session_ids else 0
            
            selected_id = st.selectbox(
                "Switch Session",
                options=session_ids,
                index=current_index,
                format_func=fmt_session,
                label_visibility="collapsed",
                key="session_switcher"
            )

            if st.button("📂 Load Session", use_container_width=True, type="primary"):
                s = next((x for x in sessions if x["session_id"] == selected_id), None)
                load_session(selected_id, s.get("name") if s else None)
                st.rerun()
        else:
            st.markdown('<div class="info-msg">No sessions available</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # System Info
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span>📊</span> System Info</div>', unsafe_allow_html=True)
        
        metrics = cached_get_json(f"{API_BASE_URL}/metrics")
        if metrics:
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{metrics.get('chunks_indexed', 0)}</div>
                    <div class="metric-label">Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{metrics.get('active_sessions', 0)}</div>
                    <div class="metric-label">Sessions</div>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{metrics.get('total_interactions', 0)}</div>
                    <div class="metric-label">Interactions</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-item">
                    <div class="metric-value">{metrics.get('cache_size', 0)}</div>
                    <div class="metric-label">Cache</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-msg">📡 Metrics unavailable</div>', unsafe_allow_html=True)
        
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close sidebar-content

        # Bottom Navigation
        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        
        pages = ["Chat", "Documents", "Analytics", "Settings"]
        icons = ["💬", "📄", "📊", "⚙️"]
        
        for page, icon in zip(pages, icons):
            active_class = " active" if st.session_state.current_page == page else ""
            st.markdown(f"""
            <div class="nav-button{active_class}" onclick="document.getElementById('nav_{page}').click();">
                <span>{icon}</span>
                <span>{page}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Hidden button for actual navigation
            if st.button(page, key=f"nav_btn_{page}", help=f"Go to {page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Content Area based on selected page
    if st.session_state.current_page == "Chat":
        # Clean chat interface
        st.markdown("""
        <div class="chat-title">
            <h1>💬 Chat Interface</h1>
            <p>Ask questions about your documents and get intelligent answers</p>
        </div>
        """, unsafe_allow_html=True)

        # Conversation history
        for i, exchange in enumerate(st.session_state.conversation_history):
            user_text = html.escape(exchange.get("query", ""))
            ans_text = html.escape(exchange.get("answer", ""))
            
            st.markdown(f'<div class="message-user">{user_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="message-assistant">{ans_text}</div>', unsafe_allow_html=True)

            # Metadata row
            col1, col2, col3, col4 = st.columns([2, 1, 1, 4])
            
            with col1:
                conf = exchange.get("confidence")
                if conf is not None:
                    st.progress(conf, text=f"Confidence: {conf:.0%}")

            interaction_id = exchange.get("interaction_id")
            
            with col2:
                button_disabled = not interaction_id or exchange.get('feedback_given') is not None
                if st.button("👍", key=f"thumb_up_{i}", disabled=button_disabled):
                    if submit_feedback(interaction_id, 5, i):
                        st.rerun()

            with col3:
                if st.button("👎", key=f"thumb_down_{i}", disabled=button_disabled):
                    if submit_feedback(interaction_id, 1, i):
                        st.rerun()

            # Sources
            sources = exchange.get("sources") or []
            if sources:
                with st.expander("📚 View Source Documents"):
                    for idx, s in enumerate(sources):
                        doc_name = "Unknown"
                        doc_id = s.get('doc_id', '')
                        for doc in st.session_state.uploaded_documents:
                            if doc["id"] == doc_id:
                                doc_name = doc["name"]
                                break
                        
                        st.markdown(f"**Source {idx+1}:** {doc_name}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"📄 Page: {s.get('page_num', 'N/A')}")
                        with col2:
                            st.caption(f"📑 Chunk: {s.get('chunk_num', 'N/A')}")
                        with col3:
                            if s.get("score") is not None:
                                st.caption(f"⭐ Score: {s['score']:.3f}")
                        if idx < len(sources) - 1:
                            st.divider()

        # Chat input
        user_query = st.chat_input("Ask a question about your documents...")

        if user_query:
            # Add to history
            st.session_state.conversation_history.append({
                "query": user_query,
                "answer": "",
                "timestamp": datetime.now().isoformat(),
                "status": "pending"
            })
            
            # Build document filter
            doc_filters = {}
            if st.session_state.selected_doc_ids:
                if len(st.session_state.selected_doc_ids) == 1:
                    doc_filters = {"doc_id": st.session_state.selected_doc_ids[0]}
                elif len(st.session_state.selected_doc_ids) > 1:
                    doc_filters = {"doc_id": st.session_state.selected_doc_ids}

            # Get response
            with st.spinner("Thinking..."):
                resp = safe_post(
                    f"{API_BASE_URL}/query",
                    json={
                        "query": user_query,
                        "session_id": st.session_state.session_id,
                        "document_filters": doc_filters,
                    },
                )

                if resp and resp.status_code == 200:
                    result = resp.json()
                    
                    st.session_state.conversation_history[-1].update({
                        "answer": result.get("answer", ""),
                        "confidence": result.get("confidence"),
                        "sources": result.get("sources"),
                        "interaction_id": result.get("interaction_id"),
                        "status": "completed"
                    })
                    
                    st.markdown('<div class="success-msg">✅ Response received!</div>', unsafe_allow_html=True)
                else:
                    st.session_state.conversation_history[-1].update({
                        "answer": "❌ Sorry, I encountered an error. Please try again.",
                        "status": "error"
                    })
                    st.markdown('<div class="error-msg">❌ Error getting response</div>', unsafe_allow_html=True)
            
            st.rerun()

    elif st.session_state.current_page == "Documents":
        st.markdown("## 📚 Document Library")
        
        if st.session_state.uploaded_documents:
            cols = st.columns(3)
            for idx, doc in enumerate(st.session_state.uploaded_documents):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="card" style="text-align: center;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
                        <h4 style="margin: 0.5rem 0;">{doc['name'][:20]}...</h4>
                        <p style="color: var(--text-secondary);">{doc['chunks']} chunks</p>
                        <p style="color: var(--accent-primary); font-size: 0.8rem;">ID: {doc['id'][:8]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-msg">📭 No documents uploaded yet. Upload some documents to get started!</div>', unsafe_allow_html=True)

    elif st.session_state.current_page == "Analytics":
        st.markdown("## 📊 Analytics Dashboard")
        
        metrics = cached_get_json(f"{API_BASE_URL}/metrics")
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📚</div>
                    <div style="font-size: 2rem; font-weight: 700;">{metrics.get('chunks_indexed', 0)}</div>
                    <div style="color: var(--text-secondary);">Chunks Indexed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
                    <div style="font-size: 2rem; font-weight: 700;">{metrics.get('total_interactions', 0)}</div>
                    <div style="color: var(--text-secondary);">Interactions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">👥</div>
                    <div style="font-size: 2rem; font-weight: 700;">{metrics.get('active_sessions', 0)}</div>
                    <div style="color: var(--text-secondary);">Active Sessions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <div style="font-size: 2rem; font-weight: 700;">{metrics.get('cache_size', 0)}</div>
                    <div style="color: var(--text-secondary);">Cache Size</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Activity timeline
        if st.session_state.conversation_history:
            st.markdown("### 📈 Activity Timeline")
            
            activity = []
            for ex in st.session_state.conversation_history[-20:]:
                ts = ex.get("timestamp")
                if not ts:
                    continue
                try:
                    t = datetime.fromisoformat(ts)
                except:
                    continue

                activity.append({"time": t, "type": "Question", "value": 1})
                if ex.get("answer"):
                    activity.append({"time": t, "type": "Answer", "value": 1})

            if activity:
                df = pd.DataFrame(activity)
                fig = go.Figure()

                q_df = df[df["type"] == "Question"]
                if not q_df.empty:
                    fig.add_trace(go.Scatter(
                        x=q_df["time"],
                        y=[1] * len(q_df),
                        mode="markers",
                        marker=dict(size=12, color="#6366F1", line=dict(color="white", width=1)),
                        name="Questions"
                    ))

                a_df = df[df["type"] == "Answer"]
                if not a_df.empty:
                    fig.add_trace(go.Scatter(
                        x=a_df["time"],
                        y=[1] * len(a_df),
                        mode="markers",
                        marker=dict(size=12, color="#22D3EE", line=dict(color="white", width=1)),
                        name="Answers"
                    ))

                fig.update_layout(
                    title="Recent Activity",
                    xaxis_title="Time",
                    showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F1F5FF"),
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="#1F2A44",
                        tickfont=dict(color="#9BA8C7")
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=False,
                        range=[0.5, 1.5]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.current_page == "Settings":
        st.markdown("## ⚙️ Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎨 Appearance")
            theme = st.selectbox("Theme", ["Dark", "Light", "System"], index=0)
            
            st.markdown("#### 🔔 Notifications")
            enable_notifications = st.checkbox("Enable Notifications", value=True)
        
        with col2:
            st.markdown("#### 🧠 AI Settings")
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.1)
            
            st.markdown("#### 🔒 Privacy")
            save_history = st.checkbox("Save Conversation History", value=True)
        
        if st.button("💾 Save Settings", use_container_width=True):
            st.markdown('<div class="success-msg">✅ Settings saved successfully!</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()