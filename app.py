import os
import datetime
import streamlit as st
from dotenv import load_dotenv

from document_loader import load_file_content, LoadedDocument
from rag_engine import RAGEngine
from llm_provider import PROVIDERS_CONFIG, stream_llm_answer

# Load .env file if present
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Doc Q&A Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MASSIVE CUSTOM CSS — Pastel Glassmorphism Chat UI
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ===== Global Reset & Background ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(145deg, #f5e6f0 0%, #fce4ec 20%, #fff3e0 40%, #e8eaf6 60%, #f3e5f5 80%, #fce4ec 100%) !important;
        background-attachment: fixed !important;
    }

    /* ===== Hide Streamlit defaults ===== */
    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Remove default padding */
    .block-container {
        padding-top: 2.8rem !important;
        padding-bottom: 250px !important; /* Extra space so last message isn't hidden */
        max-width: 900px !important;
    }

    /* ===== Sidebar Styling ===== */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.4) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #4a4a6a !important;
    }

    /* ===== Chat Message Styling ===== */
    /* User bubble — pink, right aligned */
    .user-bubble {
        background: linear-gradient(135deg, #ec407a 0%, #e91e90 100%);
        color: #ffffff;
        padding: 12px 20px;
        border-radius: 20px 20px 4px 20px;
        max-width: 75%;
        margin-left: auto;
        margin-right: 0;
        font-size: 0.92rem;
        font-weight: 400;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.2);
        word-wrap: break-word;
    }

    .user-bubble p { color: #ffffff !important; margin: 0 !important; }

    .user-meta {
        text-align: right;
        color: #b0a0b8;
        font-size: 0.72rem;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    /* AI bubble — white glassmorphic, left aligned */
    .ai-bubble-wrapper {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        max-width: 80%;
        margin-bottom: 4px;
    }

    .ai-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #e1bee7, #b3e5fc, #f8bbd0, #c5e1a5);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .ai-bubble {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        padding: 14px 20px;
        border-radius: 4px 20px 20px 20px;
        font-size: 0.92rem;
        font-weight: 400;
        line-height: 1.6;
        color: #2d2d44;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        word-wrap: break-word;
    }

    .ai-bubble p { margin: 0 0 8px 0 !important; color: #2d2d44 !important; }
    .ai-bubble p:last-child { margin-bottom: 0 !important; }
    .ai-bubble ul, .ai-bubble ol { margin: 4px 0; padding-left: 20px; }
    .ai-bubble code { background: rgba(0,0,0,0.06); padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
    .ai-bubble pre { background: rgba(0,0,0,0.05); padding: 12px; border-radius: 8px; overflow-x: auto; }

    .ai-meta {
        color: #b0a0b8;
        font-size: 0.72rem;
        margin-top: 4px;
        margin-bottom: 18px;
        margin-left: 46px;
    }

    /* ===== Model Badge ===== */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(255,255,255,0.5);
        color: #7c6f8a;
        border: 1px solid rgba(255,255,255,0.6);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-top: 2px;
        backdrop-filter: blur(6px);
    }

    /* ===== Source Card ===== */
    .source-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        border-left: 3px solid #ec407a;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.84rem;
        color: #4a4a6a;
    }



    /* Style bordered containers to look glassmorphic */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1.5px solid rgba(255, 200, 220, 0.5) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04) !important;
    }

    /* Streamlit bottom container overrides */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* The actual input card inside bottom (Mobile First) */
    [data-testid="stBottom"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1.5px solid rgba(233, 30, 99, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
        width: 95% !important; /* Wide on mobile */
        max-width: 100% !important;
        margin: 0 auto !important; /* Centers it horizontally */
    }

    /* Narrower input card on Desktop */
    @media (min-width: 768px) {
        [data-testid="stBottom"] div[data-testid="stVerticalBlockBorderWrapper"] {
            width: 75% !important;
            max-width: 860px !important;
        }
    }


    /* Style file uploader area */
    [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.5) !important;
        background-color: rgba(255, 255, 255, 0.5) !important;
        border-radius: 12px !important;
        border: 1.5px dashed rgba(233, 30, 99, 0.4) !important;
        padding: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #2d2d44 !important;
    }

    /* Universally fix input fields colors to prevent dark mode corruption */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.7) !important;
        background-color: rgba(255, 255, 255, 0.7) !important;
        color: #2d2d44 !important;
        -webkit-text-fill-color: #2d2d44 !important; /* Forces text color on iOS/Webkit */
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Ensure selectbox selected text is visible */
    [data-testid="stSelectbox"] span, [data-testid="stSelectbox"] div {
        color: #2d2d44 !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(233, 30, 99, 0.3) !important;
        box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.1) !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #b0a0b8 !important;
    }

    /* Style selectboxes */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
    }

    /* Style text input (API key) */
    [data-testid="stTextInput"] input {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
    }

    /* Send/Ask button - Pink styled */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ec407a, #e91e90) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Secondary buttons */
    .stButton > button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 10px !important;
        color: #5a5a7a !important;
        font-size: 0.82rem !important;
        backdrop-filter: blur(6px) !important;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        padding: 8px 12px;
        text-align: center;
    }

    /* Divider */
    hr {
        border-color: rgba(0, 0, 0, 0.04) !important;
    }

    /* Header area */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 8px 0 4px 0;
    }

    .app-logo {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, #e1bee7, #b3e5fc, #f8bbd0);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .app-title {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ad1457 0%, #7b1fa2 50%, #283593 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .app-subtitle {
        color: #9a8ca8;
        font-size: 0.82rem;
        text-align: center;
        margin-bottom: 16px;
        font-weight: 400;
    }

    /* Hide default chat message containers since we use custom HTML */
    [data-testid="stChatMessage"] {
        display: none !important;
    }

    /* Thinking/analyzing indicator */
    .analyzing-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #7c6f8a;
        font-size: 0.88rem;
        margin-left: 46px;
        margin-bottom: 16px;
    }

    /* Toast/notification styling */
    [data-testid="stToast"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
    }

    /* Pill tags for controls */
    .controls-row {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Initialize Session State
# ==========================================
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "loaded_files_map" not in st.session_state:
    st.session_state.loaded_files_map = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file_signatures" not in st.session_state:
    st.session_state.processed_file_signatures = set()


# ==========================================
# SIDEBAR — Minimal Knowledge Base & Settings
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.divider()

    st.markdown("#### 📊 Knowledge Base")
    stats = st.session_state.rag_engine.get_summary_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Docs", stats["total_docs"])
    with col2:
        st.metric("🧩 Chunks", stats["total_chunks"])

    st.caption(f"**{stats['total_chars']:,}** characters indexed")

    if stats["documents"]:
        with st.expander("📑 Document Inspector", expanded=False):
            for doc_info in stats["documents"]:
                st.markdown(f"**{doc_info['name']}**")
                st.caption(f"`{doc_info['type']}` · {doc_info['sections']} sections · {doc_info['chars']:,} chars")
                doc_obj = st.session_state.loaded_files_map.get(doc_info['name'])
                if doc_obj:
                    preview_text = doc_obj.full_text[:300] + ("..." if len(doc_obj.full_text) > 300 else "")
                    st.text_area(f"Preview: {doc_info['name']}", preview_text, height=80, disabled=True)
                st.divider()

    st.divider()

    with st.expander("🛠️ Advanced", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="0.0 = most strict and factual."
        )
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=400, max_value=3000, value=1200, step=100,
            help="Text chunk size for indexing."
        )

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🗑️ Reset All", use_container_width=True):
        st.session_state.loaded_files_map = {}
        st.session_state.rag_engine = RAGEngine()
        st.session_state.processed_file_signatures = set()
        st.session_state.chat_history = []
        st.rerun()


# ==========================================
# MAIN — Header
# ==========================================
st.markdown("""
    <div class="app-header">
        <div class="app-logo">✨</div>
        <div class="app-title">Doc Q&A Agent</div>
    </div>
    <div class="app-subtitle">Upload documents and ask questions — answers sourced strictly from your files.</div>
""", unsafe_allow_html=True)


# ==========================================
# File Upload (compact)
# ==========================================
uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=None,
    label_visibility="collapsed",
    help="Drag & drop PDF, Word, Excel, CSV, Text, Code files..."
)

if uploaded_files:
    # Build current file signatures from the uploader
    current_sigs = {f"{file.name}_{file.size}": file for file in uploaded_files}
    current_names = {file.name for file in uploaded_files}

    # 1. Detect and process NEW files
    new_files_processed = False
    for file in uploaded_files:
        file_sig = f"{file.name}_{file.size}"
        if file_sig not in st.session_state.processed_file_signatures:
            with st.spinner(f"Parsing `{file.name}`..."):
                file_bytes = file.getvalue()
                loaded_doc = load_file_content(file_bytes, file.name)
                st.session_state.loaded_files_map[file.name] = loaded_doc
                st.session_state.processed_file_signatures.add(file_sig)
                new_files_processed = True

    # 2. Detect and handle REMOVED files (user deleted from uploader)
    removed_names = set(st.session_state.loaded_files_map.keys()) - current_names
    files_removed = False
    if removed_names:
        for name in removed_names:
            del st.session_state.loaded_files_map[name]
        # Clean up stale signatures
        st.session_state.processed_file_signatures = {
            sig for sig in st.session_state.processed_file_signatures
            if sig.rsplit("_", 1)[0] in current_names
        }
        files_removed = True
        st.toast(f"🗑️ Removed {len(removed_names)} file(s). Index updated.", icon="♻️")

    # 3. Rebuild RAG index if anything changed
    if new_files_processed or files_removed:
        if st.session_state.loaded_files_map:
            st.session_state.rag_engine.add_documents(list(st.session_state.loaded_files_map.values()))
        else:
            st.session_state.rag_engine = RAGEngine()
        if new_files_processed:
            st.toast(f"✅ Indexed {len(st.session_state.loaded_files_map)} documents!", icon="🎉")

else:
    # All files removed from uploader
    if st.session_state.loaded_files_map:
        st.session_state.loaded_files_map = {}
        st.session_state.processed_file_signatures = set()
        st.session_state.rag_engine = RAGEngine()
        st.toast("🗑️ All documents removed. Index cleared.", icon="♻️")


# ==========================================
# Render Chat History (Custom HTML bubbles)
# ==========================================
def get_timestamp():
    return datetime.datetime.now().strftime("%I:%M %p")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        badge_html = ""
        if message.get("provider") and message.get("model"):
            badge_html = f'<div style="text-align:right;"><span class="model-badge">⚡ {message["provider"]} · {message["model"]}</span></div>'
        ts = message.get("timestamp", "")
        st.markdown(
            f"""<div class="user-bubble">{message["content"]}</div>
                {badge_html}
                <div class="user-meta">{ts}</div>""",
            unsafe_allow_html=True
        )
    else:
        ts = message.get("timestamp", "")
        st.markdown(
            f"""<div class="ai-bubble-wrapper">
                    <div class="ai-avatar">🤖</div>
                    <div class="ai-bubble">{message["content"]}</div>
                </div>
                <div class="ai-meta">{ts}</div>""",
            unsafe_allow_html=True
        )
        # Sources removed by user request




# ==========================================
# INPUT CARD — Glassmorphic with gradient border
# ==========================================
with st.bottom:
    input_card = st.container(border=True)

    # Handle pending query from quick suggestions
    if "pending_query" in st.session_state and st.session_state.pending_query:
        default_query = st.session_state.pending_query
    else:
        default_query = ""

    with input_card:
        user_query_text = st.text_area(
            "Question",
            value=default_query,
            placeholder="Ask me anything...",
            height=72,
            label_visibility="collapsed",
            key="query_input"
        )

    with input_card:
        # Controls row: Provider, Model, API Key, Send
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 2.5, 0.8])

        with ctrl_col1:
            provider = st.selectbox(
                "Provider",
                options=list(PROVIDERS_CONFIG.keys()),
                index=0,
                key="user_card_provider",
                label_visibility="collapsed"
            )

        prov_info = PROVIDERS_CONFIG[provider]

        with ctrl_col2:
            model_name = st.selectbox(
                "Model",
                options=prov_info["models"],
                index=0,
                key=f"user_card_model_{provider}",
                label_visibility="collapsed"
            )

        with ctrl_col3:
            env_key = os.getenv(prov_info["env_var"], "")
            api_key_input = st.text_input(
                "API Key",
                value=env_key,
                type="password",
                placeholder=f"🔑 {provider} API Key",
                key=f"user_card_key_{provider}",
                label_visibility="collapsed"
            )

        with ctrl_col4:
            submit_btn = st.button("Send", type="primary", use_container_width=True)

# Handle submit: store the query in session state so it survives the rerun
if submit_btn and user_query_text.strip():
    st.session_state.submitted_query = user_query_text.strip()
    st.session_state.submitted_provider = provider
    st.session_state.submitted_model = model_name
    st.session_state.submitted_api_key = api_key_input
    # Clear the pending quick query
    st.session_state.pending_query = ""
    st.rerun()

# Process the submitted query (runs after rerun)
if "submitted_query" in st.session_state and st.session_state.submitted_query:
    active_prompt = st.session_state.submitted_query
    active_provider = st.session_state.get("submitted_provider", provider)
    active_model = st.session_state.get("submitted_model", model_name)
    active_api_key = st.session_state.get("submitted_api_key", api_key_input)

    # Clear submitted query so it doesn't re-process
    del st.session_state.submitted_query

    if not active_api_key or not active_api_key.strip():
        st.error(f"⚠️ Please enter your **{active_provider} API Key** in the input card above.")
    elif not st.session_state.loaded_files_map:
        st.error("⚠️ Please upload at least one document before asking questions.")
    else:
        timestamp = get_timestamp()

        # Render user bubble
        badge_html = f'<div style="text-align:right;"><span class="model-badge">⚡ {active_provider} · {active_model}</span></div>'
        st.markdown(
            f"""<div class="user-bubble">{active_prompt}</div>
                {badge_html}
                <div class="user-meta">{timestamp}</div>""",
            unsafe_allow_html=True
        )

        st.session_state.chat_history.append({
            "role": "user",
            "content": active_prompt,
            "provider": active_provider,
            "model": active_model,
            "timestamp": timestamp
        })

        # Retrieve context
        context_str, cited_sources = st.session_state.rag_engine.format_context(active_prompt)

        # Show analyzing indicator
        analyzing_placeholder = st.empty()
        analyzing_placeholder.markdown(
            """<div class="ai-bubble-wrapper">
                   <div class="ai-avatar">🤖</div>
                   <div class="ai-bubble">✨ Analyzing documents, please wait...</div>
               </div>""",
            unsafe_allow_html=True
        )

        # Stream response
        full_response = ""
        stream_gen = stream_llm_answer(
            provider=active_provider,
            api_key=active_api_key.strip(),
            model_name=active_model,
            query=active_prompt,
            context=context_str,
            chat_history=st.session_state.chat_history,
            temperature=temperature
        )

        for chunk in stream_gen:
            full_response += chunk
            analyzing_placeholder.markdown(
                f"""<div class="ai-bubble-wrapper">
                        <div class="ai-avatar">🤖</div>
                        <div class="ai-bubble">{full_response}▌</div>
                    </div>""",
                unsafe_allow_html=True
            )

        # Final render
        resp_timestamp = get_timestamp()
        analyzing_placeholder.markdown(
            f"""<div class="ai-bubble-wrapper">
                    <div class="ai-avatar">🤖</div>
                    <div class="ai-bubble">{full_response}</div>
                </div>
                <div class="ai-meta">{resp_timestamp}</div>""",
            unsafe_allow_html=True
        )

        # Sources removed by user request

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "sources": cited_sources,
            "timestamp": resp_timestamp
        })

