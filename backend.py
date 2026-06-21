# =============================================================================
# backend.py (Fixed Cloud-Ready Version)
# =============================================================================
import os
import re
import psycopg2
import requests
from typing import Annotated, TypedDict

# ── LangChain / LangGraph ──────────────────────────────────────────────────
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Web server ────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from pinecone import Pinecone



import tempfile


# OPTION A: Save it completely outside the workspace in the system temp directory
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "chatbot_uploaded_docs")

# OPTION B: Or just move it one level above your current project root directory
# UPLOAD_FOLDER = "../uploaded_docs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# =============================================================================
# CONFIG & CREDENTIALS (Sourced from Environment Variables)
# =============================================================================
# =============================================================================
# CONFIG & CREDENTIALS (Strictly using environment variables for safety)
# =============================================================================
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

# Safeguard check to ensure the server crashes gracefully if variables are missing
if not all([os.environ["GROQ_API_KEY"], HF_TOKEN, DATABASE_URL, PINECONE_API_KEY]):
    print("❌ WARNING: One or more environment variables are missing! Check your cloud dashboard.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_INDEX = "chatbot-docs"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


# =============================================================================
# FREE API EMBEDDINGS (Safely Flattened for Pinecone)
# =============================================================================
# =============================================================================
# OPTIMIZED BATCH EMBEDDINGS (Processes entire document lists in 1 call)
# =============================================================================
def get_cloud_embeddings_batch(texts: list) -> list:
    """Sends an array of text chunks to Hugging Face in a single optimized HTTP call."""
    api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    response = requests.post(api_url, json={"inputs": texts, "options": {"wait_for_model": True}}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Hugging Face API Error: {response.text}")
        
    raw_embeddings = response.json()
    
    # Standardize output shapes (stripping potential nested token dimension tensors)
    cleaned_embeddings = []
    for item in raw_embeddings:
        while isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
            item = item[0]
        cleaned_embeddings.append(item)
        
    return cleaned_embeddings


# =============================================================================
# OPTIMIZED ASK DOC TOOL
# =============================================================================
@tool
def ask_doc(input_text: str) -> str:
    """Use this tool whenever the user references an uploaded document or asks about their PDFs (for example: summarize, extract key points, find specific facts, list headings, or return page excerpts). If given a file path, index the document: load, split into chunks, embed, and upsert to the vector store, then return a confirmation with the number of chunks indexed. If given a natural-language query, embed the query, perform a vector search, and return the most relevant excerpt (trimmed to a reasonable length) or 'Nothing found in document."""
    if not input_text or not input_text.strip():
        return "No input."
    text = input_text.strip()

    # Processing a file indexing sequence
    if os.path.isfile(text):
        loader = PyPDFLoader(text) if text.lower().endswith(".pdf") else TextLoader(text)
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        
        # Extract text content strings into a clean list
        texts_to_embed = [chunk.page_content for chunk in chunks]
        
        # Request all embeddings in a single swift roundtrip
        embeddings = get_cloud_embeddings_batch(texts_to_embed)
        
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vectors_to_upsert.append({
                "id": f"chunk_{i}_{os.path.basename(text)}",
                "values": embeddings[i],
                "metadata": {"text": chunk.page_content}
            })
            
        index.upsert(vectors=vectors_to_upsert)
        return f"Successfully indexed {len(chunks)} chunks to Pinecone cloud!"

    # Fallback to semantic querying (Single string operation)
    api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(api_url, json={"inputs": [text], "options": {"wait_for_model": True}}, headers=headers)
    
    query_vector = response.json()[0]
    while isinstance(query_vector, list) and len(query_vector) > 0 and isinstance(query_vector[0], list):
        query_vector = query_vector[0]
        
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    
    if results and results.get("matches"):
        return results["matches"][0]["metadata"]["text"][:600]
    return "Nothing found in document."

# =============================================================================
# TOOLS
# =============================================================================
@tool
def load_tasks() -> str:
    """Read user preferences from the cloud PostgreSQL instance.Befor answering any question, use this tool to know user details."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT preferences FROM user_prefs WHERE username = 'Mohan' LIMIT 1;")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else "No preferences saved."
    except Exception as e:
        return f"Error connecting to cloud DB: {str(e)}"

# =============================================================================
# NEON POSTGRESQL WRITE LOGIC (UPSERT)
# =============================================================================
def _write_prefs(text: str) -> None:
    """Writes or overwrites user preferences in Neon using PostgreSQL UPSERT syntax."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn:
            with conn.cursor() as cur:
                # 'ON CONFLICT' targets the PRIMARY KEY (username) and updates the text if it exists
                cur.execute("""
                    INSERT INTO user_prefs (username, preferences) 
                    VALUES ('Mohan', %s) 
                    ON CONFLICT (username) 
                    DO UPDATE SET preferences = EXCLUDED.preferences;
                """, (text,))
        conn.close()
    except Exception as e:
        print(f"[Database Error] Failed writing preferences to Neon: {str(e)}")


# =============================================================================
# UPDATED USER PREFERENCE TOOL
# =============================================================================
@tool
def update_task(preferences: str) -> str:
    """
    Update the long-term preferences of the user in the cloud database.
    CRITICAL: Only call this tool IF you have already identified important user details.
    """
    if not preferences.strip():
        return "Nothing to save."
    
    _write_prefs(preferences.strip())
    return "Preferences successfully updated in cloud database."


HISTORY_PAIRS = 1
BANNED_KEYWORDS = ["hack", "exploit", "malware", "jailbreak", "bypass"]
PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[email]"),
    (re.compile(r'\b(\+91[\-\s]?)?[6-9]\d{9}\b'), "[phone]"),
    (re.compile(r'\b\d{12}\b'), "[id-number]"),
]

SYSTEM_PROMPT = (
    "Be a helpful agent. Don't tell users which tools you use, just do the task. "
    "Before answering any question, use 'load_tasks' tool for knowing user details and address user by their name. "
    "use 'ask_doc' tool if user gives a file path or for any document questions like summarizing and extracting information or any kind of question about their stored pdf files ."
    "And after thinking answer if you found any important thing in the conversation like their likes and dislikes use update_task tool to modify their preferences to new one ."
)

app = Flask(__name__)
CORS(app)

# =============================================================================
# GUARDRAILS
# =============================================================================
def guardrail_content_filter(text: str) -> str | None:
    lower = text.lower()
    for kw in BANNED_KEYWORDS:
        if kw in lower:
            return "⚠️ I can't help with that request. It contains content that isn't allowed."
    return None

def guardrail_pii_redact(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def guardrail_safety_check(response_text: str) -> str:
    
    # Avoid expensive per-response API calls - just do lightweight checks
    unsafe_phrases = ["execute code", "system command", "delete", "hack"]
    lower_response = response_text.lower()
    for phrase in unsafe_phrases:
        if phrase in lower_response:
            return "I'm not able to provide that response. Please rephrase your question."
    return response_text

# =============================================================================
# FIXED AGENT INITIALIZATION
# =============================================================================
groq_model = ChatGroq(model="llama-3.3-70b-versatile")
llm = create_agent(
    model=groq_model,
    tools=[load_tasks, ask_doc,update_task],
    
)

conversation_history: list = []
conversation_history.append(SystemMessage(content=SYSTEM_PROMPT))
def _trim(messages: list) -> list:
    if not messages:
        return messages
    return messages[-(HISTORY_PAIRS * 2):]

def run_with_guardrails(user_text: str) -> str:
    global conversation_history
    blocked = guardrail_content_filter(user_text)
    if blocked: return blocked

    clean_text = guardrail_pii_redact(user_text)
    conversation_history.append(HumanMessage(content=clean_text))
    trimmed = _trim(conversation_history)
    print("Trimmed conversation history:", trimmed)
    response = llm.invoke({"messages": trimmed})
    agent_reply = response["messages"][-1].content or ""
    print("Agent reply:", response)
    safe_reply = guardrail_safety_check(agent_reply)
    conversation_history.append(AIMessage(content=safe_reply))
    print("Safe reply:", conversation_history)
    return safe_reply

# =============================================================================
# ROUTES
# =============================================================================
@app.route('/')
def home_page():
    """Serves the frontend interface directly to the user's browser."""
    return render_template('frontend.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Explicitly bypasses conversational ambiguity and indexes directly via the tool pipeline."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    safe_name = os.path.basename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    f.save(save_path)
    print('hi')
    try:
        # Directly call the indexing logic deterministically
        reply = ask_doc.invoke(save_path)
        conversation_history.append(HumanMessage(content=f"Uploaded document: {safe_name}"))
        conversation_history.append(AIMessage(content=reply))
        print(reply)
        return jsonify({"reply": reply, "filename": save_path})
    except Exception as e:
        return jsonify({"error": f"Failed indexing document array: {str(e)}"}), 500

@app.route('/add', methods=['POST'])
def add():
    data = request.json or {}
    msg = str(data.get("message", "")).strip()
    if not msg:
        return jsonify({"error": "No message."}), 400
    print(msg)
    reply = run_with_guardrails(msg)
    return jsonify({"reply": reply})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Add use_reloader=False to stop OneDrive from triggering endless restarts
    app.run(
        debug=True, 
        threaded=True, 
        use_reloader=False, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 5000))
    )