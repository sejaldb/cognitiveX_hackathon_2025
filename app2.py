import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

model="ibm-granite/granite-4.0-h-1b"

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language configurations
LANGUAGES = {
    'en': {'name': '🇬🇧 English', 'code': 'en'},
    'es': {'name': '🇪🇸 Spanish', 'code': 'es'},
    'fr': {'name': '🇫🇷 French', 'code': 'fr'},
    'de': {'name': '🇩🇪 German', 'code': 'de'},
    'hi': {'name': '🇮🇳 Hindi', 'code': 'hi'},
    'zh': {'name': '🇨🇳 Chinese', 'code': 'zh'},
    'ar': {'name': '🇸🇦 Arabic', 'code': 'ar'},
    'pt': {'name': '🇵🇹 Portuguese', 'code': 'pt'},
    'ja': {'name': '🇯🇵 Japanese', 'code': 'ja'},
    'ru': {'name': '🇷🇺 Russian', 'code': 'ru'}
}

# Custom CSS for clean theme
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .upload-section {
        background-color: #fafafa;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cccccc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #fff8e1;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    h1 {
        color: #1976D2;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""

# Extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

# Preprocess text
def preprocess_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    return text.strip()

# Split text into chunks
def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Create embeddings and FAISS index
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(text_chunks: List[str], model) -> Tuple[faiss.Index, np.ndarray]:
    """Create FAISS index from text chunks"""
    with st.spinner("🧠 Creating semantic embeddings..."):
        embeddings = model.encode(text_chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
    return index, embeddings

# Search for relevant chunks
def semantic_search(query: str, index, text_chunks: List[str], model, k: int = 3) -> List[str]:
    """Perform semantic search using FAISS"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

# Generate summary of document
def generate_summary(text: str, language: str = 'en') -> str:
    """Generate a summary of the document in the selected language"""
    # Take first 3000 characters for summary
    text_sample = text[:3000] if len(text) > 3000 else text
    
    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'pt': 'Portuguese',
        'ja': 'Japanese',
        'ru': 'Russian'
    }
    
    prompt = f"""Please provide a comprehensive summary of this document in {language_names.get(language, 'English')}. 
Include:
- Main topics covered
- Key concepts and themes
- Important points or findings
- Overall structure of the document

Document excerpt:
{text_sample}

Please provide the summary in {language_names.get(language, 'English')}."""
    
    try:
        import requests
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.ok:
            data = response.json()
            summary = data['content'][0]['text']
            return summary
        else:
            return f"Document contains approximately {len(text.split())} words across {len(text_sample)} characters. The content discusses various academic topics."
    except:
        return f"Document contains approximately {len(text.split())} words. Summary generation is temporarily unavailable."

# Generate answer using LLM
def generate_answer(query: str, context: List[str], language: str = 'en') -> str:
    """Generate answer based on context in the selected language"""
    context_text = "\n\n".join(context)
    
    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'pt': 'Portuguese',
        'ja': 'Japanese',
        'ru': 'Russian'
    }
    
    prompt = f"""Based on the following context from the academic document, please answer the question accurately and concisely in {language_names.get(language, 'English')}.

Context:
{context_text}

Question: {query}

Please provide your answer in {language_names.get(language, 'English')}."""
    
    try:
        import requests
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.ok:
            data = response.json()
            answer = data['content'][0]['text']
            return answer
        else:
            return f"Based on the document: {context[0][:300]}..."
    except:
        return f"Based on the provided context: {context[0][:400]}..."

# Main app
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📚 StudyMate")
        st.markdown("*Your AI-Powered Multi-Language Academic Assistant*")
    
    # Sidebar
    with st.sidebar:
        st.header("🌐 Language Selection")
        selected_lang = st.selectbox(
            "Choose your language:",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x]['name'],
            index=list(LANGUAGES.keys()).index(st.session_state.language)
        )
        
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            # Regenerate summary in new language if document is loaded
            if st.session_state.document_processed and st.session_state.full_text:
                with st.spinner("🔄 Translating summary..."):
                    st.session_state.document_summary = generate_summary(
                        st.session_state.full_text,
                        st.session_state.language
                    )
                st.rerun()
        
        st.markdown("---")
        st.header("📖 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your study material (PDF)",
            type=['pdf'],
            help="Upload textbooks, lecture notes, or research papers"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("📄 Extracting text from PDF..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                
                if raw_text:
                    st.session_state.full_text = raw_text
                    
                    with st.spinner("🧹 Preprocessing text..."):
                        cleaned_text = preprocess_text(raw_text)
                        st.session_state.text_chunks = split_into_chunks(cleaned_text)
                    
                    # Load model and create index
                    st.session_state.model = load_embedding_model()
                    st.session_state.index, st.session_state.embeddings = create_faiss_index(
                        st.session_state.text_chunks,
                        st.session_state.model
                    )
                    
                    # Generate summary
                    with st.spinner("📝 Generating document summary..."):
                        st.session_state.document_summary = generate_summary(
                            cleaned_text,
                            st.session_state.language
                        )
                    
                    st.session_state.document_processed = True
                    st.success("✅ Document processed successfully!")
                    st.markdown(f"""
                    <div class='success-box'>
                        <strong>Document Stats:</strong><br>
                        📊 Total chunks: {len(st.session_state.text_chunks)}<br>
                        📝 Characters: {len(cleaned_text):,}<br>
                        🌐 Language: {LANGUAGES[st.session_state.language]['name']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        
        if st.session_state.document_processed:
            st.markdown("---")
            st.markdown("### 🎯 Features Active")
            st.markdown("✓ Semantic Search")
            st.markdown("✓ Context-Aware Q&A")
            st.markdown("✓ FAISS Indexing")
            st.markdown("✓ Multi-Language Support")
            st.markdown("✓ Auto-Summary")
            
            if st.button("🗑️ Clear Document"):
                st.session_state.document_processed = False
                st.session_state.text_chunks = []
                st.session_state.embeddings = None
                st.session_state.index = None
                st.session_state.messages = []
                st.session_state.document_summary = None
                st.session_state.full_text = ""
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. Select your preferred language
        2. Upload a PDF document
        3. Click 'Process Document'
        4. Read the auto-generated summary
        5. Ask questions in any language!
        """)
    
    # Main content area
    if not st.session_state.document_processed:
        st.markdown("""
        <div class='upload-section'>
            <h3>👋 Welcome to StudyMate!</h3>
            <p>Upload your academic materials to get started with AI-powered learning assistance.</p>
            <p><strong>Supported formats:</strong> PDF (textbooks, notes, papers)</p>
            <p><strong>Multi-language support:</strong> Get answers in 10+ languages!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🔍 Semantic Search
            Advanced FAISS-based search to find relevant content quickly
            """)
        with col2:
            st.markdown("""
            ### 🤖 AI Answers
            Get intelligent, context-aware responses in your language
            """)
        with col3:
            st.markdown("""
            ### 📊 Auto-Summary
            Automatic document summarization upon upload
            """)
    else:
        # Display document summary
        if st.session_state.document_summary:
            st.markdown("### 📋 Document Summary")
            st.markdown(f"""
            <div class='summary-box'>
                {st.session_state.document_summary}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
        
        # Chat interface
        st.markdown("### 💬 Ask Questions About Your Material")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class='chat-message user-message'>
                        <strong>🙋 You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-message assistant-message'>
                        <strong>🤖 StudyMate:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask a question:",
                placeholder="e.g., What are the main concepts in Chapter 3?",
                key="user_input",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.button("Send 📤", type="primary")
        
        # Process query
        if send_button and user_input:
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input
            })
            
            # Perform semantic search
            with st.spinner("🔍 Searching document..."):
                relevant_chunks = semantic_search(
                    user_input,
                    st.session_state.index,
                    st.session_state.text_chunks,
                    st.session_state.model,
                    k=3
                )
            
            # Generate answer in selected language
            with st.spinner("💭 Generating answer..."):
                answer = generate_answer(
                    user_input,
                    relevant_chunks,
                    st.session_state.language
                )
            
            # Add assistant message
            st.session_state.messages.append({
                'role': 'assistant',
                'content': answer
            })
            
            st.rerun()

if __name__ == "__main__":

    main()
