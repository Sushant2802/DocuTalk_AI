import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()  # Load GOOGLE_API_KEY from .env

import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader as SafePdfReader


# --- Updated Imports for LangChain 1.x ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# --- Flask App Config ---
app = Flask(__name__)
CORS(app)

# --- Globals ---
vector_store = None
llm = None
embeddings = None
text_splitter = None
youtube_api_client = YouTubeTranscriptApi()

# --- Initialize Models ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print("✅ Models initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing models: {e}")


# --- Helper: Translate to English if needed ---
def get_english_transcript(transcript_text):
    if not llm:
        return transcript_text
    try:
        prompt = f"Detect the language of this text (reply with 2-letter code): {transcript_text[:500]}"
        lang_code = llm.invoke(prompt).content.strip().lower()
        if 'en' in lang_code:
            return transcript_text
        translation_prompt = f"Translate this text to English: {transcript_text}"
        return llm.invoke(translation_prompt).content
    except Exception as e:
        print(f"Error translating transcript: {e}")
        return transcript_text


# --- Endpoint: Process Webpage ---
@app.route('/process_webpage', methods=['POST'])
def process_webpage():
    global vector_store
    try:
        data = request.json
        html = data.get('content', '')
        if not html:
            return jsonify({"error": "No content provided"}), 400

        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)

        docs = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(docs, embeddings)

        print("✅ Webpage processed successfully.")
        return jsonify({"status": "ready", "message": f"Processed {len(text)} characters."})
    except Exception as e:
        print(f"Error processing webpage: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint: Process YouTube ---
@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    global vector_store
    try:
        data = request.json
        video_id = data.get('videoId', '')
        if not video_id:
            return jsonify({"error": "No video ID provided"}), 400

        print(f"🎥 Fetching transcript for video ID: {video_id}")
        transcript_list = youtube_api_client.list(video_id)
        transcript = transcript_list.find_transcript(['en', 'hi', 'es', 'de', 'fr'])
        transcript_data = transcript.fetch()

        raw_text = " ".join([t.text for t in transcript_data])
        english_text = get_english_transcript(raw_text)

        docs = text_splitter.split_text(english_text)
        vector_store = FAISS.from_texts(docs, embeddings)

        print("✅ YouTube transcript processed successfully.")
        return jsonify({"status": "ready", "message": f"Processed {len(english_text)} characters."})
    except (NoTranscriptFound, TranscriptsDisabled):
        return jsonify({"error": "Transcript not available for this video."}), 400
    except Exception as e:
        print(f"Error processing YouTube: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint: Process PDF ---
@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global vector_store
    if not llm:
        return jsonify({"error": "Models not initialized."}), 500

    try:
        data = request.json or {}
        pdf_url = data.get('pdf_url', '').strip()

        if not pdf_url:
            return jsonify({"error": "No PDF URL provided"}), 400

        # ✅ Handle Google Drive / file:/// / and invalid URLs gracefully
        if pdf_url.startswith("file://"):
            return jsonify({"error": "Local file paths are not supported. Please use an online PDF URL."}), 400

        print(f"📘 Fetching PDF from: {pdf_url}")
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"⚠️ PDF download failed: {e}")
            return jsonify({"error": f"Failed to fetch PDF from URL: {e}"}), 400

        # ✅ Try reading the PDF
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)

        pdf_text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                pdf_text += page_text + "\n"
            except Exception as e:
                print(f"⚠️ Error reading page {i+1}: {e}")

        pdf_text = pdf_text.strip()

        if not pdf_text or len(pdf_text) < 100:
            return jsonify({
                "error": "Could not extract readable text from the PDF. It may be scanned or image-based."
            }), 400

        # ✅ Split and embed text
        docs = text_splitter.split_text(pdf_text)
        vector_store = FAISS.from_texts(docs, embeddings)

        print(f"✅ PDF processed successfully — {len(pdf_text)} characters, {len(docs)} chunks.")
        return jsonify({
            "status": "ready",
            "message": f"Processed {len(pdf_text)} characters and {len(docs)} chunks."
        })

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return jsonify({"error": str(e)}), 500


# --- Endpoint: Ask Question ---
@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_store, llm
    if not vector_store:
        return jsonify({"error": "No document processed yet"}), 400

    try:
        data = request.json or {}
        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # --- 1) Get a retriever and fetch top documents safely ---
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Try a few common retriever methods to be robust across versions
        docs = []
        try:
            # most common API
            docs = retriever.get_relevant_documents(question)
        except Exception:
            try:
                # some retrievers expose `retrieve`
                docs = retriever.retrieve(question)
            except Exception:
                try:
                    # fallback: some retrievers are runnables
                    docs = retriever.invoke({"query": question})
                except Exception:
                    docs = []

        # Normalize docs to list of text
        doc_texts = []
        for d in docs:
            if not d:
                continue
            # Document objects usually have .page_content
            content = None
            if hasattr(d, "page_content"):
                content = d.page_content
            elif isinstance(d, dict):
                content = d.get("page_content") or d.get("text") or str(d)
            else:
                content = str(d)
            if content:
                doc_texts.append(content)

        context_text = "\n\n".join(doc_texts).strip()
        if not context_text:
            context_text = "No relevant context found in the processed document."

        # --- 2) Build a plain text prompt from your system template ---
        # SYSTEM_TEMPLATE is expected to contain "{context}" already (from your file)
        prompt_text = SYSTEM_TEMPLATE.replace("{context}", context_text) + f"\n\nQuestion: {question}"

        # --- 3) Call the LLM safely and extract a string answer ---
        answer = None
        try:
            # ChatGoogleGenerativeAI usually exposes .invoke(...) returning a message-like object
            resp = llm.invoke(prompt_text)
            # resp might be an object with .content
            if hasattr(resp, "content"):
                answer = resp.content
            elif isinstance(resp, dict):
                # check several likely keys
                answer = resp.get("output") or resp.get("content") or resp.get("answer") or resp.get("output_text")
            else:
                answer = str(resp)
        except Exception as e:
            # Fallback: try calling llm as a function (some versions)
            try:
                resp2 = llm(prompt_text)
                if hasattr(resp2, "content"):
                    answer = resp2.content
                elif isinstance(resp2, dict):
                    answer = resp2.get("output") or resp2.get("content") or resp2.get("answer") or resp2.get("output_text")
                else:
                    answer = str(resp2)
            except Exception as e2:
                print(f"LLM invocation failed: {e} / fallback failed: {e2}")
                return jsonify({"error": "LLM invocation failed: " + str(e)}), 500

        # Ensure answer is a plain string
        if isinstance(answer, dict):
            answer = answer.get("output") or answer.get("content") or answer.get("answer") or str(answer)
        answer = "" if answer is None else str(answer)

        # Optional: truncate extremely long answers for safety (adjust as needed)
        if len(answer) > 20000:
            answer = answer[:20000] + "\n\n[Answer truncated]"

        print(f"Q: {question}\nA: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error answering question: {e}")
        return jsonify({"error": str(e)}), 500

SYSTEM_TEMPLATE = """
You are a helpful assistant. Your primary goal is to answer questions about the webpage, PDF, or video transcript context provided.
First, try to answer the user's question based *only* on the provided context below.
If the information is not available in the context, politely tell the user you couldn’t find it and then give a general answer.
Always keep your response clear and concise.

Context:
{context}
"""

@app.route('/')
def home():
    return "✅ DocuTalk AI backend is running locally!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
