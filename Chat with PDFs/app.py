import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import requests

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load external CSS file
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create and store vector database
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, respond with "answer is not available in the context".

    Context:
    {context}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        temperature=0.5,
        top_p=1,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user input and display result
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.markdown("### 🤖 Gemini's Reply")
    st.markdown(f"""
    <div class='response-box'>
        <p>{response["output_text"]}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Relevant Chunks"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i + 1}**")
            st.write(doc.page_content)

# MAIN APP
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini", page_icon="📄", layout="wide")

    # Load external styles
    load_local_css("styles.css")

    # --- Header with Animation and Title ---
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        try:
            lottie_pdf = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_tfb3estd.json")
            if lottie_pdf:
                st_lottie(lottie_pdf, height=120, key="pdf")
            else:
                st.image("fallback_image.png", width=100)
        except:
            st.warning("⚠️ Could not load animation.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='header-box'>
            <h1 style='margin-bottom: 0.3em;'>Chat with Your PDFs</h1>
            <p style='font-size: 1.1em; color: #555;'>
                Upload your documents to effortlessly extract key insights, generate summaries,
                and receive instant answers — all in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)


    # --- Sidebar for Upload and Processing ---
    with st.sidebar:
        st.markdown("### 📂 Upload and Process Files")
        pdf_docs = st.file_uploader("📁 Upload PDF Files", accept_multiple_files=True)
        if st.button("✅ Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.sidebar.success("✅ PDFs processed! Scroll down to ask questions.")
            else:
                st.sidebar.warning("⚠️ Please upload at least one PDF.")

    # --- Main Section for Question and Results ---
    st.markdown("---")
    st.markdown("### 💬 Ask a Question About Your PDFs")

    user_question = st.text_input("Type your question here:")
    if user_question:
        user_input(user_question)

    # --- Footer ---
    st.markdown("""
    <hr>
    <div class='footer' style='font-size: 0.9em; text-align: center; color: gray;'>
        Built with <a href='https://streamlit.io' target='_blank' style='color: gray; text-decoration: none;'>Streamlit</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
