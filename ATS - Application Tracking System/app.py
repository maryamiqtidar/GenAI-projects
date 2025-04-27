from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
from PIL import Image
import pdf2image
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini Model
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(input_prompt, pdf_content, job_description):
    response = model.generate_content([
        {"text": input_prompt},
        pdf_content[0],
        {"text": job_description}
    ])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=r"C:\Program Files (x86)\poppler\Library\bin")
        first_page = images[0]

        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_byte_arr
                }
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# --- Streamlit Page Config ---
st.set_page_config(page_title="ATS Resume Expert 🚀", page_icon="📄", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f0f2f6 0%, #d9e2ec 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #f0f2f6 0%, #d9e2ec 100%);
        padding: 10px;
    }
    button {
        height: 3em;
        width: 100%;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        background: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s ease-in-out;
    }
    button:hover {
        background: #45a049;
        transform: scale(1.02);
    }
    .title {
        text-align: center;
        font-size: 36px;
        color: #333333;
        margin-bottom: 5px;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555555;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .stFileUploader {
        border-radius: 10px;
        background: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar:
    st.title("About App 📚")
    st.info("""
    **ATS Resume Expert** helps you:
    - Analyze your Resume 📄
    - Match against any Job Description 💼
    - Get ATS percentage match 🎯
    - Find missing keywords 🔍
    Powered by Google's **Gemini 1.5 Flash** 🚀
    """)
    st.markdown("---")
    st.caption("Created by Maryam Iqtidar")

# --- Main Title ---
st.markdown("<div class='title'>📄 ATS Resume Expert</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smartly match your Resume with Job Descriptions using AI!</div>", unsafe_allow_html=True)

st.divider()

# --- Resume Upload Section ---
st.subheader("Step 1: Upload Your Resume 📑")
uploaded_file = st.file_uploader("📑 Upload Your Resume (PDF only)", type=["pdf"])

if uploaded_file:
    st.success("✅ Resume Uploaded Successfully!")
    st.divider()
    
    # --- Job Description and Skills Inputs ---
    input_text = st.text_area("🧾 Paste the Job Description below:", height=200, placeholder="Enter Job Description here...")

    # New Feature: Skills Input
    skills_input = st.text_area("🧑‍💻 Enter Skills (Comma-separated):", height=100, placeholder="E.g., Python, Machine Learning, Data Analysis")

    # --- Buttons Section ---
    st.subheader("Step 2: Choose Action 🚀")

    col1, col2, col3 = st.columns(3)

    with col1:
        analyze_resume = st.button("📝 Analyze Resume")

    with col2:
        match_percentage = st.button("🎯 Calculate Match %")

    with col3:
        match_skills = st.button("🔑 Check Skills Match")

    # --- Prompts ---
    input_prompt1 = """
        You are an experienced Technical HR Manager.
        Review the resume against the job description and provide:
        - 3-5 bullet points highlighting key strengths
        - 3-5 bullet points highlighting key weaknesses
        - Be concise and professional.
        """

    input_prompt3 = """
    You are an expert ATS (Applicant Tracking System) scanner.
    Evaluate the resume against the job description and provide:
    - ATS match percentage (only a single number like 75%)
    - List missing keywords (in bullet points, 5-10 items max)
    - Give a short 2-line final remark only.
    Be precise, avoid long paragraphs.
    """

    # --- Skills Matching Logic ---
    def skill_matching(resume_text, skills_list):
        # Tokenize both the resume and the skills input
        cv_words = set(re.findall(r'\w+', resume_text.lower()))
        skills_set = set([skill.lower().strip() for skill in skills_list.split(',')])

        # Compare skills
        missing_skills = skills_set - cv_words
        return missing_skills

    # --- Logic Section ---
    if analyze_resume:
        if uploaded_file:
            with st.spinner('🔎 Analyzing your resume...'):
                pdf_content = input_pdf_setup(uploaded_file)
                response = get_gemini_response(input_prompt1, pdf_content, input_text)
            st.success("✅ Resume Evaluation Ready!")
            st.subheader("📝 Evaluation Result:")
            st.write(response)
        else:
            st.warning("⚠️ Please upload a resume first.")

    if match_percentage:
        if uploaded_file:
            with st.spinner('🎯 Calculating match percentage...'):
                pdf_content = input_pdf_setup(uploaded_file)
                response = get_gemini_response(input_prompt3, pdf_content, input_text)
            st.success("✅ Match Calculation Ready!")
            st.subheader("🎯 Match Result:")
            st.write(response)
        else:
            st.warning("⚠️ Please upload a resume first.")

    if match_skills:
        if uploaded_file and skills_input:
            with st.spinner('🔑 Checking skills match...'):
                pdf_content = input_pdf_setup(uploaded_file)
                resume_text = get_gemini_response(
                    """
                    Extract plain text content from the resume only.
                    Do not add any explanation.
                    Return the resume text without extra formatting.
                    """,
                    pdf_content, input_text
                )
                missing_skills = skill_matching(resume_text, skills_input)
                if missing_skills:
                    st.success("✅ Skills Match Complete!")
                    st.subheader("🧑‍💻 Missing Skills:")
                    st.write(f"The following required skills were not found in the resume: {', '.join(missing_skills)}")
                else:
                    st.success("✅ Skills Match Complete!")
                    st.write("No missing skills were found in the resume!")
        else:
            st.warning("⚠️ Please upload a resume and enter skills first.")

else:
    st.warning("⚠️ Please upload a resume to proceed.")
