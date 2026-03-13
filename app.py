import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import os
from groq import Groq

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
import os
API_KEY = os.getenv("GROQ_API_KEY")
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("AI Resume Screener using LLaMA 3")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Screen Resumes"):

    results = []

    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        
        resume_embedding = model.encode([resume_text])
        job_embedding = model.encode([job_description])
        
        score = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        results.append((file.name, resume_text, round(score*100, 2)))

    results.sort(key=lambda x: x[2], reverse=True)

    st.subheader("Ranked Candidates")

    for rank, (name, resume_text, score) in enumerate(results, start=1):
        st.write(f"### {rank}. {name}")
        st.write(f"Matching Score: {score}%")

        # LLaMA 3 Analysis for Top 1 only
        if rank == 1:
            prompt = f"""
            You are an HR assistant.

            Job Description:
            {job_description}

            Candidate Resume:
            {resume_text}

            Provide:
            1. Matching Skills
            2. Missing Skills
            3. Overall Evaluation
            4. Final Hiring Recommendation
            """

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            st.write("#### LLaMA 3 Evaluation:")
            st.write(response.choices[0].message.content)
