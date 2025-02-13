import os
import faiss
import pickle
import torch
import openai
import streamlit as st
import numpy as np
import pandas as pd
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load BERT model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def extract_text_from_resume(pdf_path):
    """Extract text from PDF resume using pdfplumber"""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(
            page.extract_text() 
            for page in pdf.pages 
            if page.extract_text()
        )

def get_embedding(text):
    """Generate BERT embeddings for text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Initialize FAISS index
INDEX_PATH = "faiss_index.pkl"

def initialize_faiss(job_embeddings):
    """Create or load FAISS index with job embeddings"""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    
    embedding_dim = job_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(job_embeddings)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    return index

# Load job data and prepare embeddings
jobs = pd.read_csv("C:/Users/Owner/OneDrive/Documentos/Job_Marketing/resume_builder_project/Jobs.csv")
job_embeddings = np.array(
    [get_embedding(job) for job in jobs["Description"]],
    dtype=np.float32
).squeeze()  # Remove singleton dimensions

index = initialize_faiss(job_embeddings)

def match_jobs(resume_text, top_k=5):
    """Find top matching jobs using FAISS similarity search"""
    resume_embedding = get_embedding(resume_text)
    distances, indices = index.search(resume_embedding, k=top_k)
    return jobs.iloc[indices.flatten()]

# FastAPI backend
app = FastAPI()

@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile = File(...)):
    """API endpoint for resume analysis"""
    # Save uploaded file temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(await file.read())
    
    resume_text = extract_text_from_resume("temp_resume.pdf")
    matches = match_jobs(resume_text)
    os.remove("temp_resume.pdf")
    return {"matches": matches.to_dict(orient="records")}

# Streamlit frontend
def streamlit_app():
    st.title("AI Resume Analyzer & Job Matcher")
    
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)", 
        type=["pdf"]
    )
    
    if uploaded_file:
        with st.spinner("Analyzing resume..."):
            # Save uploaded file
            with open("temp_resume.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            resume_text = extract_text_from_resume("temp_resume.pdf")
            matches = match_jobs(resume_text)
            os.remove("temp_resume.pdf")

            st.subheader("Extracted Text:")
            st.text(resume_text[:1000] + "...")  # Show first 1000 chars
            
            st.subheader("Top Job Matches:")
            st.dataframe(matches[["Job Title", "Company", "Description"]])

if __name__ == "__main__":
    # Run Streamlit app with: streamlit run app.py
    streamlit_app()