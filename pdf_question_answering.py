"""Pdf Question Answering Streamlit App"""

import PyPDF2
import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_pdf(pdf_file, questions):
    pdf_text = extract_text_from_pdf(pdf_file)

    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    sec_key = "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" 
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.7
    )

    answers = []
    for question in questions:
        prompt = f"Context: {pdf_text}\n\nQuestion: {question}"
        answer = llm.invoke(prompt)
        answers.append(answer)

    result = "\n\n".join([f"Question: {questions[i]}\nAnswer: {answers[i]}" for i in range(len(questions))])

    return result

st.title("PDF Q&A with HuggingFace Model")

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

questions_input = st.text_input("Enter your questions (comma separated)", "What is the main topic of the document?, Summarize the key points.")

if st.button("Generate Answers"):
    if pdf_file is not None and questions_input:
        questions = [q.strip() for q in questions_input.split(",")]
        output = process_pdf(pdf_file, questions)
        st.text_area("Output", output, height=300)
    else:
        st.error("Please upload a PDF and enter some questions.")
