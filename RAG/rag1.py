import json
import os
import sys
import boto3
import streamlit as st
from dotenv import load_dotenv

## We will be using Titan Embeddings Model To generate Embedding
from langchain_aws import BedrockEmbeddings, BedrockLLM

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Load AWS credentials from .env file
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')  # if using temporary credentials
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vector_store(docs):
    if not docs:
        st.error("No text found in the PDF. Please upload a valid PDF.")
        return None
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def get_titan_llm():
    ## Create the Titan Model
    llm = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock, model_kwargs={'max_tokens': 1000})
    return llm

def get_titan_lite_llm():
    ## Create the Titan Lite Model
    llm = BedrockLLM(model_id="amazon.titan-text-lite-v1", client=bedrock, model_kwargs={'max_tokens': 1000})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({"query": query})
    return answer['result']

def main():
    st.set_page_config(page_title="svj", layout="wide")
    st.title("sruthas RAG Application💁")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            docs = data_ingestion(uploaded_file)
            vectorstore_faiss = get_vector_store(docs)
            if vectorstore_faiss:
                st.success("Vector store created and updated successfully!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Titan Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_titan_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)
            st.success("Done")
            st.session_state.history.append({"question": user_question, "response": response})

    if st.button("Titan Lite Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_titan_lite_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)
            st.success("Done")
            st.session_state.history.append({"question": user_question, "response": response})

    st.sidebar.title("History")
    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        st.sidebar.write(f"**Question:** {entry['question']}")
        st.sidebar.write(f"**Response:** {entry['response']}")

if __name__ == "__main__":
    main()
