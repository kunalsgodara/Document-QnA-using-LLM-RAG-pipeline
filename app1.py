import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import time 

load_dotenv()

# loading the environment variables to use API keys 
groq_api_key = os.environ.get("GROQ_API_KEY")

st.title("ChatGroq with Llama3")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
'''
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question: {input}
'''
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="gemma2:2b") #creating a word embedding instance 
        st.session_state.loader = PyPDFDirectoryLoader("./census data") # documents ingestion
        st.session_state.documents = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) # spiltting docs into chunks 
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.documents[:50] ) # '''we will only use first 50 docs for learinng as loading full doc may take more time''' 
        st.sesion_state.vectordb = FAISS.from_documents(st.session_state.final_doc,st.session_state.embeddings) # storing final doc into vector database
prompt1 = st.text_input("Enter Query from documents ")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("vector store DB is ready")

                                
    




if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({"input":prompt1})
    st.write(response["answer"])

        # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")