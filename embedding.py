from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import google.generativeai as genai
import os
import shutil
import glob
from dotenv import load_dotenv

load_dotenv(".env")

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def embedding_doc(file, exist_file):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if file.lower().endswith(('.txt')):
            loader = TextLoader(f"docs/{file}")
            documents = loader.load()
        else:
            print("PDF file")
            loader = PyPDFLoader(f"docs/{file}")
            documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1200, chunk_overlap=200
            )
        text_chunks = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(text_chunks,embeddings)
        print(vector_store)
        if exist_file=="":
            print("Embedd")
            vector_store.save_local(f"embeddings/{file}")
            return True
        
        elif (os.path.exists(f"embeddings/{exist_file}")):
            vector_store_update = FAISS.load_local(f"embeddings/{exist_file}",embeddings)
            vector_store_update.merge_from(vector_store)
            vector_store_update.save_local(f"embeddings/{exist_file}")
            return True
        else:
            return "The file to merge embeddings does not exist."
        
    except Exception as e:
        return e