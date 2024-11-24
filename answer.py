import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain_community.tools import _import_ddg_search_tool_DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

def answer_query(question,file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(f"embeddings/{file}", embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", temperature=0.7)
    prompt_template = """
    {context}
    {question}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain_type_kwargs = {"prompt":prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        verbose=False,
        chain_type_kwargs=chain_type_kwargs
    )
    result = qa({"query":question})["result"]
    return result