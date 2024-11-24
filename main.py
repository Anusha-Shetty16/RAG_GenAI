from fastapi import FastAPI
import uvicorn


app = FastAPI(title="RAG GENAI")

if __name__=="__main__":
    uvicorn.run("main:app",
                host="localhost",
                port=2006,
                reload=True)