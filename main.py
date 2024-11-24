from fastapi import FastAPI, Body, UploadFile, File
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from embedding import embedding_doc
from answer import answer_query

app = FastAPI(title="RAG GENAI")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.post("/upload_doc")
def document_upload(exist_file: str = "", file: UploadFile = File(...)):
    try:
        name = file.filename
        contents = file.file.read()
        with open(f"docs/{name}", 'wb') as f:
            f.write(contents)
        
        embeddings = embedding_doc(name, exist_file)
        if embeddings==True:
            return "File upload successful"
        else:
            return embeddings
    except Exception as e:
        return e
    
@app.get("/query")
def querying(question: str, filename:str):
    try:
        response = answer_query(question=question, file=filename)
        return response
    except Exception as e:
        return e

if __name__=="__main__":
    uvicorn.run("main:app",
                host="localhost",
                port=2006,
                reload=True)