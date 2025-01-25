from fastapi import FastAPI, File, Form, UploadFile

from ingestion import ingest_in_vectordb
from llm import get_llm_response
from retrieval import get_relevant_docs

app = FastAPI()


@app.post("/upload")
async def upload_pdf(username: str = Form(...), file: UploadFile = File(...)):
    try:
        pdf_path = f"temp_{username}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())  # Use await to read the file
        ingest_in_vectordb(pdf_path, username)
        return {"result": "Successful"}
    except Exception as e:
        print(e)
        return {"error": str(e)}


@app.post("/get_response")
def get_response(username: str = Form(...), question: str = Form(...)):
    try:
        context = get_relevant_docs(question, username)
        response = get_llm_response(question, context)
        return response
    except Exception as e:
        print(e)
