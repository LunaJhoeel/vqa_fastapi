from typing import Annotated
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()

@app.post("/visual-question/")
async def visual_question(
    file: Annotated[UploadFile, File()],
    question: Annotated[str, Form()]
):
    return {
        "file_name": file.filename,
        "question": question
    }
