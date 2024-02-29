from typing import Annotated
from fastapi import FastAPI, File, Form, UploadFile
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import io

app = FastAPI()

# Hugging Face test model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


@app.post("/visual-question/")
async def visual_question(
        file: Annotated[UploadFile, File()],
        question: Annotated[str, Form()]
        ):
    # Format the input image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Process the image and question
    inputs = processor(image, question, return_tensors="pt")
    
    # Generate answer
    out = model.generate(**inputs, max_new_tokens=30)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return {
        "file_name": file.filename,
        "question": question,
        "answer": answer
        }
