from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

app = FastAPI()

MODEL_PATH = "selimhafez/review-bert-model"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model.eval()


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Restaurant review rating API is running"}


@app.post("/predict")
def predict_rating(request: ReviewRequest):
    text = request.text.strip()

    if not text:
        return {
            "rating": 0,
            "probabilities": []
        }

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item() + 1

    probabilities = [float(p) for p in probs[0]]

    return {
        "rating": predicted_class,
        "probabilities": probabilities
    }
