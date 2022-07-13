from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from kospacing_helper import KoSpacingHelper

class Item(BaseModel):
	sentences: list

koSpacingHelper = KoSpacingHelper()

app = FastAPI()

origins = [
    "*",
    "https://nextlab-newspaper.s3.ap-northeast-2.amazonaws.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"api": "Korean Spacing"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}


@app.post("/kospacing/predict")
def predict(item: Item):
    sentences = item.sentences
    ret = koSpacingHelper.predict(sentences)
    print(ret)
    return ret