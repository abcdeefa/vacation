from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import json
from scipy.spatial.distance import cosine
from transformers import CLIPProcessor, CLIPModel

# ----------------------------
# 서버 세팅
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 테스트용, 실제 서비스 시 도메인 지정
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 데이터셋 경로
dataset_dir = r"C:\parasunrin\dataset\dataset"

# ----------------------------
# 모델 로드 (전역)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# ----------------------------
# embedding.json 로드 또는 생성
embedding_file = "embedding.json"
if not os.path.exists(embedding_file):
    embeddings = []
    for image_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt")
        outputs = model.get_image_features(**inputs)
        embedding = outputs.detach().numpy().flatten()
        embeddings.append([embedding.tolist(), image_name])
    with open(embedding_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)
else:
    with open(embedding_file, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

# ----------------------------
def embedding_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy().flatten()

def similarity(embeded_image):
    """
    샘플링 제거: Top1 코사인 유사도 기반 확률 반환
    """
    similarities = []
    for embedding, image_name in embeddings:
        sim_score = 1 - cosine(embeded_image, embedding)
        similarities.append((sim_score, image_name))

    # 내림차순 정렬
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Top1 선택
    top_score, top_image_name = similarities[0]
    return {"probability": float(top_score), "image_name": top_image_name}

# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embed = embedding_image(image_bytes)
    top_match = similarity(embed)
    return JSONResponse(content=top_match)

# ----------------------------
@app.get("/image/{image_name}")
async def get_image(image_name: str):
    path = os.path.join(dataset_dir, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Image not found"}, status_code=404)
