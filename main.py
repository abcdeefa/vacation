from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp
import cv2
import json
import numpy as np
from scipy.spatial.distance import cosine
import os
import random

# ----------------------------
# 데이터셋 경로
dataset_dir = r"C:\parasunrin\dataset\dataset"

# ----------------------------
# 전역 모델 변수
processor, model = None, None

def load_model_once():
    """
    프로그램 시작 시 한 번만 모델 로드
    """
    global processor, model
    if processor is None or model is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)  # Torchvision 필요 없음
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# ----------------------------
def preprocess_embedding_images(directory):
    """
    데이터셋 이미지들을 임베딩하고 embedding.json에 저장
    """
    global processor, model
    embeddings = []
    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt")
        outputs = model.get_image_features(**inputs)
        embedding = outputs.detach().numpy().flatten()
        embeddings.append([embedding.tolist(), image_name])
    with open("embedding.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)

# ----------------------------
def embedding_image(image):
    """
    단일 이미지 임베딩 생성
    """
    global processor, model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy().flatten()

# ----------------------------
def similarity(embeded_image, top_k=5, top_p=0.9, temperature=1):
    """
    embedding.json과 코사인 유사도 비교 후 top match 반환
    """
    with open("embedding.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    similarities = []
    for embedding, image_name in embeddings:
        sim_score = 1 - cosine(embeded_image, embedding)
        similarities.append((sim_score, image_name))

    scores = np.array([s[0] for s in similarities])
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / exp_scores.sum()

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_sims = [similarities[i] for i in sorted_indices]

    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, top_p) + 1
    filtered_sims = sorted_sims[:cutoff]
    filtered_probs = sorted_probs[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()

    if len(filtered_sims) > top_k:
        filtered_sims = filtered_sims[:top_k]
        filtered_probs = filtered_probs[:top_k]
        filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_index = np.random.choice(len(filtered_sims), size=1, replace=False, p=filtered_probs)[0]
    return {"probability": float(filtered_probs[sampled_index]), "image_name": filtered_sims[sampled_index][1]}

# ----------------------------
def main():
    # 모델 한 번만 로드
    load_model_once()

    # embedding.json 생성
    if not os.path.exists("embedding.json"):
        print("Generating embeddings...")
        preprocess_embedding_images(dataset_dir)
        print("Done!")

    # 웹캠 세팅
    cap = cv2.VideoCapture(0)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cropped_face = frame[y:y+h, x:x+w]

                    embed = embedding_image(cropped_face)
                    top_match = similarity(embed)
                    print("Top match:", top_match)

                    # Top match 이미지 화면에 띄우기
                    top_image_path = os.path.join(dataset_dir, top_match["image_name"])
                    top_image = cv2.imread(top_image_path)
                    if top_image is not None:
                        cv2.imshow("Top Match", top_image)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
if __name__ == "__main__":
    main()
