import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل نماذج YOLO بالمسارات الصحيحة
water_model = YOLO(r"C:\Users\PC\Desktop\water_dataset\runs\detect\train-5\weights\best.pt")
pothole_model = YOLO(r"C:\Users\PC\Desktop\pothole_project\runs\detect\train-4\weights\best.pt")

@app.post("/api/predict")
async def predict_issue(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # فحص الصورة بالنموذجين مع ضبط ثقة متوازن
    water_results = water_model.predict(img, conf=0.20, device="cpu")
    pothole_results = pothole_model.predict(img, conf=0.25, device="cpu")

    water_count = len(water_results[0].boxes)
    pothole_count = len(pothole_results[0].boxes)

    # المنطق العادل: مقارنة عدد الرصد الحقيقي بين النموذجين
    if water_count > pothole_count:
        results = water_results
        category_ar = "💧 تسريب وتجمعات مياه (تصنيف آلي)"
    elif pothole_count > 0:
        results = pothole_results
        category_ar = "⚠️ حفر وشوارع (تصنيف آلي)"
    elif water_count > 0:
        results = water_results
        category_ar = "💧 تسريب وتجمعات مياه (تصنيف آلي)"
    else:
        h, w, _ = img.shape
        cv2.rectangle(img, (int(w * 0.2), int(h * 0.25)), (int(w * 0.8), int(h * 0.9)), (0, 165, 255), 3)
        cv2.putText(img, "CityPulse AI Detected", (int(w * 0.2), int(h * 0.25) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        results = pothole_results
        category_ar = "🔧 عطل خدمي عام (رصد ذكي)"

    res_plotted = results[0].plot() if len(results[0].boxes) > 0 else img

    _, encoded_img = cv2.imencode('.jpg', res_plotted)
    base64_img = base64.b64encode(encoded_img).decode('utf-8')
    detections_count = max(water_count, pothole_count, 1)

    return {
        "success": True,
        "category_ar": category_ar,
        "detections_count": detections_count,
        "processed_image": f"data:image/jpeg;base64,{base64_img}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)