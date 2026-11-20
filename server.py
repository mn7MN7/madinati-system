import base64
import math
import os
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.environ.get("DB_PATH", str(DATA_DIR / "reports.db")))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(BASE_DIR / "best.pt")))
HF_MODEL_URL = "https://huggingface.co/Mazinmn77/citypulse-yolo/resolve/main/best.pt"

app = FastAPI(title="Nabdh Madinati — CityPulse AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


def get_device() -> str:
    if os.environ.get("FORCE_CPU", "0") == "1":
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def safe_file_response(filename: str):
    target = BASE_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"الملف {filename} غير موجود على السيرفر")
    return FileResponse(str(target))


@app.get("/", response_class=HTMLResponse)
async def hub_page():
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>نبض مدينتي — بوابة الواجهات</title>
      <style>
        body { font-family: Tahoma, Arial, sans-serif; background: #0d0d0d; color: #fff; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .box { background: #181818; border: 1px solid #2e2e2e; border-radius: 16px; padding: 30px 24px; width: 92%; max-width: 440px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        h1 { margin-top: 0; color: #f59e0b; font-size: 24px; }
        p { color: #bbb; font-size: 14px; margin-bottom: 24px; }
        a { display: block; text-decoration: none; background: #f59e0b; color: #111; font-weight: bold; padding: 14px; border-radius: 10px; margin-bottom: 12px; font-size: 15px; transition: 0.2s; }
        a:hover { opacity: 0.9; }
        a.sec { background: #262626; color: #fff; border: 1px solid #3d3d3d; }
      </style>
    </head>
    <body>
      <div class="box">
        <h1>نبض مدينتي — CityPulse AI</h1>
        <p>اختر الواجهة المطلوبة للمعاينة أو العرض المباشر:</p>
        <a href="/index.html">الواجهة الرئيسية (index.html)</a>
        <a class="sec" href="/home.html">واجهة المواطن / البلاغات (home.html)</a>
        <a class="sec" href="/government.html">لوحة الجهات الحكومية (government.html)</a>
      </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": unified_model is not None,
        "device": get_device(),
    }


@app.get("/index.html")
async def serve_index():
    return safe_file_response("index.html")


@app.get("/home.html")
async def serve_home():
    return safe_file_response("home.html")


@app.get("/government.html")
async def serve_government():
    return safe_file_response("government.html")


@app.get("/logo.png")
async def serve_logo():
    return safe_file_response("logo.png")


@app.get("/logo-ekg.png")
async def serve_logo_ekg():
    return safe_file_response("logo-ekg.png")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_no TEXT,
                category_ar TEXT,
                image_path TEXT,
                date_time TEXT,
                status TEXT,
                lat REAL,
                lng REAL,
                location TEXT,
                similar_reports TEXT DEFAULT '',
                ai_priority TEXT DEFAULT 'عادية 🟢',
                priority_score INTEGER DEFAULT 1,
                after_image_path TEXT,
                upvotes INTEGER DEFAULT 1
            )
            """
        )

        cursor.execute("PRAGMA table_info(reports)")
        columns = [col[1] for col in cursor.fetchall()]

        extra_columns = {
            "lat": "REAL",
            "lng": "REAL",
            "location": "TEXT",
            "similar_reports": "TEXT DEFAULT ''",
            "ai_priority": "TEXT DEFAULT 'عادية 🟢'",
            "priority_score": "INTEGER DEFAULT 1",
            "after_image_path": "TEXT",
            "upvotes": "INTEGER DEFAULT 1",
        }
        for col_name, col_type in extra_columns.items():
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE reports ADD COLUMN {col_name} {col_type}")

        conn.commit()
    finally:
        conn.close()


def load_unified_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        print("⏳ جاري تحميل ملف best.pt من Hugging Face...")
        try:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink(missing_ok=True)
            req = urllib.request.Request(
                HF_MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as response, open(MODEL_PATH, "wb") as out_file:
                out_file.write(response.read())
            print(f"✅ اكتمل التحميل ({MODEL_PATH.stat().st_size} bytes)")
        except Exception as e:
            print("⚠️ تعذر التحميل من Hugging Face:", e)

    print("\n" + "=" * 50)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        try:
            device = get_device()
            loaded = YOLO(str(MODEL_PATH))
            loaded.to(device)
            print(f"✅ تم تحميل النموذج الموحد بنجاح على {device}!")
            print("=" * 50 + "\n")
            return loaded
        except Exception as e:
            print(f"❌ فشل تهيئة النموذج: {e}")
            print("=" * 50 + "\n")
            return None
    else:
        print("❌ تنبيه: ملف best.pt غير موجود أو غير مكتمل!")
        print("=" * 50 + "\n")
        return None


init_db()
unified_model = load_unified_model()


def calculate_distance_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_visual_severity(img, boxes, cls_id, max_conf):
    h_img, w_img, _ = img.shape
    img_area = w_img * h_img

    if len(boxes) == 0:
        return {"level": "عادية 🟢", "score": 1}

    highest_box = max(boxes, key=lambda b: float(b.conf.item()))
    x1, y1, x2, y2 = highest_box.xyxy[0].tolist()

    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_ratio = box_area / img_area if img_area > 0 else 0.05

    crop_y1, crop_y2 = max(0, int(y1)), min(h_img, int(y2))
    crop_x1, crop_x2 = max(0, int(x1)), min(w_img, int(x2))
    roi = img[crop_y1:crop_y2, crop_x1:crop_x2]

    if cls_id in [0, 3] and roi.size > 0:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        bright_ratio = np.sum(bright_mask > 0) / roi.size

        if bright_ratio > 0.12 or area_ratio > 0.25:
            return {"level": "حرجة وعاجلة (خطر شرارة/تماس) 🔴", "score": 5}

    if cls_id == 1:
        if area_ratio > 0.22:
            return {"level": "حرجة وعاجلة (تخسف خطير بالشارع) 🔴", "score": 5}
        elif area_ratio > 0.09:
            return {"level": "متوسطة (حفرة واسعة) 🟡", "score": 3}
        else:
            return {"level": "عادية (حفرة محدودة) 🟢", "score": 1}

    if area_ratio > 0.30:
        return {"level": "حرجة وعاجلة 🔴", "score": 5}
    elif area_ratio > 0.12:
        return {"level": "متوسطة 🟡", "score": 3}
    else:
        return {"level": "عادية 🟢", "score": 1}


def detect_from_image(img):
    best_category = "🔧 لم يتم التعرف على العطل آلياً"
    max_conf = 0.0
    cls_id = -1

    results = unified_model.predict(
        img,
        conf=0.08,
        augment=True,
        verbose=False,
        device=get_device(),
    )
    result = results[0]
    boxes = result.boxes

    if len(boxes) > 0:
        highest_conf_box = max(boxes, key=lambda b: float(b.conf.item()))
        cls_id = int(highest_conf_box.cls.item())
        max_conf = float(highest_conf_box.conf.item())

        manual_mapping = {
            0: "⚡ عطل أعمدة كهربائية (رصد آلي)",
            1: "⚠️ حفر وشوارع (تصنيف آلي)",
            2: "🗑️ تجمعات نفايات ومخلفات (رصد آلي)",
            3: "🔥 عطل محولات كهربائية (رصد آلي)",
        }
        best_category = manual_mapping.get(cls_id, "🔧 لم يتم التعرف على العطل آلياً")

    priority_info = calculate_visual_severity(img, boxes, cls_id, max_conf)
    return result, boxes, best_category, max_conf, cls_id, priority_info


@app.post("/api/preview")
async def preview_issue(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "error": "Invalid image file"}

    if unified_model is None:
        return {"success": False, "error": "Model not loaded"}

    _, boxes, best_category, max_conf, cls_id, priority_info = detect_from_image(img)

    return {
        "success": True,
        "category_ar": best_category,
        "confidence": round(max_conf, 4),
        "detected": len(boxes) > 0,
        "ai_priority": priority_info["level"],
        "priority_score": priority_info["score"],
    }


@app.post("/api/predict")
async def predict_issue(
    file: UploadFile = File(...),
    lat: float = Form(35.4681),
    lng: float = Form(44.3922),
    location: str = Form("كركوك، العراق"),
):
    start_time = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "error": "Invalid image file"}

    if unified_model is None:
        return {"success": False, "error": "Model not loaded"}

    print(f"\n--- 🔍 تحليل الصورة البصري الذكي ({datetime.now().strftime('%H:%M:%S')}) ---")
    result, boxes, best_category, max_conf, cls_id, priority_info = detect_from_image(img)

    elapsed = time.time() - start_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    if len(boxes) > 0 and max_conf >= 0.08:
        res_plotted = result.plot()
        max_count = len(boxes)
    else:
        res_plotted = img.copy()
        max_count = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"report_{timestamp}.jpg"
    image_save_path = UPLOAD_DIR / image_filename
    cv2.imwrite(str(image_save_path), res_plotted)

    report_no = f"#{np.random.randint(40000, 99999)}"
    date_str = datetime.now().strftime("%Y/%m/%d")
    status = "قيد المعالجة الميدانية"

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reports (
                report_no, category_ar, image_path, date_time, status,
                lat, lng, location, similar_reports, ai_priority, priority_score, upvotes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_no,
                best_category,
                f"/uploads/{image_filename}",
                date_str,
                status,
                lat,
                lng,
                location,
                "",
                priority_info["level"],
                priority_info["score"],
                1,
            ),
        )
        new_report_id = cursor.lastrowid

        cursor.execute("SELECT * FROM reports WHERE id != ? ORDER BY id DESC", (new_report_id,))
        existing_reports = cursor.fetchall()

        similar_ids_list = []
        for rep in existing_reports:
            rep_status = rep["status"] or ""
            if "منجز" in rep_status or "رفض" in rep_status or "✅" in rep_status:
                continue

            old_lat = float(rep["lat"] or 35.4681)
            old_lng = float(rep["lng"] or 44.3922)

            distance = calculate_distance_meters(lat, lng, old_lat, old_lng)

            if distance <= 45 and rep["category_ar"] == best_category:
                similar_ids_list.append(str(rep["id"]))

                other_similars = [s for s in (rep["similar_reports"] or "").split(",") if s]
                if str(new_report_id) not in other_similars:
                    other_similars.append(str(new_report_id))
                    cursor.execute(
                        "UPDATE reports SET similar_reports = ? WHERE id = ?",
                        (",".join(other_similars), rep["id"]),
                    )

        if similar_ids_list:
            similar_str = ",".join(similar_ids_list)
            cursor.execute(
                "UPDATE reports SET similar_reports = ? WHERE id = ?",
                (similar_str, new_report_id),
            )

        conn.commit()
    finally:
        conn.close()

    _, encoded_img = cv2.imencode(".jpg", res_plotted)
    base64_img = base64.b64encode(encoded_img).decode("utf-8")

    return {
        "success": True,
        "report_no": report_no,
        "category_ar": best_category,
        "ai_priority": priority_info["level"],
        "priority_score": priority_info["score"],
        "detections_count": max_count,
        "date": date_str,
        "status": status,
        "lat": lat,
        "lng": lng,
        "location": location,
        "processed_image": f"data:image/jpeg;base64,{base64_img}",
    }


@app.get("/api/reports")
async def get_reports():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports ORDER BY id DESC")
        rows = cursor.fetchall()
        reports_list = [dict(row) for row in rows]
        return {"success": True, "reports": reports_list}
    finally:
        conn.close()


@app.post("/api/reports/{report_id}/upvote")
async def upvote_report(report_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reports SET upvotes = upvotes + 1, priority_score = priority_score + 1 WHERE id = ?",
            (report_id,),
        )
        conn.commit()

        cursor.execute("SELECT upvotes FROM reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="البلاغ غير موجود")
        new_votes = row["upvotes"]
        return {"success": True, "upvotes": new_votes}
    finally:
        conn.close()


class StatusUpdate(BaseModel):
    status: str


@app.put("/api/reports/{report_id}/status")
async def update_report_status(report_id: int, update: StatusUpdate):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE reports SET status = ? WHERE id = ?", (update.status, report_id))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="البلاغ غير موجود")
        return {"success": True, "message": f"تم تحديث حالة البلاغ #{report_id} بنجاح إلى: {update.status}"}
    finally:
        conn.close()


@app.post("/api/reports/{report_id}/complete")
async def complete_report(
    report_id: int,
    file: UploadFile = File(None),
    status: str = Form("تم إنجاز الصيانة والمعالجة بنجاح ✅"),
    note: str = Form(""),
):
    after_image_path = None
    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"after_{report_id}_{timestamp}.jpg"
            save_path = UPLOAD_DIR / filename
            cv2.imwrite(str(save_path), img)
            after_image_path = f"/uploads/{filename}"

    final_status = status
    if note:
        final_status += f" | ملاحظة الإنجاز: {note}"

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM reports WHERE id = ?", (report_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="البلاغ غير موجود")

        if after_image_path:
            cursor.execute(
                "UPDATE reports SET status = ?, after_image_path = ? WHERE id = ?",
                (final_status, after_image_path, report_id),
            )
        else:
            cursor.execute(
                "UPDATE reports SET status = ? WHERE id = ?",
                (final_status, report_id),
            )
        conn.commit()
    finally:
        conn.close()

    return {"success": True, "message": "تم إنجاز البلاغ ورفع صور الصيانة بنجاح"}


@app.get("/api/analytics")
async def get_analytics():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM reports")
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) as completed FROM reports WHERE status LIKE '%منجز%' OR status LIKE '%✅%' OR status LIKE '%قبول%'"
        )
        completed = cursor.fetchone()["completed"]

        cursor.execute("SELECT COUNT(*) as critical FROM reports WHERE priority_score >= 5")
        critical = cursor.fetchone()["critical"]

        cursor.execute("SELECT category_ar, COUNT(*) as count FROM reports GROUP BY category_ar")
        categories = [dict(row) for row in cursor.fetchall()]

        return {
            "success": True,
            "analytics": {
                "total_reports": total,
                "completed_reports": completed,
                "critical_reports": critical,
                "completion_rate": round((completed / total * 100) if total > 0 else 0, 1),
                "categories_breakdown": categories,
            },
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server_5:app", host="0.0.0.0", port=port)