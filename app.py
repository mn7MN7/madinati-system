import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

# تحميل نماذج YOLO المدربة الخاصة بمشروع نبض مدينتي
water_model = YOLO("water_best.pt")
pothole_model = YOLO("pothole_best.pt")

def predict(image):
    if image is None:
        return None, "الرجاء رفع صورة الشارع أولاً."
    
    # تحويل الصورة من الواجهة إلى صيغة OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    water_results = water_model.predict(img_bgr, conf=0.20, device="cpu")
    pothole_results = pothole_model.predict(img_bgr, conf=0.25, device="cpu")

    water_count = len(water_results[0].boxes)
    pothole_count = len(pothole_results[0].boxes)

    if water_count > pothole_count or (water_count > 0 and pothole_count == 0):
        results = water_results
        category = "💧 تسريب وتجمعات مياه (تصنيف آلي)"
    elif pothole_count > 0:
        results = pothole_results
        category = "⚠️ حفر وشوارع (تصنيف آلي)"
    else:
        category = "🔧 عطل خدمي عام (رصد ذكي)"
        results = pothole_results

    res_plotted = results[0].plot() if len(results[0].boxes) > 0 else img_bgr
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    report = f"التصنيف: {category}\nعدد الاكتشافات: {max(water_count, pothole_count, 0)}"
    return res_rgb, report

# بناء واجهة التطبيق الاحترافية
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="رفع صورة الشارع أو المشكلة"),
    outputs=[
        gr.Image(type="numpy", label="الصورة بعد تحليل الذكاء الاصطناعي"),
        gr.Textbox(label="تقرير الرصد الآلي")
    ],
    title="CityPulse AI - نبض مدينتي",
    description="منصة الذكاء الاصطناعي لرصد البنية التحتية في العراق (مشروع تخرج - مصطفى، ديار، ومازن)"
)

if __name__ == "__main__":
    demo.launch()