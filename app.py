from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Đường dẫn video
VIDEO_PATH = r"C:\GIAOTHONG\traffic-monitor\traffic-backend\videos\154647-808044372_small.mp4"

# Kiểm tra video có tồn tại không
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"⚠️ Lỗi: Không tìm thấy video tại {VIDEO_PATH}")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("⚠️ Lỗi: Không thể mở video!")

# Load mô hình YOLO
model = YOLO("yolov8n.pt")  # Sử dụng YOLOv8n (phiên bản nhỏ)

# Biến toàn cục lưu số lượng xe
vehicle_count_data = 0

def detect_traffic():
    """ Phát hiện xe và phát video stream """
    global vehicle_count_data

    while True:
        success, frame = cap.read()
        if not success:
            # Reset video khi kết thúc
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Phát hiện đối tượng bằng YOLO
        results = model(frame, stream=True)

        # Reset số lượng xe trong khung hình hiện tại
        current_vehicle_count = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ và class của đối tượng
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Chỉ xử lý các đối tượng là xe (car, truck, bus, motorcycle)
                if class_name in ["car", "truck", "bus", "motorcycle"]:
                    # Tăng số lượng xe hiện có trong khung hình
                    current_vehicle_count += 1

                    # Vẽ khung xe
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cập nhật số lượng xe toàn cục
        vehicle_count_data = current_vehicle_count

        # Hiển thị số xe trên video
        cv2.putText(frame, f"Vehicles: {vehicle_count_data}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Chuyển đổi frame thành định dạng JPEG để hiển thị
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# API phát video
@app.route('/video_feed')
def video_feed():
    return Response(detect_traffic(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API lấy số lượng xe
@app.route('/vehicle_count')
def get_vehicle_count():
    return jsonify({"count": vehicle_count_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)