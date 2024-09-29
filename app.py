import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
# Tải mô hình kNN đã huấn luyện
# Giả sử bạn đã lưu mô hình kNN vào file 'knn_model.pkl'
import joblib
knn_model = joblib.load('knn_model.pkl')

import cv2
import os

# Đường dẫn tới file Haar Cascade
face_cascade_path = './haarcascade_frontalface_alt.xml'
eye_cascade_path = './haarcascade_eye_tree_eyeglasses.xml'

# Khởi tạo Haar Cascade Classifiers
if os.path.exists(face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
else:
    st.error("File Haar Cascade cho mặt không tồn tại.")

if os.path.exists(eye_cascade_path):
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
else:
    st.error("File Haar Cascade cho mắt không tồn tại.")

# Hàm trích xuất và phát hiện khuôn mặt
def detect_faces_and_eyes(image):
    # Chuyển đổi ảnh thành màu xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    # Duyệt qua từng khuôn mặt phát hiện
    for (x, y, w, h) in faces:
        # Vẽ hình tròn quanh khuôn mặt
  
        cv2.circle(image, (x + w // 2, y + h // 2), int(0.5 * (w + h) / 2), (255, 0, 0), 2)

        # Vùng khuôn mặt
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Phát hiện đôi mắt
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

        # Duyệt qua các đôi mắt và vẽ hình tròn quanh chúng
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(roi_color, (ex + ew // 2, ey + eh // 2), int(0.3 * (ew + eh) / 2), (0, 255, 0), 2)

    return image

# Giao diện Streamlit
st.title("Ứng dụng phát hiện khuôn mặt")
st.write("Tải lên hình ảnh của bạn để phát hiện khuôn mặt và đôi mắt.")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc và hiển thị ảnh
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Phát hiện khuôn mặt và đôi mắt
    result_image = detect_faces_and_eyes(image)

    # Hiển thị ảnh kết quả
    st.image(result_image, caption='Ảnh với khuôn mặt và mắt được phát hiện', use_column_width=True)
