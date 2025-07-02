import cv2
import numpy as np
from keras.models import load_model
import time

from utils.datasets import get_labels
from utils.inference import (
    draw_text,
    draw_bounding_box,
)
from utils.preprocessor import preprocess_input

def run_webcam_emotion_and_gender_detection():
    # Load models
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    
    # Load model for gender detection
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'  # Path to your gender model file
    gender_classifier = load_model(gender_model_path, compile=False)

    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_labels = get_labels('fer2013')
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # Load face detection model
    face_detection = cv2.CascadeClassifier(detection_model_path)

    # Membuka webcam
    cap = cv2.VideoCapture(0)  # 0 = kamera default

    if not cap.isOpened():
        print("Tidak bisa membuka kamera.")
        return

    print("🎥 Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

    # Menyimpan hasil deteksi
    result_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Membalik gambar secara horizontal

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dengan haarcascade
        faces = face_detection.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_coordinates = (x, y, x + w, y + h)

            # Resize wajah dan preprocess untuk emosi
            gray_face = gray_image[y:y + h, x:x + w]
            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # Prediksi emosi
            start_time = time.time()  # Mulai waktu
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_score = float(np.max(emotion_prediction))

            # Preprocessing wajah untuk gender (convert ke RGB dan resize)
            face_rgb = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            face_rgb_resized = cv2.resize(face_rgb, (48, 48))  # Resize ke (48, 48, 3)
            face_rgb_resized = np.expand_dims(face_rgb_resized, axis=0)

            # Prediksi gender
            gender_prediction = gender_classifier.predict(face_rgb_resized)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = "Male" if gender_label_arg == 0 else "Female"  # 0: Male, 1: Female
            gender_score = float(np.max(gender_prediction))  # Confidence score

            # Hitung waktu deteksi
            elapsed_time = time.time() - start_time

            # Menampilkan bounding box dan teks emosi dan gender
            color = (0, 255, 0)
            draw_bounding_box(face_coordinates, frame, color)
            draw_text(face_coordinates, frame, f"{emotion_text} ({emotion_score:.2f})", color, 0, -30, 1, 2)
            draw_text(face_coordinates, frame, f"Gender: {gender_text} ({gender_score:.2f})", color, 0, -50, 1, 2)

            # Simpan hasil deteksi dalam log
            result_log.append({
                "Frame": len(result_log) + 1,
                "Emotion": emotion_text,
                "Gender": gender_text,
                "Processing Time (ms)": elapsed_time * 1000  # Mengonversi ke milidetik
            })

        cv2.imshow('Real-Time Emotion and Gender Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Setelah keluar dari loop, tampilkan hasil deteksi
    print("\nHasil Deteksi:")
    print(f"{'Frame':<6}{'Emotion':<10}{'Gender':<10}{'Processing Time (ms)':<20}")
    for result in result_log:
        print(f"{result['Frame']:<6}{result['Emotion']:<10}{result['Gender']:<10}{result['Processing Time (ms)']:<20}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_emotion_and_gender_detection()
