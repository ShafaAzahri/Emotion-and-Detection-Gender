import sys
import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input


def process():
    # parameters for loading data and images
    image_path = sys.argv[1]
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'

    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    gender_offsets = (30, 60)
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    gender_classifier = load_model(gender_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]

    # loading images
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)

    emotions_detected = []

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        # Gender prediction
        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        # Emotion prediction
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        emotions_detected.append(emotion_text)

        # Set color based on gender
        if gender_text == gender_labels[0]:
            color = (0, 0, 255)  # Red for female
        else:
            color = (255, 0, 0)  # Blue for male

        # Draw results
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

    # Convert and save result
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('../images/predicted_test_image.png', bgr_image)

    # Return most frequent emotion
    if emotions_detected:
        most_frequent_emotion = max(set(emotions_detected), key=emotions_detected.count)
        return emotions_detected, most_frequent_emotion
    else:
        return [], "No face detected"


def load_image(image_path, grayscale=False, target_size=None):
    from keras.preprocessing.image import load_img
    # Handle color_mode instead of grayscale parameter
    color_mode = 'grayscale' if grayscale else 'rgb'
    pil_image = load_img(image_path, color_mode=color_mode, target_size=target_size)
    return np.array(pil_image)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_emotion_classifier.py <image_path>")
        sys.exit(1)

    emotions, most_frequent = process()
    print(f"Detected emotions: {emotions}")
    print(f"Most frequent emotion: {most_frequent}")
    print("Result saved to: ../images/predicted_test_image.png")
