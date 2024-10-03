from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from keras.models import load_model
import imutils

# Parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.102-0.66.hdf5'

# Loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load the image
# image_path = r'C:\Users\Tandra Debarati\Desktop\Photo\WhatsApp Image 2023-12-20 at 22.19.21_616c38cb.jpg'
# image = cv2.imread(image_path)
image_path= r'C:\Users\Tandra Debarati\Desktop\Stress-detection-Techniques-and-Chat-Bot-Depression-Therapy-master\sad.jpg'
image = cv2.imread(image_path)
# Preprocess the image
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# If no face is detected, just show the image
if len(faces) == 0:
    print("No face detected")
    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Get the largest detected face (if multiple faces are detected)
faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
(fX, fY, fW, fH) = faces

# Extract the region of interest (ROI) of the face and preprocess it
roi = gray[fY:fY + fH, fX:fX + fW]
roi = cv2.resize(roi, (64, 64))
roi = roi.astype("float") / 255.0
roi = img_to_array(roi)
roi = np.expand_dims(roi, axis=0)

# Predict the emotion
preds = emotion_classifier.predict(roi)[0]
emotion_probability = np.max(preds)
emotion_index = preds.argmax()
label = EMOTIONS[emotion_index]

# Calculate stress value (example logic)
negative_emotions = ["angry", "disgust", "scared", "sad"]
stress_value = sum([preds[EMOTIONS.index(emotion)] for emotion in negative_emotions]) * 10

# Define suggestions based on stress value
if label == "happy":
    suggestions = "All good! 😊 Keep smiling!"
elif stress_value > 5:
    suggestions = (
        "Suggested actions to reduce stress:\n"
        "- Try doing yoga or meditation.\n"
        "- Listen to calming music.\n"
        "- Take a walk for a while to relax.\n"
        "- Drink a glass of water and take deep breaths."
    )
else:
    suggestions = "You seem to be relaxed. Keep up the positive vibes!"

# Print results like in the provided picture
print("----------------------------------------------------------")
print("Probabilities of each class:")
print(f" 0:angry , 1:disgust , 2:fear , 3:happy , 4:sad , 5:surprise , 6:neutral")
print(preds)
print("----------------------------------------------------------")
print(f"Emotion class: {label}")
print(f"Maximum probability emotion: {emotion_probability}")
print(f"Stress Value: {stress_value}")
print("----------------------------------------------------------")
print(suggestions)

# Display the results on the image
cv2.putText(image, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

# Show the image with detected emotion and stress level
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
