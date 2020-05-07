"""
Facial Expression Recognition from a given image.
"""

from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
the_model = keras.models.load_model("FER.h5")
Emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Suprise", "Neutral"]

# CHANGE THIS PATH TO YOUR IMAGE.
imagePath = r"image1.jpg"

while True:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in faces:

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = the_model.predict(roi)[0]
        label = Emotion_classes[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(Emotion_classes, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label + probability bar on the canvas
            w = int(prob * 300)
            cv2.putText(image, text, (fX, fY - (20 * i) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(image, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)

    cv2.imshow("IMAGE", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

status = cv2.imwrite('faces_detected.jpg', image)