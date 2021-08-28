"""
Program for scanning the faces via webcam and reads the emotions from facial expressions
to all the people in the webcam.
"""
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array


camera = cv2.VideoCapture(0)


def FER():
    cascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascadePath)
    the_model = keras.models.load_model("FER.h5")
    Emotion_classes = ["Angry", "Fear", "Happy", "Sad", "Suprise", "Neutral"]
    camera = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = camera.read()

        # resize frame and convert it to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frameClone = frame.copy()

        # detect faces in the input frame, then clone the frame so that we can
        # draw on it
        rects = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(rects) > 0:
            for (fX, fY, fW, fH) in rects:
                roi = gray[fY : fY + fH, fX : fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class label
                preds = the_model.predict(roi)[0]
                label = Emotion_classes[preds.argmax()]

                # loop over the labels + probabilities and draw them
                for (i, (emotion, prob)) in enumerate(zip(Emotion_classes, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                    w = int(prob * 300)

                    cv2.putText(
                        frameClone,
                        text,
                        (fX, fY - (20 * i) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frameClone,
                        label,
                        (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        2,
                    )
                    cv2.rectangle(
                        frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2
                    )

        cv2.imshow("Face", frameClone)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FER()
