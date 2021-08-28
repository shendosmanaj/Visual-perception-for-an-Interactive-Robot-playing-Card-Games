"""
Program for running the Facial Expression Recognition and the Card detection through webcam or recorded video.
The result is shown to the user in real time, and saved as a video.
"""

from ctypes import *

# import math
# import random
import os
import cv2
import numpy as np
import darknet
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

netMain = None
metaMain = None
altNames = None
classes = [
    "Big_Ben",
    "Bradenburg_Gate",
    "Colosseo",
    "Duomo",
    "Eiffel",
    "Sagrada_familia",
    "Saint_Basil_Cathedral",
    "Statue_of_liberty",
]

card_model = keras.models.load_model("cards_classification_model.h5")
opt = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
card_model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)

cascadePath = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascadePath)

FER_model = tf.keras.models.load_model("FER.h5")
opt = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
FER_model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)

Emotion_classes = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def convertBack(x, y, w, h):
    """Convert back from YOLO format"""
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cardsDrawBoxes(detections, img, card_model):
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]

        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        if detection[0] == b"Covered":
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(
                img,
                detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]",
                (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                [0, 255, 0],
                2,
            )
        else:
            try:
                roi = img[ymin:ymax, xmin:xmax]
                card_predict = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                card_predict = cv2.resize(card_predict, (100, 100))
                card_predict = card_predict.astype("float") / 255.0
                card_predict = np.expand_dims(card_predict, axis=0)
                card_predict = card_predict[..., np.newaxis]
                preds = card_model.predict(card_predict)
                label = classes[preds.argmax()]
                cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
                cv2.putText(
                    img,
                    str(label),
                    (pt2[0] - 100, pt2[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    [0, 0, 255],
                    2,
                )

            except Exception as e:
                print(str(e))

    return img


def YOLO():
    global metaMain, netMain, altNames
    configPath = "yolov3-tiny-obj.cfg"
    weightPath = "yolov3-tiny-obj_3000.weights"
    metaPath = "obj.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(
            configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1
        )  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re

                match = re.search(
                    "names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE
                )
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # cap = cv2.VideoCapture(0)  # Uncomment this if you want to run it on your webcam
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(
        "MOV_2400.mp4"
    )  # Uncomment this if you want to run it on a video. Change the path to the video first.
    # cap.set(3, 1280)
    # cap.set(4, 720)

    vid_cod = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        "cam_video.avi",
        vid_cod,
        10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)),
    )
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(
        darknet.network_width(netMain), darknet.network_height(netMain), 3
    )
    while True:
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(
            frame_rgb,
            (darknet.network_width(netMain), darknet.network_height(netMain)),
            interpolation=cv2.INTER_LINEAR,
        )

        gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(
            gray,
            (darknet.network_width(netMain), darknet.network_height(netMain)),
            interpolation=cv2.INTER_LINEAR,
        )

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        rects = detector.detectMultiScale(
            gray_resized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(rects) > 0:
            for (fX, fY, fW, fH) in rects:
                roi = gray_resized[fY : fY + fH, fX : fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class label
                preds = FER_model.predict(roi)[0]
                label = Emotion_classes[preds.argmax()]

                # loop over the labels + probabilities and draw them
                for (i, (emotion, prob)) in enumerate(zip(Emotion_classes, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                    # w = int(prob * 300)

                    cv2.putText(
                        frame_resized,
                        text,
                        (fX, fY - (20 * i) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_resized,
                        label,
                        (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 0),
                        2,
                    )
                    cv2.rectangle(
                        frame_resized, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2
                    )

        image = cardsDrawBoxes(detections, frame_resized, card_model)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow("Demo", image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLO()
