"""
Program for running the Facial Expression Recognition and the Card detection through webcam or recorded video.
The results are saved on text file as states that can be used in Reinforcement Learning.
"""

import os
import cv2
import numpy as np
import darknet
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from scipy import stats as s

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the  GPU
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
cascadePath = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascadePath)
FER_model = keras.models.load_model("FER.h5")
card_model = keras.models.load_model("cards_classification_model.h5")
Emotion_classes = ["Angry", "Fear", "Happy", "Sad", "Suprise", "Neutral"]


def convert_back_from_yolo(x, y, w, h):
    """Convert back from YOLO format"""
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def classify_cards(detections, img, card_model):
    """Cards classification """
    card_index = 0
    cards_deck_state = []
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]

        xmin, ymin, xmax, ymax = convert_back_from_yolo(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        if detection[0] == b"Covered":
            card_tuple = ("Card" + str(card_index), "Covered")
            cards_deck_state.append(card_tuple)
            card_index += 1
        else:
            try:
                roi = img[ymin:ymax, xmin:xmax]
                card_predict = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                card_predict = cv2.resize(card_predict, (100, 100))
                card_predict = card_predict.astype("float") / 255.0
                card_predict = np.expand_dims(card_predict, axis=0)
                card_predict = card_predict[..., np.newaxis]
                predictions = card_model.predict(card_predict)
                label = classes[predictions.argmax()]
                card_tuple = ("Card" + str(card_index), label)
                cards_deck_state.append(card_tuple)
                card_index += 1

            except Exception as e:
                print(str(e))

    return cards_deck_state


def FER(img):
    """Facial Expression Recognition"""
    gray_resized = cv2.resize(
        img,
        (darknet.network_width(netMain), darknet.network_height(netMain)),
        interpolation=cv2.INTER_LINEAR,
    )
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
            predictions = FER_model.predict(roi)[0]
            label = Emotion_classes[predictions.argmax()]
    else:
        label = "Face not detected"

    return label


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

    face_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture("MOV_2400.mp4")

    vid_cod = cv2.VideoWriter_fourcc(*"XVID")
    states_file = open("FER_card_detection_labels.txt", "w")
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(
        darknet.network_width(netMain), darknet.network_height(netMain), 3
    )
    states = []

    while True:
        ret, frame_read = cap.read()
        face_ret, face_read = face_cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        face_frame_rgb = cv2.cvtColor(face_read, cv2.COLOR_BGR2GRAY)

        frame_resized = cv2.resize(
            frame_rgb,
            (darknet.network_width(netMain), darknet.network_height(netMain)),
            interpolation=cv2.INTER_LINEAR,
        )

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        label = FER(face_frame_rgb)
        image = classify_cards(detections, frame_resized, card_model)
        result = label + " " + str(image)
        states.append(result)
        if len(states) == 10:
            most_common_state = s.mode(states)[0]
            states_file.write(str(most_common_state) + "\n")
            states.clear()
        else:
            continue

    cap.release()
    face_cap.release()
    cv2.destroyAllWindows()
    states_file.close()


if __name__ == "__main__":
    YOLO()
