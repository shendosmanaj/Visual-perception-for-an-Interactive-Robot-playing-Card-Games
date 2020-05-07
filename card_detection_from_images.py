"""
Program for running the Card detection in an image.
A new image will be created together with the predicted values.
"""

from ctypes import *
import os
import cv2
import numpy as np
import darknet
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

netMain = None
metaMain = None
altNames = None
classes = ["Big_Ben", "Bradenburg_Gate", "Colosseo", "Duomo", "Eiffel", "Sagrada_familia",
           "Saint_Basil_Cathedral", "Statue_of_liberty"]


def convertBack(x, y, w, h):
    """Convert back from YOLO format"""
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cardsDrawBoxes(detections, img, card_model):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]

        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        if detection[0] == b'Covered':
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
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
                cv2.putText(img,
                            str(label),
                            (pt2[0] - 100, pt2[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            [0, 0, 255], 2)
            # print(label)

            except Exception as e:
                print(str(e))

    return img

def YOLO():
    global metaMain, netMain, altNames
    configPath = "yolov3-tiny-obj.cfg"
    weightPath = "yolov3-tiny-obj_3000.weights"
    metaPath = "obj.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
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


    cap = cv2.imread("image1.jpg")   # Write the path to your image here.

    card_model = keras.models.load_model("cards_classification_model.h5")
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    while True:
        frame_rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

        
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (darknet.network_width(netMain),
                                         darknet.network_height(netMain)),
                                  interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        image = cardsDrawBoxes(detections, frame_resized, card_model)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite("image1_prediction.jpg", image)
        cv2.imshow('Demo', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLO()
