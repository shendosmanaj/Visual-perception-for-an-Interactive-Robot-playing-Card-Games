# Visual perception for an Interactive Robot playing Card Games
Implementation of different deep learning techniques in the scenario of learning a robot to play card games.

To see a demo check the files: video cam_video.avi and image1_prediction.jpg.
To see the saved results check the file: FER_card_detection_labels.txt


1.FER_and_card_detection.py - doing card detection and facial expression recognition in the same time.
The result will be shown to you in real time in an opened window the predictions and the bounding boxes, and so on.
One video will be generated and saved when the program is finished.(See the video prediction.avi as an example).<br>
2. FER_card_detection_labels.py - does same thing as the one above, but instead of saving a video, it will save a text
file with all the results during the prediction. The results are saved as states that could be used to train
a Reinforcement Learning agent.<br>
3.facial_expression_recognition_from_images.py - Takes an image as input, will detect the face and do a prediction about
facial expression.<br>
4.facial_expression_recognition.py - does the same as the one above but in videos. A video will be saved as a result.
(see the video named FER_detection.avi as an example).<br>
5.Card_detection.py - will detect all the cards on the table. Takes a video as input.<br>
6.card_detection_from_images.py -  detect cards on table from an image.<br>
7.generate_deck_of_cards.py - create a dataset for training the tiny yolo card detection model. It will generate images
of decks of cards together with corresponding bounding boxes saved in text files.(See folder: dataset_table_cards).<br>
8.generate_individual_cards.py - create a dataset of individual cards, used for creating a model to predict the classes
of each card.<br>
9.darknet.py - the library used.<br>
10. split_data.py - Split the generated images of decks into train and test dataset for YOLO

The dll files are files that are needed for darknet framework to work.
This framework is using opencv and cuda 10.2
If you want to modify the detection model to detect other types of objects, you can follow this link:
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
