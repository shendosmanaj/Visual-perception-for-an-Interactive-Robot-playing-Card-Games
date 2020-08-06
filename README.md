# Visual perception for an Interactive Robot playing Card Games
Implementation of different deep learning techniques in the scenario of learning a robot to play card games.

An image with predictions made by robot in real time:
![image1_prediction](https://github.com/shend5/Visual-perception-for-an-Interactive-Robot-playing-Card-Games/blob/master/image1_prediction.jpg)

A gif with predictions taken from a video. Predictions are made at 15 FPS.
![github](https://media.giphy.com/media/Z9nZuWLZ68haS6TGM3/giphy.gif)


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
