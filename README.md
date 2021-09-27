# Visual perception for an Interactive Robot playing Card Games
Implementation of different deep learning techniques in the scenario of learning a robot to play card games.

The objective is to analyze the visual perception of Pepper robot while playing the Memory(also known as Concentration) card game with another person.
The memory game is a card game in which all of the cards are laid face down on a surface and two cards are flipped face up over each turn. The objective of the game is to turn over pairs of matching cards.
The goal for the robot is not to always win the game, but to make it more interesting so people would want to play the game without getting angry or frustrated in one way or another. This will make the robot appear friendlier and more human-like to the people, and not just a machine.
The intention of creating games and quizzes is so that people will interact with the robot, see how it works, and what it is able to do, so next time they will ask questions and information to the robot for other purposes too, and not just for games.
There are two modules that are implemented into the robot for achieving this goal. One module is to detect the deck of cards and classify the objects within the cards. The other model scans for the faces of the people and reads their facial emotions. This will give the robot a feedback of how the people are feeling while playing the game, and what they are feeling at a particular game movement. This feedback then will be used by the robot to choose the next moves.

There are three deep learning models implemented:

* Facial Expression Recognition<br>
trained to classify facial expressions of people who are playing against Pepper robot, in real time.
You can find all the details in the folder Deep Learning Models in Jupyter, in Visual_Sentiment_Analysis notebook.

* Card Detection<br>
The card detection model detects the cards on the table and checks if they are covered or uncovered(face up or down).
This model uses the Tiny YOLO algorithm, used for devices with lower computing power capacity.
The training dataset was generated with the generate_deck_of_cards.py module. You can find it and some sample images
under the folder [Dataset generation scripts](https://github.com/shendosmanaj/Visual-perception-for-an-Interactive-Robot-playing-Card-Games/tree/master/Dataset%20generation%20scripts).
To train this model, the [darknet framework](https://github.com/AlexeyAB/darknet) is used.
This model is used in the module FER_and_card_detection.py, where it does real time predictions.

* Card Classification<br>
This model classifies the uncovered cards of the deck. There are eight classes of cards which represent the uncovered
cards of the deck. There are eight classes of cards which represent monuments from different countries, like: Colosseum,
Duomo di Milano, Statue of Liberty, Eiffel Tower, Big Ben, Bradenburg Gate, Saint Basil Cathedral, Sagrada Familia.
Two different techniques were used: one with traditional CNNs trained on thousands of instances, and the other one is 
few-shot learning, where we have very few instances for each class.
The training dataset was generated with the [generate_individual_cards.py](https://github.com/shendosmanaj/Visual-perception-for-an-Interactive-Robot-playing-Card-Games/blob/master/Dataset%20generation%20scripts/generate_individual_cards.py) module.<br>
You can see the implementation of these two models under [Deep Learning Models in Jupyter](https://github.com/shendosmanaj/Visual-perception-for-an-Interactive-Robot-playing-Card-Games/tree/master/Deep%20Learning%20Models%20in%20Jupyter) folder.

An image with predictions made by robot in real time: <br>
![image1_prediction](https://github.com/shend5/Visual-perception-for-an-Interactive-Robot-playing-Card-Games/blob/master/image1_prediction.jpg)
<br>
A gif with predictions taken from a video can be found [here](https://giphy.com/gifs/Z9nZuWLZ68haS6TGM3).
Predictions are made at 15 FPS.<br>
