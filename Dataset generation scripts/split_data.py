import glob
import os

"""
Split the generated images of decks into train and test dataset for YOLO
"""

current_dir = r"D:\darknet\darknet\build\darknet\x64\data\obj"

# Percentage of images to be used for the test set
percentage_test = 10

train_file = open("train.txt", "w")
test_file = open("test.txt", "w")

# Populate train.txt and test.txt with full paths of the images
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        test_file.write(current_dir + "\\" + title + ".jpg" + "\n")
    else:
        train_file.write(current_dir + "\\" + title + ".jpg" + "\n")
        counter += 1
