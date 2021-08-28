"""
This program will be used to generate a dataset of individual cards.
It will be used to train the Neural Network for finding the classes of the uncovered cards.
The dataset will contain images of individual uncovered cards of 8 different classes.
This dataset will be fed as input to a Convolutional Neural Network which will be trained to predict the label
of cards.
"""

import random
from PIL import Image, ImageEnhance

images = [
    "Big_Ben",
    "Bradenburg_Gate",
    "Colosseo",
    "Duomo",
    "Eiffel",
    "Saint_Basil_Cathedral",
    "Statue_of_liberty",
    "Sagrada_familia",
]


def random_card(images):
    """
    Shuffle the cards
    """
    return random.choice(images)


def augment_card(card):
    """
    Rotate the card
    """
    random_rotation = random.randint(-20, 20)
    rotated_card = card.rotate(random_rotation, expand=True)
    mask = Image.new("L", card.size, 255)
    mask = mask.rotate(random_rotation, expand=True)
    return [rotated_card, mask]


def apply_brightness(card):
    """
    Apply brightness to image
    """
    value = random.uniform(0.6, 1.3)
    contrast = ImageEnhance.Brightness(card)
    contrast = contrast.enhance(value)
    return contrast


def card_to_image(card):
    """
    Return the resulting card as an image.
    """
    images_path = "Images\\"
    random_chosen_background = str(random.randint(1, 12))
    table_image = Image.open(f"{images_path}Background {random_chosen_background}.jpeg")
    table_image = table_image.resize((100, 100))
    card_image = Image.open(images_path + card + ".jpeg")
    card_image = card_image.resize((70, 70))
    card_and_mask = augment_card(card_image)
    card_image = card_and_mask[0]
    mask = card_and_mask[1]

    table_image.paste(card_image, mask=mask)
    table_image = apply_brightness(table_image)
    table_image = table_image.convert("L")
    return table_image


# Generate the images, and save them to disk.
for i in range(0, 10000):
    card = random_card(images)
    card_image = card_to_image(card)
    saved_images_path = f"individual_cards_dataset\\{card}\\Image{str(i)}.jpeg"
    card_image.save(saved_images_path)
