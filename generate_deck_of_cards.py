"""
Program for generating a dataset with decks of cards.
This dataset is used for training the model for card detection.
The cards are saved together with their labels and the bounding boxes.
"""

import numpy as np
import random
from PIL import Image, ImageEnhance

images = ["Big_Ben", "Bradenburg_Gate", "Colosseo", "Duomo", "Eiffel", "Saint_Basil_Cathedral",
          "Statue_of_liberty", "Sagrada_familia"]


def shuffled_cards(images, m=4, n=4):
    """
    Make two pairs of each card, and shuffle them.
    """
    images = np.repeat(images, 2)
    random.shuffle(images)
    return images.reshape((m, n))


def cards_to_remove_indices(images):
    """
    Choose card at random to be removed from the deck.
    """
    removed_cards = random.choice(random.choice(images))
    removed_cards_indices = [(i, j) for i, row in enumerate(images) for j, elem in enumerate(row) if
                             elem == removed_cards]
    return removed_cards_indices


def chosen_cards_indices(images, m=4, n=4):
    """
    Choose two cards to open(uncover)
    """
    row_indexes = [i for i in range(0, m)]
    col_indexes = [i for i in range(0, n)]

    rows_chosen = random.choices(row_indexes, k=2)
    cols_chosen = random.choices(col_indexes, k=2)

    first_card = tuple([rows_chosen[0], cols_chosen[0]])
    second_card = tuple([rows_chosen[1], cols_chosen[1]])
    return [first_card, second_card]


def augment_card(card):
    """
    Rotate the card
    """
    random_rotation = random.randint(-20, 20)
    rotated_card = card.rotate(random_rotation, expand=True)
    mask = Image.new('L', card.size, 255)
    mask = mask.rotate(random_rotation, expand=True)
    return [rotated_card, mask]


def data_yolo_format(x, y, card_width, card_height, table_width, table_height):
    """
    Convert our data to YOLO format for bounding boxes of cards.
    """
    x_center = (x + (card_width / 2)) / table_width
    y_center = (y + (card_height / 2)) / table_height
    width_final = card_width / table_width
    height_final = card_height / table_height
    return [x_center, y_center, width_final, height_final]


def brightness(table):
    """
    Apply brightness to the image
    """
    value = random.uniform(0.95, 1.1)
    contrast = ImageEnhance.Brightness(table)
    contrast = contrast.enhance(value)
    return contrast


def cards_to_image(cards, removed_cards_indices, chosen_cards_indices):
    """
    Save the generated deck into an image.
    Find and save the corresponding bounding boxes of cards into txt files, using YOLO format.
    """
    images_path = "Images\\"
    table_image = Image.open(images_path + "Background 11.jpeg")
    table_height, table_width = table_image.size
    chosen_card_1, chosen_card_2 = chosen_cards_indices[0], chosen_cards_indices[1]
    labels = ""

    for i in range(0, len(cards)):
        for j in range(0, len(cards[i])):
            if (i, j) not in removed_cards_indices:
                if (i, j) not in chosen_cards_indices:
                    card_image = Image.open(images_path + "Card.jpeg")
                    card_type = "0"
                else:
                    card_image = Image.open(images_path + cards[i][j] + ".jpeg")
                    card_type = "1"

                card_image = card_image.resize((153, 150))
                card_and_mask = augment_card(card_image)
                card_image = card_and_mask[0]
                mask = card_and_mask[1]

                area = ((210 * j) + 30, (210 * i) + 30)
                table_image.paste(card_image, box=area, mask=mask)
                card_height, card_width = card_image.size
                x, y = area
                x_center, y_center, width, height = data_yolo_format(x, y, card_width, card_height, table_width,
                                                                     table_height)
                labels += card_type + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(
                    height) + "\n"
                table_image = brightness(table_image)
    return table_image, labels


# Generate images of decks of cards, and save them to disk.
for i in range(0, 10000):
    cards = shuffled_cards(images)
    removed_cards = cards_to_remove_indices(cards)
    chosen_cards = chosen_cards_indices(cards)
    save_directory = "dataset_table_cards\\images\\"
    final_deck, labels_of_deck = cards_to_image(cards, removed_cards, chosen_cards)
    final_deck.save(save_directory + "image_" + str(i) + ".jpg")
    f = open("dataset_table_cards\\labels\\image_" + str(i) + ".txt", "w")
    f.write(labels_of_deck)
