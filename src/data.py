from .images import load, normalize, resize
from .noise import random_jitter


def load_image_train(image_file, type='RtoL', IMG_HEIGHT=256, IMG_WIDTH=256):
    left, right = load(image_file)
    left, right = random_jitter(left, right, IMG_HEIGHT, IMG_WIDTH)
    left, right = normalize(left, right)

    # i switched these
    if type == 'LtoR':
        return left, right
    else:
        return right, left


def load_image_test(image_file, type='RtoL', IMG_HEIGHT=256, IMG_WIDTH=256):
    left, right = load(image_file)
    left, right = resize(left, right, IMG_HEIGHT, IMG_WIDTH)
    left, right = normalize(left, right)

    if type == 'LtoR':
        return left, right
    else:
        return right, left
