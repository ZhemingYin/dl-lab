import gin
import tensorflow as tf
import shutil
import os
import tensorflow_addons as tfa
import numpy as np


def crop_margin(img):
    img = tf.make_ndarray(img)
    img2 = img.sum(axis=2)
    (row, col) = img2.shape
    row_top = 0
    raw_down = 0
    col_top = 0
    col_down = 0
    for r in range(0, row):
        if img2.sum(axis=1)[r] > 20 * col:
            row_top = r
            break

    for r in range(row - 1, 0, -1):
        if img2.sum(axis=1)[r] > 20 * col:
            raw_down = r
            break

    for c in range(0, col):
        if img2.sum(axis=0)[c] > 20 * row:
            col_top = c
            break

    for c in range(col - 1, 0, -1):
        if img2.sum(axis=0)[c] > 20 * row:
            col_down = c
            break

    new_img = img[row_top:raw_down + 1, col_top:col_down + 1, 0:3]
    new_img = tf.convert_to_tensor(new_img)
    return new_img


def to_image(filename, path):

    '''
    Copy the images to do oversampling
    Args:
        filename: the complete path of the image which will be copied
        path: the new folder to save new images
    '''

    for i in range(0, 8, 1):
        file_name_with_jpg = filename.split("train/", 1)[1]
        file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
        filename_new = path + file_name_without_jpg + str(i) + ".jpg"
        shutil.copy(filename, filename_new)
    return


def to_csv(list, label, i):

    '''
    Insert the corresponding labels of the copied images
    Args:
        list: list of the true labels
        label: the label of the copied image
        i: the position of the label in list
    :return:
    '''

    for j in range(7):
        list.insert(i+j, label)
    return


@gin.configurable
def preprocess(image):

    '''
    Dataset preprocessing: Normalizing and resizing
    Args:
        image: the image which will be preprocessed
    Return:
        preprocessed image
    '''

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32)

    # Resize image
    image = tf.expand_dims(image, axis=0)
    image = tf.image.crop_and_resize(
        image,
        [[-280/2848, 290/4288, 3128/2848, 3698/4288]],
        box_indices=[0],
        crop_size=(256, 256))
    image = tf.squeeze(image)

    return image


def augment(image):

    '''
    Data Augmentation
    Args:
        image: the image which will be augmented
    Return:
        the augmented image
    '''

    image = tf.image.random_contrast(image, lower=0.1, upper=1.0)
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tfa.image.rotate(image, tf.random.uniform(shape=(1,), maxval=180, minval=0, dtype=np.float32))
    return image
