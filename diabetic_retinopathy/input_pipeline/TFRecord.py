import logging
import gin
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
from sklearn import model_selection
from input_pipeline.preprocessing import preprocess, augment, to_image, to_csv
import cv2


@gin.configurable
def read_labels(data_dir, dataset, classification):

    """read labels from the csv files and split the train.csv for train set and validation set
    Args:
        data_dir: the directory that stores the data
        dataset: specify which dataset would be read, "train", "test" or "validation"
        classification: specify whether 2-class ("binary") or 5-class ("multiple") classification
    Returns: the labels for the corresponding dataset in a list
    """

    files = pd.read_csv(data_dir + "/labels/train.csv")
    # Drop the rows where all elements are missing
    files.dropna(inplace=True, axis='columns')
    # Convert the column 'Retinopathy grade' to a list
    files_with_labels = files['Retinopathy grade'].values.tolist()
    if dataset == "train":
        # Use the 330 before labels as train set
        train_files_with_labels = files_with_labels[:330]
        histogram(train_files_with_labels)

        return train_files_with_labels

    # Split the validation set from the original train set
    elif dataset == "validation":
        # Use the labels after 330th as validation set
        val_files_with_labels = files_with_labels[330:]
        # If binary, set label 2, 3, 4 as label 1; Otherwise set as label 0
        if classification == 'binary':
            for i in range(len(val_files_with_labels)):
                if val_files_with_labels[i] <= 1:
                    val_files_with_labels[i] = 0
                else:
                    val_files_with_labels[i] = 1
        return val_files_with_labels

    elif dataset == "test":
        files = pd.read_csv(data_dir + "/labels/test.csv")
        files.dropna(inplace=True, axis='columns')
        test_files_with_labels = files['Retinopathy grade'].values.tolist()
        if classification == 'binary':
            for i in range(len(test_files_with_labels)):
                if test_files_with_labels[i] <= 1:
                    test_files_with_labels[i] = 0
                else:
                    test_files_with_labels[i] = 1
        return test_files_with_labels

    else:
        return ValueError

    
@gin.configurable
def resampling(data_dir, dataset, classification):

    '''
    Oversample the imbalanced dataset
    Args:
        data_dir: the path of IDRID file
        dataset: the type of dataset, train, validation or test
        classification: type of classification
    Return:
        the list of new balanced labels
    '''

    # Read the images of train set
    train_dir = data_dir + "/images/train/"
    filenames = [train_dir + filename for filename in os.listdir(train_dir)]
    # Make sure that the images match labels
    filenames.sort(key=lambda x: x[-7:-4])
    filenames = filenames[:330]
    # if len(filenames) == 684:
    #     filenames.pop()
    train_files_with_labels = read_labels(data_dir, dataset, classification)
    label_3_num = 1
    label_4_num = 1
    i = 0

    # Make new folder to save the balanced images
    fold_new = "tfrecord/train_resampling/"
    if os.path.exists(fold_new):
        shutil.rmtree(fold_new)
    os.makedirs(fold_new)

    # For multiple classification, we only copy the images with label 1, 3, 4
    if classification == 'multiple' or classification == 'regression':
        for filename in filenames:
            if train_files_with_labels[i] == 3 and label_3_num <= 7:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=3, i=i)
                i += 8
                label_3_num += 1

            elif train_files_with_labels[i] == 4 and label_4_num <= 8:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=4, i=i)
                i += 8
                label_4_num += 1

            elif train_files_with_labels[i] == 1:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=1, i=i)
                i += 8

            elif train_files_with_labels[i] == 0 or train_files_with_labels[i] == 2:
                file_name_with_jpg = filename.split("train/", 1)[1]
                file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
                filename_new = fold_new + file_name_without_jpg + "0.jpg"
                shutil.copy(filename, filename_new)
                i += 1

            else:
                file_name_with_jpg = filename.split("train/", 1)[1]
                file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
                filename_new = fold_new + file_name_without_jpg + "0.jpg"
                shutil.copy(filename, filename_new)
                i += 1

    # For binary classification, we only copy the images with label 1
    elif classification == 'binary':
        for filename in filenames:
            if train_files_with_labels[i] == 1:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=1, i=i)
                i += 8
            else:
                file_name_with_jpg = filename.split("train/", 1)[1]
                file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
                filename_new = fold_new + file_name_without_jpg + "0.jpg"
                shutil.copy(filename, filename_new)
                i += 1
        print(train_files_with_labels)
        for i in range(len(train_files_with_labels)):
            if train_files_with_labels[i] <= 1:
                train_files_with_labels[i] = 0
            else:
                train_files_with_labels[i] = 1

    else:
        return ValueError
    histogram(train_files_with_labels)
    logging.info('The training dataset is resampled.')
    return train_files_with_labels


@gin.configurable
def prepare_images(data_dir, dataset):
    """read images and define a path for TFRecord file
    Args:
        data_dir: the directory that stores the data
        dataset: specify which dataset would be read, "train", "test" or "validation"
    Returns:
        path: the path for TFRecord file
        filenames: the name of the images for the corresponding dataset
    """

    if dataset == "train":
        train_dir = "tfrecord/train_resampling/"
        # train_dir = data_dir + "/images/train/" # without resampling
        path = "tfrecord/train.tfrecords"
        filenames = [train_dir + filename for filename in os.listdir(train_dir)]
        # filenames.sort(key=lambda x: int(x.split('.')[0][-3:])) # without resampling
        filenames.sort(key=lambda x: x[-8:-4])
        return path, filenames

    if dataset == "validation":
        val_dir = data_dir + "/images/train/"
        path = "tfrecord/val.tfrecords"
        filenames = [val_dir + filename for filename in os.listdir(val_dir)]
        filenames.sort(key=lambda x: x[-7:-4])
        return path, filenames[330:]

    elif dataset == "test":
        test_dir = data_dir + "/images/test/"
        path = "tfrecord/test.tfrecords"
        filenames = [test_dir + filename for filename in os.listdir(test_dir)]
        filenames.sort(key=lambda x: x[-7:-4])
        return path, filenames

    else:
        return ValueError


def histogram(label):
    """Visualize the data distribution
    Args:
        feature:
        label:

    Returns:

    """
    hist = plt.hist(label, bins='auto')
    plt.title("Class distribution")
    plt.xlabel("Retinopathy grade")
    plt.xticks(np.arange(0, 5), fontsize=8)
    plt.ylabel("Numbers")
    plt.tight_layout()
    plt.show()
    return


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@gin.configurable
def write_tfrecord(filenames, labels, path):

    '''
    Create TFRecord file from training or testing set
        filenames: complete path of the file
        labels: the corresponding label of file
        path: path to save tfrecord file
    '''

    logging.info(f"Creating TFRecord file to {path} now...")
    with tf.io.TFRecordWriter(path) as writer:
        for filename, label in zip(filenames, labels):
            image_string = open(filename, 'rb').read()  # read image to RAM in binary mode
            image_shape = tf.io.decode_jpeg(image_string).shape
            feature = {  # build Feature dictionary
                'image': _bytes_feature(image_string),
                'label': _int64_feature(label),
                # 'image_height': _int64_feature(image_shape[0]),
                # 'image_width': _int64_feature(image_shape[1]),
                # 'image_depth': _int64_feature(image_shape[2]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # build Example
            writer.write(example.SerializeToString())
    logging.info("A new TFRecord file is created.")
    return


@gin.configurable
def creating_action(data_dir, classification):

    '''
    Combine resampling, read_labels and write_tfrecord
    Args:
        data_dir: the path saving dataset
        classification: type of classification
    Return:
        three tfrecord files for train set, validation set and test set
    '''

    # Read labels
    train_labels = resampling(data_dir, "train", classification)
    # train_labels = read_labels(data_dir, "train") # without resampling
    logging.info(train_labels)
    val_labels = read_labels(data_dir, "validation", classification)
    test_labels = read_labels(data_dir, "test", classification)

    # Read images
    train_tfrecord_file, train_filenames = prepare_images(data_dir, "train")
    print(len(train_filenames))
    val_tfrecord_file, val_filenames = prepare_images(data_dir, "validation")
    test_tfrecord_file, test_filenames = prepare_images(data_dir, "test")

    # Create TFRecord files
    write_tfrecord(train_filenames, train_labels, train_tfrecord_file)
    write_tfrecord(test_filenames, test_labels, test_tfrecord_file)
    write_tfrecord(val_filenames, val_labels, val_tfrecord_file)
    return train_tfrecord_file, test_tfrecord_file, val_tfrecord_file
