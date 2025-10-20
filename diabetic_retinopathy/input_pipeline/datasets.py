import os
import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

import input_pipeline.TFRecord as tfr
from input_pipeline.preprocessing import preprocess, augment
from input_pipeline.TFRecord import histogram


@gin.configurable
def load(name, data_dir, classification):

    """
    Load dataset from tfrecord
    Args:
        name: name of the dataset, IDRID or eyepacs
        data_dir: the folder saving tfrecord files
        classification: type of classification
    Return:
        loaded dataset which need to do preparation
    """

    # Get the path of tfrecords
    if name == "IDRID":
        logging.info(f"Preparing dataset {name}...")
        train_file, test_file, val_file = tfr.creating_action(data_dir, classification)

        ds_info = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        # Parse the dataset
        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, ds_info)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
            feature_dict['image'] = preprocess(feature_dict['image']) / 255.0
            return feature_dict['image'], feature_dict['label']

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # Import the train set from TFRecord file
        ds_train = tf.data.TFRecordDataset(train_file)
        ds_train = ds_train.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Import the train set from TFRecord file
        ds_val = tf.data.TFRecordDataset(val_file)
        ds_val = ds_val.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Import the train set from TFRecord file
        ds_test = tf.data.TFRecordDataset(test_file)
        ds_test = ds_test.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Check the class distribution of the labels
        # histogram(labels_num, data_dir)

        # Preprocessing and augmentation
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")

        train_files, val_files, test_files = get_eyepacs_tfrecord(data_dir)
        ds_train = tf.data.TFRecordDataset(train_files)
        ds_val = tf.data.TFRecordDataset(val_files)
        ds_test = tf.data.TFRecordDataset(test_files)

        ds_info = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _preprocess(img_label_dict):
            temp = tf.io.parse_single_example(img_label_dict, ds_info)
            img_raw = tf.io.decode_jpeg(temp['image'], channels=3)
            image = tf.image.resize(img_raw, (256, 256))
            # image = tf.image.resize(image, size=(288, 288))
            label = temp['label']
            return image, label

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def get_eyepacs_tfrecord(data_dir):
    """read the TFRecord files of EyePACS dataset

    Parameters:
        data_dir: original path directory where the data is stored

    Returns:
        train_files: list of TFRecord files for training set
        val_files: list of TFRecord files for validation set
        test_files: list of TFRecord files for test set
    """

    train_files = []
    val_files = []
    test_files = []
    for file in os.listdir(data_dir + '/diabetic_retinopathy_detection/btgraham-300/3.0.0/'):
        if 'train' in file:
            train_files.append([data_dir + '/diabetic_retinopathy_detection/btgraham-300/3.0.0/' + file])
        elif 'validation' in file:
            val_files.append([data_dir + '/diabetic_retinopathy_detection/btgraham-300/3.0.0/' + file])
        elif 'test' in file:
            test_files.append([data_dir + '/diabetic_retinopathy_detection/btgraham-300/3.0.0/' + file])
        else:
            continue
    return train_files, val_files, test_files


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, buffer_size=256, batch_size=16, caching=False):

    '''
    Prepare the dataset, such as batch, augment, prefetch and so on
    Args:
        ds_train: train set
        ds_val: validation set
        ds_test: test set
        ds_info: struction of dataset
        buffer_size: size of shuffle buffer
        batch_size: size of dataset batch
        caching: whether caching or not
    Return:
        ds_train: train set after preparing
        ds_val: validation set after preparing
        ds_test: test set after preparing
        ds_info: struction of dataset
    '''

    logging.info('Preparing the datasets...')
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Prepare training dataset
    ds_train = ds_train.map(lambda image, label: (augment(image), label))
    logging.info('Train images are augmented.')
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(5)
    ds_train = ds_train.prefetch(AUTOTUNE)
    logging.info('Train set is prepared')

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(AUTOTUNE)
    logging.info('Validation set is prepared')

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(AUTOTUNE)
    logging.info('Test set is prepared')

    return ds_train, ds_val, ds_test, ds_info


