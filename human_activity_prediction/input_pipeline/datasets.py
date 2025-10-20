import os
import gin
import logging
import tensorflow as tf
import input_pipeline.TFRecord as tfr


@gin.configurable
def load(name, data_dir, position, label_type):
    '''
    Load dataset from tfrecord
    Args:
        name: name of the dataset, HAPT or HAR
        data_dir: the folder saving tfrecord files
        position: postion of the dataset source
        label_type: type of label, s2s or s2l
    Return:
        loaded dataset which need to do preparation
    '''

    logging.info(f"Preparing dataset {name}...")
    if name == "HAPT":
        logging.info('Creating TFRecord files for HAPT...')
        tfr.creating_action(data_dir, label_type, name, position)
        ds_train = tf.data.TFRecordDataset("tfrecord/HAPT_train.tfrecords")
        ds_val = tf.data.TFRecordDataset("tfrecord/HAPT_val.tfrecords")
        ds_test = tf.data.TFRecordDataset("tfrecord/HAPT_test.tfrecords")

    elif name == "HAR":
        logging.info('Creating TFRecord files for HAR...')
        tfr.creating_action(data_dir, label_type, name, position)
        ds_train = tf.data.TFRecordDataset("tfrecord/HAR_train.tfrecords")
        ds_val = tf.data.TFRecordDataset("tfrecord/HAR_val.tfrecords")
        ds_test = tf.data.TFRecordDataset("tfrecord/HAR_test.tfrecords")

    else:
        raise ValueError

    ds_info = {
        'window': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the dataset
    def _parse_example(example_window):
        feature_dict = tf.io.parse_single_example(example_window, ds_info)
        feature_window = tf.io.parse_tensor(feature_dict['window'], tf.float64)
        label_window = tf.io.parse_tensor(feature_dict['label'], tf.int64)
        return feature_window, label_window

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Map the datasets
    ds_train = ds_train.map(_parse_example, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(_parse_example, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.map(_parse_example, num_parallel_calls=AUTOTUNE)

    # Check the class distribution of the labels
    # histogram(labels_num, data_dir)

    # Preprocessing and augmentation
    return prepare(ds_train, ds_val, ds_test, ds_info)


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
    if caching:
        ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    # ds_train = ds_train.repeat(10)
    ds_train = ds_train.prefetch(AUTOTUNE)
    logging.info('Train set is prepared')

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size, drop_remainder=True)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(AUTOTUNE)
    logging.info('Validation set is prepared')

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size, drop_remainder=True)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(AUTOTUNE)
    logging.info('Test set is prepared')

    return ds_train, ds_val, ds_test, ds_info
