import logging
import gin
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import glob
from scipy import interpolate
import matplotlib.pyplot as plt
from input_pipeline.data_visualization import plot

labeling = {1: 'climbingdown', 2: 'climbingup', 3: 'jumping', 4: 'lying',
                5: 'running', 6: 'sitting', 7: 'standing', 8: 'walking'}
positions = {1: 'chest', 2: 'forearm', 3: 'head', 4: 'shin', 5: 'thigh', 6: 'upperarm', 7: 'waist'}


@gin.configurable
def read_labels(data_dir):
    """Read the labels of raw data for HAPT
    Args:
        data_dir: the data directory
    Returns:
        parsed_labels: the labels are parsed into arrays
    """

    # Read labels from the file and parse them according to the experiment id
    labels = pd.read_csv(data_dir + "/RawData/labels.txt", sep=' ', header=None)
    labels_list = labels.values.tolist()
    parsed_labels = [None] * 61
    for exp_id in range(1, 62):
        idx_labels = []
        for label in labels_list:  # Combine the labels with the same experiment id together
            if label[0] == exp_id:
                idx_labels.append(label)
        parsed_labels[exp_id - 1] = np.array(idx_labels, dtype=object)
    parsed_labels = np.array(parsed_labels, dtype=object)
    return parsed_labels


@gin.configurable
def har_preprocess(data_dir):
    """preprocess the raw data and create new files to remove the incorrect alignment and noise for HAR
    Args:
        data_dir: position of raw data
    Returns:
        new csv files according to the subjects and the activities
    """

    subject = {'train': [1, 2, 5, 8, 11, 12, 13, 15], 'val': [3], 'test': [9, 10]}
    data_path = os.path.join(data_dir + '/proband{}/data')
    if not os.path.exists(os.getcwd() + "/har_dataset"):
        os.mkdir(os.getcwd() + "/har_dataset")
        logging.info("A new directory for csv files is created")
    save_path = os.getcwd() + "/har_dataset"
    sensors = ['Gyroscope', 'acc']

    # Create new csv files that integrating each feature data of one activity for each subject
    for ds in subject.keys():
        for num in subject[ds]:
            # Read all .csv files in the subject{num} and sort them
            files = glob.glob(data_path.format(num) + '/*.csv')
            files.sort()
            for activity in labeling.values():
                # Get the range of time stamp
                start = 0
                end = float('inf')
                # Create a dictionary to store the functions correspondingly
                data_dict = {}
                # Create a list to store all feature name, e.g., x, y, z...
                feature_name = []
                # Name of new csv files
                filename = os.path.join(save_path, ds + '_' + str(num) + '_' + activity + '.csv')
                if os.path.isfile(filename):
                    logging.info(f'{filename} already exists')
                    continue
                else:
                    pass
                for file in files:
                    for sensor in sensors:
                        if (sensor in file and activity in file) is True:
                            index = str(num) + '_' + os.path.basename(file).split('.')[0]
                            csv_df = pd.read_csv(file)
                            position = index.split('_')[-1]
                            title = [position + '_' + sensor + '_' + a for a in csv_df.columns.tolist()[2:]]

                            # Using
                            feature = csv_df.to_numpy()[1:, 1:]
                            fun = interpolate.interp1d(feature[:, 0], feature, axis=0, kind='nearest')
                            data_dict[index] = fun

                            # Define the smallest time range
                            start = int(feature[0, 0]) if int(feature[0, 0]) > start else start
                            end = int(feature[-1, 0]) if int(feature[-1, 0]) < end else end
                            feature_name += title

                # Interval is 20 since the frequency of sensors is 50Hz
                time_range = np.arange(start + 5e3, end - 5e3, 20)
                data = []
                for idx, fun in data_dict.items():
                    data_interp = fun(time_range)
                    data.append(data_interp[:, 1:])
                data = np.concatenate(data, axis=-1)
                data = zscore(data, axis=0)

                data = pd.DataFrame(data, index=time_range, columns=feature_name)
                data.to_csv(filename, index=True)
                logging.info(f'{filename} is created')
            logging.info(f'Subject {num} is preprocessed')
        logging.info(f'The {ds} set is preprocessed')
    logging.info('All csv files are created')


@gin.configurable
def read_files(data_dir, name, position, multiple=False):
    """Read files from HAPT or HAR dataset
    Args:
        data_dir: the data directory
        name: choose the dataset to read, HAPT or HAR
    Returns:
        features and labels of train, test and validation datasets
    """

    train_feature, val_feature, test_feature = [], [], []
    train_label, val_label, test_label = [], [], []

    if name == "HAPT":
        # Read files separately
        acc_filenames = [data_dir + "/RawData/" + filename for filename in os.listdir(data_dir + "/RawData/") if
                         filename.startswith("acc")]
        acc_filenames.sort()
        gyro_filenames = [data_dir + "/RawData/" + filename for filename in os.listdir(data_dir + "/RawData/") if
                          filename.startswith("gyro")]
        gyro_filenames.sort()

        # Read data according to the label sequence
        # The data are read per experiment
        for acc_file, gyro_file, exp_id in zip(acc_filenames, gyro_filenames, range(1, 62)):
            acc_data = pd.read_csv(acc_file, sep=" ", header=None).values
            gyro_data = pd.read_csv(gyro_file, sep=" ", header=None).values
            activity_data = np.hstack((np.asarray(zscore(acc_data, axis=0)), np.asarray(zscore(gyro_data, axis=0))))
            parsed_label = read_labels(data_dir)

            # Create an all-zero list and fill it up with the corresponding labels
            activity_label = [0] * len(activity_data)
            for label in parsed_label[exp_id - 1]:
                log_start = label[3]
                log_end = label[4]
                activity = label[2]
                activity_label[log_start: log_end + 1] = [activity] * (log_end + 1 - log_start)

            # Store the activity_label into a list
            if exp_id in range(1, 44):
                train_feature.append(activity_data)
                train_label.append(activity_label)
            elif exp_id in range(56, 62):
                val_feature.append(activity_data)
                val_label.append(activity_label)
            elif exp_id in range(44, 56):
                test_feature.append(activity_data)
                test_label.append(activity_label)

    elif name == "HAR":
        har_preprocess(data_dir)
        files = [file for file in os.listdir(os.getcwd() + "/har_dataset")]
        files.sort()

        for num, name in enumerate(files):
            logging.info(f'Reading {name} ...')

            # Load labels from labeling dictionary
            name_s = name.split('.')[0]
            sub, activity = name_s.split('_')[1], name_s.split('_')[-1]
            for labels, act in labeling.items():
                if act == activity:
                    label = labels
                    break

            # Read features from csv files
            data = pd.read_csv(os.getcwd() + "/har_dataset/" + name)

            # Single or multiple position recognition
            if multiple is False:
                position_data = []
                if len(data.columns) != 43:
                    if all(positions[position] not in feature_name for feature_name in data.columns):
                        logging.info(f'{name} is lack of sensor data, it is ignored')
                        continue
                for feature_name in data.columns:
                    if positions[position] in feature_name:
                        column_data = np.array(data[feature_name])
                        position_data.append(column_data)
                position_data = np.stack(position_data, axis=-1)
            else:
                position_data = []
                if len(data.columns) != 43:
                    logging.info(f'{name} is skipped since lack of sensor data')
                    continue
                for feature_name in data.columns[1:]:
                    for pos in positions.values():
                        if pos in feature_name:
                            column_data = np.array(data[feature_name])
                            position_data.append(column_data)
                position_data = np.stack(position_data, axis=-1)

            # To fit the plot function
            for i in range(3):
                position_data[:, [i, i + 3]] = position_data[:, [i + 3, i]]

            # Concatenate the features and labels
            if "train" in name:
                train_feature.append(position_data)
                train_label.append(len(position_data) * [label])
                if label == 3:
                    logging.info("Oversampling the imbalanced class...")
                    for i in range(5):
                        train_feature.append(position_data)
                        train_label.append(len(position_data) * [label])
            elif "val" in name:
                val_feature.append(position_data)
                val_label.append(len(position_data) * [label])
            elif "test" in name:
                test_feature.append(position_data)
                test_label.append(len(position_data) * [label])
    else:
        raise ValueError

    # Store the features and labels as arrays
    train_feature = np.concatenate(train_feature)
    train_label = np.concatenate(train_label)
    val_feature = np.concatenate(val_feature)
    val_label = np.concatenate(val_label)
    test_feature = np.concatenate(test_feature)
    test_label = np.concatenate(test_label)

    return train_feature, train_label, val_feature, val_label, test_feature, test_label


@gin.configurable
def window_sliding(feature, label, label_type, window_size, shift_ratio, name, ds_test=False):
    """Generate dataset for window sliding
    Args:
        feature: parsed feature from raw data
        label: parsed labels from raw data
        window_size: length of sequence window
        window_shift: overlapped length of sequence window
        ds_test: for test dataset, there is no overlap between windows
    Returns:
        ds: FlatMap dataset with window sliding
    """

    # Batch for creating feature windows and label windows
    def _sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    # Window sliding for each dataset
    ds = tf.data.Dataset.from_tensor_slices((feature, label))
    if ds_test:
        ds = ds.window(size=window_size, shift=window_size, stride=1, drop_remainder=True)
    else:
        ds = ds.window(size=window_size, shift=int(window_size*shift_ratio), stride=1, drop_remainder=True)
    ds = ds.flat_map(lambda feature_window, label_window:
                     _sub_to_batch(tf.data.Dataset.zip((feature_window, label_window))))
    # ds = ds.map(lambda feature_window, label_window: (feature_window, tf.cast(tf.math.argmax(tf.math.bincount(tf.cast(label_window, tf.int32))), tf.int64)))
    if name == "HAPT":
        if label_type == 's2l':
            ds = ds.map(lambda feature_window, label_window: (feature_window, tf.cast(tf.math.argmax(tf.math.bincount(tf.cast(label_window, tf.int32))), tf.int64)))
        else:
            pass
        ds = ds.map(lambda feature_window, label_window: (feature_window, tf.cast(tf.one_hot(tf.subtract(label_window, 1), 12), tf.int64)))
    elif name == "HAR":
        if label_type == 's2l':
            ds = ds.map(lambda feature_window, label_window: (feature_window, tf.cast(tf.math.argmax(tf.math.bincount(tf.cast(label_window, tf.int32))), tf.int64)))
        else:
            pass
        ds = ds.map(lambda feature_window, label_window: (feature_window, tf.cast(tf.one_hot(tf.subtract(label_window, 1), 8), tf.int64)))
    else:
        raise ValueError

    return ds


def oversample_data(features, labels):
    """oversample the imbalanced data
    Args:
        features: the feature that wanted to be augmented (usually train_feature)
        labels: the corresponding labels of the features
    Returns:
        oversampled_features: return the features for creating the dataset
        oversampled_labels: return the labels for creating the dataset
    """

    # Count occurrences of elements in labels
    label_count = [None] * 12
    for i in range(1, 13):
        label_count[i - 1] = np.count_nonzero(labels == i)
    sample_amount = np.mean(np.array(label_count[:6]))

    # Iterate over postural transition labels and add them to the feature set
    for i in range(7, 13):
        repeat_feature = []
        repeat_label = []
        for j in range(len(labels)):
            if labels[j] == i:
                repeat_label.append(labels[j])
                repeat_feature.append(features[j])

        oversampling_features = np.stack(repeat_feature, axis=0)
        oversampling_labels = np.array(repeat_label)
        repeat_rate = 2
        while oversampling_labels.shape[0] + len(repeat_label) < sample_amount:
            oversampling_labels = np.tile(np.array(repeat_label), repeat_rate)
            oversampling_features = np.tile(np.stack(repeat_feature), (repeat_rate, 1))
            repeat_rate += 1

        features = np.concatenate((features, oversampling_features), axis=0)
        labels = np.concatenate((labels, oversampling_labels), axis=0)

    return features, labels


def sort_data(features, labels):
    """sort out all unlabelled data for HAPT
    Args:
        features: features that need to be preprocessed
        labels: the corresponding labels for the features
    Returns:
        sorted features and labels
    """
    idx = np.argwhere(labels == [0])
    sorted_feature = np.delete(features, idx, axis=0)
    sorted_label = np.delete(labels, idx, axis=0)
    return sorted_feature, sorted_label


@gin.configurable
def write_tfrecord(dataset, path):
    """Write TFRecord files
    Args:
        dataset: created dataset with windows
        path: the directory to save the TFRecord files
    Returns:
        created TFRecord files
    """

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    logging.info(f"Creating TFRecord file to {path} now...")
    with tf.io.TFRecordWriter(path) as writer:
        for feature_window, label_window in dataset:
            # Serialization
            feature_window = tf.io.serialize_tensor(feature_window)
            label_window = tf.io.serialize_tensor(label_window)
            feature = {  # build Feature dictionary
                'window': _bytes_feature(feature_window),
                'label': _bytes_feature(label_window),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # build Example
            writer.write(example.SerializeToString())
    return


@gin.configurable
def creating_action(data_dir, label_type, name, position=None, sortout=True, oversampling=False):
    """Create TFRecord files
    Args:
        data_dir: the path of the stored data set
        name: specify the dataset to be trained, HAPT or HAR
        position: default set to None, only for HAR
        sortout: default set to True, only for HAPT
        oversampling: default set to False, the behavior is not satisfactory
    Returns:
        TFRecord files for train, test and validation
    """
    # Get original data
    train_feature, train_label, val_feature, val_label, test_feature, test_label = read_files(data_dir, name, position)
    histogram(train_feature, train_label)

    # Delete unlabelled data from all datasets
    if sortout:
        train_feature, train_label = sort_data(train_feature, train_label)
        val_feature, val_label = sort_data(val_feature, val_label)
        test_feature, test_label = sort_data(test_feature, test_label)
        logging.info("All unlabelled data are sorted out")

    # Oversampling the train data
    if oversampling:
        train_feature, train_label = oversample_data(train_feature, train_label)
        train_feature = zscore(train_feature, axis=0)
        histogram(train_feature, train_label)
        logging.info("The dataset is oversampled")
    else:
        logging.info("The dataset is not oversampled")

    # Create the directory for saving TFRecord files
    if not os.path.exists(os.getcwd() + "/tfrecord"):
        os.mkdir(os.getcwd() + "/tfrecord")
        logging.info("A new directory for TFRecord files is created")

    # Load the datasets
    ds_train = window_sliding(train_feature, train_label, name=name, label_type=label_type)
    ds_val = window_sliding(val_feature, val_label, name=name, label_type=label_type)
    ds_test = window_sliding(test_feature, test_label, name=name, label_type=label_type, ds_test=True)

    # Write TFRecord files
    if name == "HAPT":
        plot(name, None, train_feature, train_label, input="raw_data")
        write_tfrecord(ds_train, "tfrecord/HAPT_train.tfrecords")
        write_tfrecord(ds_test, "tfrecord/HAPT_test.tfrecords")
        write_tfrecord(ds_val, "tfrecord/HAPT_val.tfrecords")
        logging.info("New TFRecord files for HAPT are created")
    elif name == "HAR":
        plot(name, None, train_feature, train_label, input="raw_data", plot_range=50000)
        write_tfrecord(ds_train, "tfrecord/HAR_train.tfrecords")
        write_tfrecord(ds_test, "tfrecord/HAR_test.tfrecords")
        write_tfrecord(ds_val, "tfrecord/HAR_val.tfrecords")
        logging.info("New TFRecord files for HAR are created")
    else:
        raise ValueError

    return


def histogram(feature, label):
    """Visualize the data distribution
    Args:
        feature:
        label:

    Returns:

    """
    _, train_label = sort_data(feature, label)
    # labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
    #           'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
    hist = plt.hist(train_label, bins='auto')
    plt.title("Class distribution")
    plt.xlabel("Activities")
    plt.xticks(np.arange(1, 9), labels=list(labeling.values()), fontsize=8, rotation=45)
    plt.ylabel("Numbers")
    plt.tight_layout()
    plt.show()
    return