import gin
import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrix
from evaluation.make_new_folder import make_folder
import numpy as np
from evaluation.Grad_CAM import deep_visualize
# from evaluation.dimension_reduction import dimensional_reduction


@gin.configurable
def evaluate(model, ds_test, path, classification):
    """evaluate performance of the model
    Args:
        model: model which will be evaluated
        ds_test: test set
        path: the path to make folder for saving visualization images
        classification: binary or multiple classification type
    """

    # load the checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory="/Users/yinzheming/Desktop/Deep_Learning/Lab/experiments/VGG_binary/run_2023-02-09T22-25-54-084553/ckpts", max_to_keep=10)
    # checkpoint.restore(tf.train.latest_checkpoint(run_paths["path_ckpts_train"]))
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    if checkpoint_manager.latest_checkpoint:
        tf.print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        tf.print("Initializing from scratch.")
    # step = int(checkpoint.step.numpy())

    # Compile the model
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss=tf.keras.losses.BinaryCrossentropy(),
    #               metrics=[ConfusionMatrix()])

    if classification == 'binary':
        y_pred_origin = model.predict(ds_test)
        y_pred = np.where(y_pred_origin > 0.3, 1, 0)
        y_pred = np.ndarray.tolist(y_pred)
        y_pred = [x[0] for x in y_pred]
        print(y_pred)
        print(len(y_pred))

    elif classification == 'multiple':
        y_pred_origin = model.predict(ds_test)
        y_pred = []
        for i in range(y_pred_origin.shape[0]):
            idx = np.argmax(y_pred_origin[i])
            y_pred.append(idx)
        print(y_pred)
        print(len(y_pred))

    elif classification == 'regression':
        y_pred_origin = model.predict(ds_test)
        y_pred = np.ndarray.tolist(y_pred_origin)
        print(y_pred)
        print(len(y_pred))

    # Predict the test set using the label

    # Get the true label list of test set
    y_true = []
    for idx, (test_images, test_labels) in enumerate(ds_test):
        dim = test_labels.shape[0]
        for i in range(dim):
            y_true.append(test_labels[i].numpy())
    print(y_true)
    print(len(y_true))

    print(y_pred_origin)
    if classification == 'binary' or classification == 'multiple':
        # Plot Confusion Matrix
        ConfusionMatrix(y_pred_origin, y_pred, y_true, classification)

        # Make folder for saving visualization image
        make_folder(path)

        # Deep Visualization
        # for idx, (test_image, test_label) in enumerate(ds_test):
        #     deep_visualize(model=model, images=test_image, dataset=ds_test, step=idx, run_paths=path, classification=classification)

        # Dimensional reduction
        # dimensional_reduction(model=model, dataset=ds_test, labels=y_true)

    elif classification == 'regression':
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        print(mse)


    '''# Compute accuracy
    for batch_idx, (test_image, test_label) in enumerate(ds_test):
        batch_result = model.evaluate(test_image, test_label, return_dict=True)
        # predictions = model.predict(test_image[:5])
        # print(predictions)
        for key, value in batch_result.items():
            if key.find('accuracy') != -1 or key.find('loss') != -1:
                batch_result[key] *= test_label.shape[0]
        if batch_idx == 0:
            result = batch_result
        else:
            for key, value in batch_result.items():
                result[key] += batch_result[key]
    ds_test = ds_test.unbatch().batch(1)
    num_test = sum(1 for _ in ds_test)
    for key, value in result.items():
        if (key.find('accuracy') != -1 or key.find('loss') != -1) and num_test != 0:
            result[key] /= num_test

    # Logging test information
    logging.info(f"Evaluating at step: {step}...")
    for key, value in result.items():
        logging.info('{}:\n{}'.format(key, value))'''

    # t_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
    #
    # y_pred = model.predict(ds_test)
    # for test_images, test_labels in ds_test:
    #     predictions = model(test_images, training=False)
    #
    #     test_loss(t_loss)
    #     test_accuracy(test_labels, predictions)
