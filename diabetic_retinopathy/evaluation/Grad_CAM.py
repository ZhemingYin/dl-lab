import gin
import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import keras
import matplotlib.cm as cm


@gin.configurable
def deep_visualize(model, images, dataset, step, run_paths, batch_size, classification):

    """
    Connection deep_visualization function with model_for_visualization
    Args:
        model: model which is used for training
        images: Each batch of images in test set
        dataset: the whole test set to get the origin image
        step: the number of the batch
        run_paths: the path to save the Grad CAM images
        batch_size: the size of the batch
        classification: type of deep learning for different ways of predicted labels
    """

    # generate the Grad-CAM images
    grad_cam_model, guided_backprop_model = model_for_visualization(model)
    deep_visualization(images, grad_cam_model, guided_backprop_model, dataset, step, run_paths, batch_size, classification)


@gin.configurable
def model_for_visualization(model):

    """
    Get the model for Grad CAM and Guided Backpropagation
    Args:
        model: model which is used for training
    Return:
        grad_cam_model: model for Grad CAM
        guided_backprop_model: model for Guided Backpropagation
    """

    # extract the corresponding layer result from the model
    grad_cam_model = tf.keras.models.Model([model.input], [model.get_layer('last_conv').output,
                                                           model.get_layer('last_output').output])
    # Implement our own gradient function for the backward pass
    @tf.custom_gradient
    def guided_relu(x):
        def grad(dy):
            return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy
        return tf.nn.relu(x), grad

    # extract the corresponding layer result from the model
    guided_backprop_model = tf.keras.models.Model([model.inputs], [model.get_layer('last_conv').output])

    # change the activation function of layers to guided_relu
    layer_dict = [layer for layer in guided_backprop_model.layers[1:] if hasattr(layer, 'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.layers.ReLU:
            layer.activation = guided_relu

    return grad_cam_model, guided_backprop_model


def deep_visualization(images, grad_cam_model, guided_backprop_model, dataset, step, path, batch_size, classification):

    """
    Calculate heatmap, guided backpropagation, guided grad cam, superimpose them with origin image, and save them
    Args:
        images: Each batch of images in test set
        grad_cam_model: model for Grad CAM
        guided_backprop_model: model for Guided Backpropagation
        dataset: the whole test set to get the origin image
        step: the number of the batch
        path: the path to save the Grad CAM images
        batch_size: the size of the batch
        classification: type of deep learning for different ways of predicted labels
    """

    dim = images.shape[0]
    # process images one by one, not in batch
    for i in range(dim):
        # calculate the gradients of the predictions to the feature maps in the last convolutional layer
        if classification == 'binary':
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_cam_model(images, training=False)
                tape.watch(conv_outputs)
                idx = np.where(predictions[i].numpy()[0] > 0.5, 1, 0)
                # if idx == 1:
                probability = predictions[i]
                # else:
                #     probability = 1 - predictions[i]
            grads = tape.gradient(probability, conv_outputs)
            # print(grads)

        elif classification == 'multiple':
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_cam_model(images, training=False)
                tape.watch(conv_outputs)
                idx = np.argmax(predictions[i])
                probability = predictions[i, idx]
            grads = tape.gradient(probability, conv_outputs)

        # calculate the gradients
        with tf.GradientTape() as tape:
            tape.watch(images)
            outputs = guided_backprop_model(images)
        grads_guided_backpropagation = tape.gradient(outputs, images)[i]
        # plt.imshow(grads_guided_backpropagation)
        # plt.show()
        grads_guided_backpropagation = grads_guided_backpropagation.numpy()

        # calculate the CAM
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_layer_output = conv_outputs.numpy()[i]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        # plt.imshow(heoatmap)
        # plt.show()

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        # heatmap = heatmap.numpy()

        # get the original image
        index = batch_size * step + i
        dataset_unbatch = dataset.unbatch()
        for s, (image, label) in enumerate(dataset_unbatch):
            if s == index:
                # plt.imshow(image)
                # plt.show()
                proto_tensor = tf.make_tensor_proto(image)  # convert `tensor a` to a proto tensor
                img = tf.make_ndarray(proto_tensor)
                break

        plt.imshow(img)
        plt.show()

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[0], img.shape[1]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.3 + img * 300
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

        # normalize the guided backpropagation images
        # grads_guided_backpropagation = tf.maximum(grads_guided_backpropagation, 0) / tf.math.reduce_max(grads_guided_backpropagation)

        grads_guided_backpropagation = (grads_guided_backpropagation - grads_guided_backpropagation.mean()) / \
                                       (grads_guided_backpropagation.std() + 1e-16)
        grads_guided_backpropagation = (grads_guided_backpropagation * 0.25 + 0.5) * 255

        # Superimpose the guided_backpropagation on original image
        superimposed_guided_backpropagation = grads_guided_backpropagation * 0.3 + img * 300
        # superimposed_guided_backpropagation = np.clip(grads_guided_backpropagation, 0, 255).astype('uint8')
        # superimposed_guided_backpropagation = np.uint8(255 * grads_guided_backpropagation)
        guided_backpropagation_img = tf.keras.utils.array_to_img(superimposed_guided_backpropagation)
        plt.imshow(guided_backpropagation_img)
        plt.show()


        # Superimpose the guided_backpropagation with heatmap to create the guided Grad-CAM
        # guided_grad_cam = np.uint8(guided_backpropagation_img * heatmap)
        resized_gradcam = cv2.resize(heatmap, grads_guided_backpropagation.shape[:-1])
        gradcam_r = np.repeat(resized_gradcam[..., None], 3, axis=2)
        guided_grad_cam = superimposed_guided_backpropagation * gradcam_r
        # guided_grad_cam = gradcam_r
        guided_grad_cam_img = tf.keras.utils.array_to_img(guided_grad_cam)
        plt.imshow(guided_grad_cam_img)
        plt.show()

        # Save the superimposed image
        index += 1
        superimposed_img.save(path + '/images/grad_cam/grad_cam_of_image_00' + str(index) + '_with_prediction_'
                              + str(idx) + '.jpg')
        guided_backpropagation_img.save(path + '/images/guided_backpropagation/guided_backpropagation_of_image_00'
                                        + str(index) + '_with_prediction_' + str(idx) + '.jpg')
        guided_grad_cam_img.save(path + '/images/guided_grad_cam/guided_grad_cam_of_image_00' + str(index)
                                 + '_with_prediction_' + str(idx) + '.jpg')
