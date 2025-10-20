
import os
import shutil
import gin


@gin.configurable
def make_folder(path):

    """
    make new folder to save the images of grad cam, guided backpropagation, guided grad cam
    Args:
        path: the path of the new folder
    """

    # make folder for grad cam
    grad_cam_path = path + "/images/grad_cam/"
    if os.path.exists(grad_cam_path):
        shutil.rmtree(grad_cam_path)
    os.makedirs(grad_cam_path)

    # make folder for guided backpropagation
    guided_backpropagation_path = path + "/images/guided_backpropagation/"
    if os.path.exists(guided_backpropagation_path):
        shutil.rmtree(guided_backpropagation_path)
    os.makedirs(guided_backpropagation_path)

    # make folder for guided grad cam
    guided_grad_cam_path = path + "/images/guided_grad_cam/"
    if os.path.exists(guided_grad_cam_path):
        shutil.rmtree(guided_grad_cam_path)
    os.makedirs(guided_grad_cam_path)