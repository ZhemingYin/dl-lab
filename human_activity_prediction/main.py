import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import *
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
import tensorflow as tf
from input_pipeline.TFRecord import *

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


@gin.configurable
def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    if FLAGS.train:
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    else:
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['/Users/rocker/Desktop/Uni Stuttgart/DL Lab/runs/run_2023-02-03T04-06-04-898610/config_operative.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup wandb
    # Add WANDB for server
    # export WANDB_API_KEY=[YOUR_API_KEY]
    wandb.init(project='human_activity_recognition', name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # Choose model
    model = RNN()
    # model = GRU()
    # model = BRNN()
    # model = CNN_RNN()
    # model = RNN_CNN()
    # model.summary()
    logging.info('The model is loaded successfully.')

    # Start the training process
    if FLAGS.train:
        logging.info('Start the training process...')
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths=run_paths)
        trainer.train(50)

    else:
        evaluate(model, ds_test)

if __name__ == "__main__":
    app.run(main)
