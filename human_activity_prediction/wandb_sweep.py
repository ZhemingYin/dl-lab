import logging
from absl import app, flags
import wandb
import gin
import math

from input_pipeline.datasets import load
from models.architectures import *
from train import Trainer
from utils import utils_params, utils_misc
# from evaluation.eval import *


def train_func():
    """the training function for W&B sweep"""
    with wandb.init() as run:

        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder()

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config_wandb.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load()

        # model
        # model = RNN()
        # model = GRU()
        model = BRNN()
        # model = CNN_RNN()
        # model = RNN_CNN()

        model.summary()

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths=run_paths)
        print("Start the training process")
        trainer.train(50)


sweep_config = {
    'name': 'BRNN-sweep-positions',
    'method': 'grid',
    'metric': {
        'name': 'best_val_accuracy',
        'goal': 'maximize'
    },
    # 'early_terminate': {
    #   'type': 'hyperband',
    #   'max_iter': '27',
    #   's': '2'
    # },
    'parameters': {
        # 'WINDOW_SIZE': {
        #     'values': [200, 250, 300]
        # },
        # 'window_sliding.shift_ratio': {
        #     'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # },
        # 'BRNN.neuron_num': {
        #     'values': [32, 64, 128]
        # },
        # 'BRNN.number_of_layer': {
        #     'values': [1, 2, 3]
        # },
        # 'Trainer.learning_rate': {
        #     'values': [0.01, 0.001, 0.001, 0.0001]
        # },
        # 'BRNN.dropout_rate': {
        #     'values': [0.1, 0.2, 0.3, 0.4]
        # }
        'load.position': {
            'values': [1, 2, 3, 4, 5, 6, 7]
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project='HAR_Tune')
wandb.agent(sweep_id, function=train_func)
