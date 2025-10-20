import logging
from absl import app, flags
import wandb
import gin
import math

from input_pipeline.datasets import load
from models.architectures import *
from train import Trainer
from models.transfer_learning import *
from utils import utils_params, utils_misc
# from evaluation.eval import *


def train_func():
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
        # model = ResNet101()
        model = VGG()
        # model = CNN()
        # model = inception_resnet_v2()
        # model = mobilenet()

        model.summary()

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths=run_paths)
        print("Start the training process")
        trainer.train(30)



sweep_config = {
    'name': 'VGG-sweep',
    'method': 'bayes',
    'metric': {
        'name': 'best_val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'VGG.dropout_rate': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'prepare.batch_size': {
            'values': [16, 32]
        },
        'Trainer.learning_rate': {
            'distribution': 'log_uniform',
            'min': math.log(1e-4),
            'max': math.log(4e-3)
        },
        'output_block.dropout_rate': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project='DR_Tune')
wandb.agent(sweep_id, function=train_func, count=30)
