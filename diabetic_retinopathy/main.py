import gin
import logging
import wandb
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *
from models.transfer_learning import mobilenet, inception_resnet_v2

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
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup wandb
    # Add WANDB for server
    # export WANDB_API_KEY=[YOUR_API_KEY]
    wandb.init(project='diabetic_retinopathy', name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # Choose model
    model = VGG()
    # model = ResNet101()
    # model = CNN()
    # model = inception_resnet_v2()
    # model = mobilenet()
    model.summary()
    logging.info('The model is loaded successfully.')

    if FLAGS.train:
        logging.info('Start the training process...')
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths=run_paths)
        trainer.train(30)

    else:
        evaluate(model=model, ds_test=ds_test)

if __name__ == "__main__":
    app.run(main)
