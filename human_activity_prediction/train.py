import gin
import tensorflow as tf
import logging
import wandb
import time


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, learning_rate):
        """
        Train the model
        Args:
            model: model to be trained
            ds_train: train set
            ds_val: validation set
            ds_info: information of dataset
            run_paths: path to save checkpoint
            learning_rate: learning rate for optimizer
        """

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.learning_rate = learning_rate

        # Summary Writer
        self.summary_writer = tf.summary.create_file_writer(self.run_paths['path_model_Tensorboard'])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model, optimizer=tf.keras.optimizers.Adam())
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  directory=self.run_paths["path_ckpts_train"],
                                                  max_to_keep=10)
        # self.manager = tf.train.CheckpointManager(self.ckpt,
        #                                           directory='/Users/rocker/dl-lab-22w-team06/experiments/run_2023-01-30T20-40-07-501849/ckpts',
        #                                           max_to_keep=10)

        # Loss objective
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,
        #                                                          decay_steps=1000,
        #                                                          alpha=0.1)
        # self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # accuracy objective
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()

    @tf.function
    def train_step(self, features, labels):

        '''One step of training'''

        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            # if self.classification == 'multiple' or self.classification == 'binary':
            #     train_loss = self.loss_object(labels, predictions)
            # elif self.classification == 'regression':
            #     train_loss = tf.keras.losses.MSE(labels, predictions)
            train_loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(train_loss)
        self.train_accuracy(labels, predictions)
        return

    @tf.function
    def val_step(self, features, labels):

        '''One step of validation'''

        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(features, training=True)
        self.val_accuracy(labels, predictions)
        self.val_loss(self.loss_object(labels, predictions))
        return

    # @tf.function
    def train(self, epochs, batch_size=32):

        '''Complete train process'''

        logging.info(f'{self.ds_train}')
        logging.info('Starting')
        self.ckpt.restore(self.manager.latest_checkpoint)

        # If training was interrupted unexpectedly, resume the training process
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # epochs = 10
        for epoch in range(epochs):
            logging.info("\nStart of epoch %d" % (epoch+1,))
            start_time = time.time()

            # Reset train metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for batch_idx, (windows, labels) in enumerate(self.ds_train):
                self.train_step(windows, labels)

            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            for val_windows, val_labels in self.ds_val:
                self.val_step(val_windows, val_labels)
            val_acc = self.val_accuracy.result().numpy()
            val_acc_max = 0

            # If the number of epochs is greater than 10,
            # save the checkpoint based on the maximum validation accuracy
            if epoch < 10 or val_acc > val_acc_max:
                val_acc_max = val_acc
                self.manager.save()
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
            else:
                # Don't save the checkpoint if the validation accuracy isn't great enough
                logging.info(f'Did not save the checkpoint at epoch {epoch + 1}')

            # Save the final checkpoint
            if epoch % (epochs - 1) == 0 and epoch > 0:
                if val_acc > val_acc_max:
                    logging.info(f'Finished training after epochs {epochs} and saved the last checkpoint')
                    self.manager.save()
                else:
                    logging.info(f'Finish training after epochs {epochs} and did not saved the last checkpoint')


            template = 'epoch {}, Loss: {}, Accuracy: {}%, ' \
                       'Validation Loss: {}, Validation Accuracy: {}%, Time taken: {}'
            logging.info(template.format(epoch + 1,
                                         self.train_loss.result().numpy(),
                                         self.train_accuracy.result().numpy() * 100,
                                         self.val_loss.result().numpy(),
                                         self.val_accuracy.result().numpy() * 100,
                                         time.time() - start_time
                                         )
                         )

            # Write summary to tensorboard
            tf.summary.trace_on(graph=True, profiler=False)
            with self.summary_writer.as_default():
                tf.summary.scalar("train_loss", self.train_loss.result(), epoch+1)
                tf.summary.scalar("train_accuracy", self.train_accuracy.result() * 100, epoch+1)
                tf.summary.scalar("val_loss", self.val_loss.result(), epoch+1)
                tf.summary.scalar("val_accuracy", self.val_accuracy.result() * 100, epoch+1)
                tf.summary.trace_export(name="Default", step=0,
                                        profiler_outdir=self.run_paths['path_model_Tensorboard'])

            # Write summary to wandb
            wandb.log({'train_loss': self.train_loss.result(),
                       "train_accuracy": self.train_accuracy.result() * 100,
                       "val_loss": self.val_loss.result(),
                       "val_accuracy": self.val_accuracy.result() * 100,
                       "epoch": epoch + 1,
                       "best_val_accuracy": val_acc_max * 100})