from pathlib import Path
from source.CNNClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
import os
from PIL import Image


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def remove_ds_store_files(self, directory):
        """
        Remove all .DS_Store files from the specified directory and its subdirectories.
        """
        for root, _, files in os.walk(directory):
            if '.DS_Store' in files:
                os.remove(os.path.join(root, '.DS_Store'))
                print(f".DS_Store has been removed from: {root}")

    def remove_invalid_images(self, directory):
        """
        Validate and remove corrupted image files from the specified directory and its subdirectories.
        """
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify that it's a valid image
                except (IOError, SyntaxError, Image.UnidentifiedImageError):
                    print(f"Removing corrupted file: {file_path}")
                    os.remove(file_path)

    def train_valid_generator(self):
        # Clean the training data directory
        training_data_dir = self.config.training_data
        print("Cleaning the training data directory...")
        self.remove_ds_store_files(training_data_dir)
        self.remove_invalid_images(training_data_dir)

        # Common data generator arguments
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            interpolation="bilinear"
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=training_data_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training data generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=training_data_dir,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
