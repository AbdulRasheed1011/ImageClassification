from pathlib import Path
from source.CNNClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        # Common data generator arguments
        datagenerator_kwargs = dict(
            rescale=1. / 255,  # Normalize pixel values to [0, 1]
            validation_split=0.20  # 20% of data for validation
        )

        # Arguments for data flow (resize and batching)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Resize images to model's input size
            batch_size=self.config.params_batch_size,       # Batch size for training and validation
            class_mode="categorical",                      # Multi-class classification
            interpolation="bilinear"                       # Resize method
        )

        # Create validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Root folder containing category subfolders
            subset="validation",                 # Use validation subset
            shuffle=False,                       # Do not shuffle for validation
            **dataflow_kwargs
        )

        # Create training generator with augmentation if specified
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
            train_datagenerator = valid_datagenerator  # No augmentation

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Root folder containing category subfolders
            subset="training",                   # Use training subset
            shuffle=True,                        # Shuffle for training
            **dataflow_kwargs
        )

        # Print the class indices for reference
        print("Class Indices:", self.train_generator.class_indices)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        # Calculate steps per epoch and validation steps
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
