import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.keras"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # Make prediction
        result = np.argmax(model.predict(test_image), axis=1)[0]

        # Define class mapping
        class_mapping = {0: "cat", 1: "dog", 2: "man", 3: "woman"}
        prediction = class_mapping.get(result, "Unknown")

        print(f"Predicted class: {prediction}")
        return prediction