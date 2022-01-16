import numpy as np
from tensorflow import keras
import gradio as gr
import tensorflow as tf
import cv2
import requests
import keras
import argparse
class Predict():
    def __init__(self,model_path):
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path)

    def predict(self,img_file):
        img_height = 124
        img_width = 124
        img = cv2.imread(img_file.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_height, img_width))
        img = img.astype(np.float32)
        img = tf.expand_dims(img, 0)  # Create a batch
        predictions = (self.model.predict(img) > 0.5).astype("int32")
        print(predictions)
        if predictions >= 0.5:
            return  "Without mask"
        elif predictions < 0.5:
            return "With mask"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict on Single Image')
    parser.add_argument("--model_path", type=str, help='Path for Model')
    args = parser.parse_args()
    p  = Predict(args.model_path)
    labels = ['with_mask','without_mask']
    inputs = gr.inputs.Image(type="file")
    outputs = gr.outputs.Label(num_top_classes=2)
    gr.Interface(fn=p.predict, inputs=inputs, outputs=outputs).launch()




