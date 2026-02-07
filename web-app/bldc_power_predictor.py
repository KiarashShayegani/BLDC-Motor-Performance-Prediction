import os
import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model(os.path.join(".", "BLDC_NN_Model1.h5"))
inputScaler = pickle.load(open(os.path.join(".", "inputScaler_1.pkl"), 'rb'))
outputScaler = pickle.load(open(os.path.join(".", "outputScaler_1.pkl"), 'rb'))

def predict(motor_fb, depth):
    inputs = np.array([[motor_fb, depth]])
    inputs_scaled = inputScaler.transform(inputs)

    prediction = model.predict(inputs_scaled)
    prediction_deNorm = outputScaler.inverse_transform(prediction)

    return prediction_deNorm[0][0], prediction_deNorm[0][1]

interface = gr.Interface(
    fn = predict,
    inputs = [
        gr.Number(label = "دور موتور"), 
        gr.Number(label = "عمق موتور")
    ],
    outputs = [
        gr.Textbox(label = "جریان"),
        gr.Textbox(label = "ولتاژ")
    ],
    title = "مدل پیش‌بینی توان مصرفی موتور",
    description = "برای پیش‌بینی توان مصرفی، مقادیر دور موتور و عمق موتور را وارد نمایید",
    css="body { font-family: 'Vazir', sans-serif; }"
)

if __name__ == "__main__":
    
    interface.launch()