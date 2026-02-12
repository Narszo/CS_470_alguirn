import numpy as np
import cv2
import matplotlib.pyplot as plt
import gradio as gr

def main():
    def on_image_upload(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                color_image = gr.Image(label="Input Image")
            with gr.Column():
                gray_image = gr.Image(label="Grayscale Image")
            
            color_image.upload(fn=on_image_upload,
                               inputs=color_image,
                               outputs=gray_image)
    
    demo.launch()            

if __name__ == "__main__":
    main()