import numpy as np
import cv2
import matplotlib.pyplot as plt
import gradio as gr
import ProfExercises_Intensity as ei
import pandas as pd

def main():
        
    def on_equalize(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(gray)
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                color_image = gr.Image(label="Input Image")
            with gr.Column():
                gray_image = gr.Image(label="Grayscale Image")
                equal_button = gr.Button("Equalize Image")
                split_button = gr.Button("Split channels")
            with gr.Column():
                hist_plot = gr.Plot(label="Original Histogram")
        with gr.Row():
            with gr.Column():
                with gr.Tabs() as channel_tabs:
                    with gr.TabItem("Red", id="red_tab") as red_tab:
                        red_image = gr.Image(label="Red")
                    with gr.TabItem("Green", id="green_tab") as green_tab:
                        green_image = gr.Image(label="Green")
                    with gr.TabItem("Blue", id="blue_tab") as blue_tab:
                        blue_image = gr.Image(label="Blue")
            with gr.Column():
                piece_points_input = gr.DataFrame(
                    headers=["x", "y"],
                    value=[[0,0], [127,127], [255,255]],
                    datatype=["number", "number"],
                    row_count=[3, "dynamic"],
                    column_count=[2, "fixed"],
                    type="pandas",
                    label="Control Points"
                )
                data_button = gr.Button("Print Data")
                
            def process_data(df):
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.fillna(0)
                param = df.values.tolist()
                print(param)
                
            data_button.click(fn=process_data, 
                              inputs=piece_points_input, 
                              outputs=None)
                
            def on_image_upload(image):
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                plot, _ = ei.create_histogram_plot(gray)
                plt.close(plot)
                return gray, plot
            
            def split_channels(image):
                return image[:,:,0], image[:,:,1], image[:,:,2]
            
            split_button.click(fn=split_channels,
                               inputs=color_image,
                               outputs=[red_image, green_image, blue_image])
            
            color_image.upload(fn=on_image_upload,
                               inputs=color_image,
                               outputs=[gray_image, hist_plot])
            equal_button.click(fn=on_equalize,
                               inputs=gray_image,
                               outputs=gray_image)
    
    demo.launch()            

if __name__ == "__main__":
    main()