###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import cv2
import pandas
import sklearn
import timm
import torchvision
import matplotlib.pyplot as plt
from enum import Enum
from torch import nn
from torchvision.transforms import v2

class IntTransform(Enum):
    ORIGINAL = "Original"
    NEGATIVE = "Negative"
    SLICE = "Intensity Slicing"
    STRETCH = "Constrast Stretching"
    HISTEQ = "Histogram Equalization (and Stretching)"
    
def do_transform(image, chosenT):
    if chosenT == IntTransform.ORIGINAL:
        output = np.copy(image)
        transform = np.arange(256, dtype="uint8")
    elif chosenT == IntTransform.NEGATIVE:
        output = 255 - image
        transform = np.arange(255, -1, -1, dtype="uint8")
    elif chosenT == IntTransform.SLICE:
        windowMin = 100
        windowMax = 150
        lut = np.zeros((256,), dtype="uint8")
        lut[windowMin:(windowMax+1)] = 255
        output = lut[image]
        transform = lut
    elif chosenT == IntTransform.STRETCH:
        points = [[0,0], [127,50], [150,200], [255,255]]
        r_knots, s_knots = zip(*points)
        one_inter = lambda r: np.interp(r, r_knots, s_knots)
        r_values = np.arange(256, dtype="float64")
        s_values = one_inter(r_values)
        transform = np.clip(np.round(s_values),0,255).astype("uint8")
        output = transform[image]
    elif chosenT == IntTransform.HISTEQ:
        output = cv2.equalizeHist(image)
        transform = np.zeros(256, dtype="uint8")
        transform[image.flatten()] = output.flatten()
            
    return output, transform

def create_transform_plot(transform, title="Transformation Function"):
    fig, subfig = plt.subplots(1, 1, figsize=(5,5))
    x = np.arange(256)
    line = subfig.plot(x, transform, color="black", linewidth=1)
    fill = subfig.fill_between(x, transform, color="gray", alpha=0.5)
    subfig.set_xlim([0, 255])
    subfig.set_ylim([0, 255])
    subfig.set_title(title)
    subfig.set_xlabel("Input Intensity")
    subfig.set_ylabel("Output Intensity")
    return fig, fill, line[0]

def update_transform_plot(transform, fig, fill, line):
    line.set_ydata(transform) 
    
    x_coords = np.arange(256)
    x_coords = np.append(x_coords, [255, 0])
    
    y_coords = np.copy(transform)
    y_coords = np.append(y_coords, [0, 0]) 
    
    fill.set_verts([np.column_stack([x_coords, y_coords])])
    fig.canvas.draw()
    fig.canvas.flush_events()
    
def create_histogram_plot(image, title="Image Histogram"):
    fig, subfig = plt.subplots(1, 1, figsize=(5,5))
    x_values = np.arange(256)
    counts,_ = np.histogram(image.flatten(), bins=256, range=[0,256])
    bars = subfig.bar(x_values, counts, color="black", width=1.0)
    subfig.set_xlim([-5, 260])
    subfig.set_xlabel("Intensity")
    subfig.set_title(title)
    return fig, bars

def update_histogram_plot(image, fig, bars):
    counts,_ = np.histogram(image.flatten(), bins=256, range=[0,256])
    for count, bar in zip(counts, bars):
        bar.set_height(count)
    fig.canvas.draw()
    fig.canvas.flush_events()

###############################################################################
# MAIN
###############################################################################

def main(): 
    
    conv_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
    model = nn.Sequential(conv_layer)
    print(model)
    
    loss_fn = nn.MSELoss() # nn.L1Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    device = "cuda" # "cpu" or "mps"
    model = model.to(device)
    
    data_transform = v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
    ])
        
    image = np.array([[0,1,2,3],
                      [3,2,1,0],
                      [2,0,3,1]], dtype="uint8")
    print(image)
    print(image.shape)
    
    lut = np.array([3,2,1,0], dtype="float64") # WARNING: should be uint8 for LUT!!!!!
    
    #output = np.copy(image)
    #for r in range(image.shape[0]):
    #    for c in range(image.shape[1]):
    #        val = image[r,c]
    #        output[r,c] = lut[val]
    output = lut[image]
    print(output)
    print(output.shape, output.dtype)
    
    
    print("INTENSITY TRANSFORMATIONS:")
    for index, item in enumerate(list(IntTransform)):
        print(index, "-", item.value)
    chosen_index = int(input("Enter choice: "))
    chosenT = list(IntTransform)[chosen_index]
        
    plt.ion()
    tfig, tfill, tline = create_transform_plot(np.arange(256))
    
    hist_fig, hist_bars = create_histogram_plot(
                            np.zeros((64,64), dtype="uint8"), 
                            title="Input Image Histogram")
    out_hist_fig, out_hist_bars = create_histogram_plot(
                                        np.zeros((64,64), dtype="uint8"),
                                        title="Output Image Histogram")
           
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print("Random Torch Numbers:")
    print(b)
    print("Do you have Torch CUDA/ROCm?:", torch.cuda.is_available())
    print("Do you have Torch MPS?:", torch.mps.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Torch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("timm:", timm.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
                
        # Webcam
        print("Opening the webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
                
        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open the camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)

        # While not closed...
        key = -1
        while key == -1:
            # Get next frame from camera
            _, image = camera.read()
            
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            gray_channel = np.expand_dims(grayscale, axis=-1)
            
            desired_output = data_transform(gray_channel)
            desired_output = torch.unsqueeze(desired_output, 0)
            
            data_input = data_transform(image)
            data_input = torch.unsqueeze(data_input, 0)
            
            #print("Input:", data_input.shape)
            #print("Output:", desired_output.shape)
            
            model.train()
            data_input = data_input.to(device)
            desired_output = desired_output.to(device)
            pred_output = model(data_input)
            loss = loss_fn(pred_output, desired_output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            out_image = pred_output.detach().cpu()
            out_image = out_image.numpy()
            out_image = out_image[0]
            out_image = np.transpose(out_image, [1,2,0])
            
            output, transform = do_transform(grayscale, chosenT)            
            update_transform_plot(transform, tfig, tfill, tline)
                        
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("OUTPUT", output)
            cv2.imshow("PREDICTED", out_image)
            
            print("Weights:", conv_layer.weight.detach().cpu().numpy())
            
            update_histogram_plot(grayscale, hist_fig, hist_bars)
            update_histogram_plot(output, out_hist_fig, out_hist_bars)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()
        plt.close()

        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) 
        
        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)
            
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        key = -1
        while key == -1:        
            # Show our image (with the filename as the window title)
            windowTitle = "PYTHON: " + filename
            cv2.imshow(windowTitle, grayscale)
            
            output, transform = do_transform(grayscale, chosenT)
            
            cv2.imshow("OUTPUT", output)
            
            update_transform_plot(transform, tfig, tfill, tline)
            
            update_histogram_plot(grayscale, hist_fig, hist_bars)
            update_histogram_plot(output, out_hist_fig, out_hist_bars)

            # Wait for a keystroke to close the window
            key = cv2.waitKey(30)

        # Cleanup this window
        cv2.destroyAllWindows()
        plt.close()

# The main function
if __name__ == "__main__": 
    main()
    