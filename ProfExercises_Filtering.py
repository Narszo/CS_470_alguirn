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
from enum import Enum
from torch import nn
from torchvision.transforms import v2

class FilterType(Enum):
    BOX = "Box Filter"
    GAUSS = "Gaussian Filter"
    MEDIAN = "Median Filter"
    LAPLACE = "Laplacian Filter"
    SHARP_LAPLACE = "Laplacian Sharpening"
    SOBEL_X = "Sobel X"
    SOBEL_Y = "Sobel Y"
    GRAD_MAG = "Gradient Image"
    CUSTOM = "Custom Filter"
    
def do_filter(image, filter_size, filter_type):
    
    if filter_type == FilterType.BOX:
        output = cv2.blur(image, ksize=(filter_size, filter_size))
    elif filter_type == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, 
                                  ksize=(filter_size, filter_size), 
                                  sigmaX=0)
    elif filter_type == FilterType.MEDIAN:
        output = cv2.medianBlur(image, ksize=filter_size)
    elif filter_type == FilterType.LAPLACE:
        laplace = cv2.Laplacian(image, 
                                ddepth=cv2.CV_64F, 
                                ksize=filter_size, 
                                scale=0.25)
        output = cv2.convertScaleAbs(laplace, alpha=0.5, beta=127.0)
    elif filter_type == FilterType.SHARP_LAPLACE:
        laplace = cv2.Laplacian(image, 
                                ddepth=cv2.CV_64F, 
                                ksize=filter_size, 
                                scale=0.25)
        fimage = image.astype("float64")
        output = fimage - laplace
        output = cv2.convertScaleAbs(output)
    elif filter_type == FilterType.SOBEL_X:
        sx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3, scale=0.25)
        output = cv2.convertScaleAbs(sx, alpha=0.5, beta=127)
    elif filter_type == FilterType.SOBEL_Y:
        sy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3, scale=0.25)
        output = cv2.convertScaleAbs(sy, alpha=0.5, beta=127)
    elif filter_type == FilterType.GRAD_MAG:
        sx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3, scale=0.25)
        sy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3, scale=0.25)
        grad_image = np.absolute(sx) + np.absolute(sy)
        output = cv2.convertScaleAbs(grad_image)
    elif filter_type == FilterType.CUSTOM:
        #if custom_filter is None:
        #    raise ValueError("Custom filter cannot be None!")
                    
        custom_filter = np.zeros((filter_size, filter_size), dtype="float64")
        scale = (2.0*np.pi*5.0)/filter_size
        x_coords = np.arange(filter_size).astype("float64")
        sine_vals = np.sin(x_coords*scale)
        custom_filter[:] = sine_vals
        
        display_scale = np.sum(np.absolute(custom_filter))
        filter_result = cv2.filter2D(image, cv2.CV_64F, custom_filter)
        output = cv2.convertScaleAbs(filter_result, 
                                     alpha=1.0/display_scale, 
                                     beta=127)    
        
    return output

###############################################################################
# MAIN
###############################################################################

def main():   
    
    conv_layer = nn.Conv2d(in_channels=1, 
                           out_channels=1, 
                           kernel_size=3, 
                           bias=False,
                           padding="same")
    model = nn.Sequential(conv_layer)
    print(model)
    
    loss_fn = nn.L1Loss() # nn.MSELoss() # nn.L1Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    device = "cuda" # "cpu" or "mps"
    model = model.to(device)
    
    data_transform = v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
    ]) 
    
    print("FILTERING OPTIONS:")
    for index, item in enumerate(list(FilterType)):
        print(index, "-", item.value)
    chosen_index = int(input("Enter choice: "))
    filter_type = list(FilterType)[chosen_index]
    filter_size = int(input("Enter size: "))    
    
        
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
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            output = do_filter(grayscale, filter_size, filter_type)
            
            gray_channel = np.expand_dims(grayscale, axis=-1)            
            data_input = data_transform(gray_channel)
            data_input = torch.unsqueeze(data_input, 0)
            
            sobelX = cv2.Sobel(grayscale, cv2.CV_64F, 
                               dx=1, dy=0, ksize=3, 
                               scale=1.0/(4.0*255))
            desired_output = data_transform(sobelX)
            desired_output = torch.unsqueeze(desired_output, 0)
                        
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
            
            #out_image = cv2.convertScaleAbs(out_image, alpha=0.5, beta=127)   
            out_image = 0.5*out_image + 0.5      
                        
            # Show the image
            cv2.imshow(windowName, grayscale)
            #cv2.imshow("FILTER OUTPUT", output)
            cv2.imshow("PREDICTION", out_image)
            
            loss_value = loss.detach().cpu().numpy()
            #print("LOSS:", loss_value)
            print("Weights:", conv_layer.weight.detach().cpu().numpy())

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'):
                filter_size += 2
                print("Filter size:", filter_size)
            elif key == ord('z'):
                filter_size -= 2
                print("Filter size:", filter_size)
                
            filter_size = max(filter_size, 3)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()

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

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        
        key = -1
        while key == -1:
            # Show image
            cv2.imshow(windowTitle, image)

            # Wait for a keystroke to close the window
            key = cv2.waitKey(30)

        # Cleanup this window
        cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    # Stuff
    