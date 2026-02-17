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
# import matplotlib.pyplot as plt
from enum import Enum
# from torch import nn
# from torchvision.transforms import v2

class FilterType(Enum):
    BOX = "Box Filter"
    GAUSS = "Gaussian Filter"
    MEDIAN = "Median Filter"
    LAPLACE = "Laplace Filter"

def do_filter(image, filter_size, filter_type):
    if filter_type == FilterType.BOX:
        output = cv2.blur(image, ksize=(filter_size, filter_size))
    elif filter_type == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, ksize=(filter_size, filter_size), sigmaX = 0)
    elif filter_type == FilterType.MEDIAN:
        output = cv2.medianBlur(image, ksize=filter_size)
    elif filter_type == FilterType.LAPLACE:
        laplace = cv2.Laplacian(image, ddepth=cv2.CV_64F, ksize = filter_size, scale = 0.25)
        ouput = cv2.convertScaleAbs(laplace, alpha = 0.5, beta = 127.0)
    
    return output

###############################################################################
# MAIN
###############################################################################

def main():        
    
    print("FILTERING OPTIONS:")
    for index, item in enumerate(list(FilterType)):
        print(index, "-", item.value)
    chosen_index = int(input("Enter choice: "))
    filter_type = list(FilterType)[chosen_index]
    filter_size = int(input("Enter size: "))

    image = np.array([[0,1,2,3],[3,2,1,0], [2,0,3,1]], dtype="uint8")
    print(image)
    print(image.shape)
    
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
            output = do_filter(grayscale, filter_size, filter_type)
            
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("FILTER OUTPUT", output)
            
            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()
        #plt.close()

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
            
            
        # TODO
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    key = -1
    while key == -1:
        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, grayscale)
        output = do_filter(grayscale, filter_size, filter_type)
        
        cv2.imshow("FILTER OUTPUT", output)
        
        # Wait for a keystroke to close the window
        key = cv2.waitKey(30)

    # Cleanup this window
    cv2.destroyAllWindows()
    # plt.close()

# The main function
if __name__ == "__main__":
    main()
    