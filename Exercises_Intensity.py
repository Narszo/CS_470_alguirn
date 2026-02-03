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
    fill.set_verts([np.column_stack([np.arange(256), transform])])
    fig.canvas.draw()
    fig.canvas.flush_events()

def do_transform(image, chosenT):
    
    output = np.copy(image)
    transform = np.arange(256, dtype="uint8")
    
    return output, transform

###############################################################################
# MAIN
###############################################################################

def main():        
    
    chosenT = 0
    
    plt.ion()
    
    tfig, tfill, tline = create_transform_plot(np.arange(256))
    
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
            output, transform = do_transform(grayscale, chosenT)
            
            update_transform_plot(transform, tfig, tfill, tline)
            
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("OUTPUT", output)

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
            
            
        # TODO
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    key = -1
    while key == -1:
        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, grayscale)
        
        output, transform = do_transform(grayscale, chosenT)
        
        cv2.imshow("OUTPUT", output)
        
        update_transform_plot(transform, tfig, tfill, tline)

        # Wait for a keystroke to close the window
        key = cv2.waitKey(10)

    # Cleanup this window
    cv2.destroyAllWindows()

# The main function
if __name__ == "__main__": 
    main()
    