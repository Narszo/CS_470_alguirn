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

###############################################################################
# MAIN
###############################################################################

def main():        
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
        
        colors = [
            (0,0,0),
            (0,0,255),
            (0,255,0),
            (255,0,0),
            (255,0,255),
            (0,255,255),
            (255,255,0),
            (255,255,255)            
        ]
        colors = np.array(colors, dtype="uint8")
        
        iterations = 1
        se_size = 3

        # While not closed...
        key = -1
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            chosen, output = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)  
            
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size,se_size))    
            #output = cv2.erode(output, element, iterations=iterations)
            
            chosen_op = cv2.MORPH_OPEN
            #chosen_op = cv2.MORPH_TOPHAT            
            output = cv2.morphologyEx(output, chosen_op, element, iterations=iterations)
            
            numConnect, labelImage = cv2.connectedComponents(output, None, connectivity=8, ltype=cv2.CV_32S)
            modLabels = labelImage % len(colors)
            #print(modLabels.dtype)
            colorRegions = colors[modLabels]  
            
            chosenLabels = 255*(labelImage == 3).astype("uint8")                
            
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("OTSU THRESHOLD", output)
            cv2.imshow("REGIONS", colorRegions)
            #cv2.imshow("Chosen", chosenLabels)
            #print("Threshold:", chosen)
            print("Region count:", numConnect)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'): iterations -= 1
            if key == ord('s'): iterations += 1
            if key == ord('q'): se_size -= 1
            if key == ord('w'): se_size += 1
            se_size = max(se_size, 3)
            iterations = max(iterations, 1)
            print("Iterations and size:", iterations, se_size)

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
    