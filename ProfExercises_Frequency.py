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

def to_numpy_complex(complex_image):
    planes = cv2.split(complex_image)
    complex_data = planes[0] + 1j*planes[1]
    return complex_data

def to_complex_image(complex_data):
    return np.stack([np.real(complex_data),
                     np.imag(complex_data)],
                    axis=2)

def complex_to_polar(complex_data):
    mag = np.abs(complex_data)
    phase = np.angle(complex_data)
    return mag, phase

def make_simple_complex(length=600):
    hl = length/2
    complex_data = np.zeros((length, length), dtype="complex")
    values = np.arange(-hl, hl, 1)
    complex_data[:] = values
    complex_data += np.reshape(1j*values, (-1,1))
    #print(complex_data)
    return complex_data
    

def do_frequency(image):
    output = np.copy(image)
    return output


###############################################################################
# MAIN
###############################################################################

def main(): 
    
    complex_data = make_simple_complex(length=600)
    mag, phase = complex_to_polar(complex_data)
    complex_image = to_complex_image(complex_data)
    
    cv2.normalize(complex_image, complex_image, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(mag, mag, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(phase, phase, norm_type=cv2.NORM_MINMAX)
    
    cv2.imshow("X", complex_image[...,0])
    cv2.imshow("Y", complex_image[...,1])
    cv2.imshow("Magnitude", mag)
    cv2.imshow("Phase", phase)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    
    exit()
       
           
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
            
            # Show the image
            cv2.imshow(windowName, image)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

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
    