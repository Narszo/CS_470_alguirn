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

mouse_down = False

def on_mouse(event, x, y, flags, param):
    global mouse_down
    if event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
    elif (event == cv2.EVENT_LBUTTONDOWN or 
          (mouse_down and event == cv2.EVENT_MOUSEMOVE)):
        mouse_down = True
        mask_image = param[0]
        circle_radius = param[1]
        fill_value = 0.0
        cv2.circle(mask_image, (x,y), 
                   circle_radius, fill_value, -1)
        

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

def polar_to_complex(mag, phase):
    return mag * np.exp(1j*phase)

def shift_complex(complex_data):
    return np.fft.fftshift(complex_data)

def unshift_complex(complex_data):
    return np.fft.ifftshift(complex_data)

def image_space_shift(image):
    rows, cols = image.shape[:2]
    row_powers = (-1)**np.arange(rows)[:,None]
    col_powers = (-1)**np.arange(cols)[None,:]
    neg_powers = row_powers * col_powers
    output = image * neg_powers
    return output

def make_simple_complex(length=600):
    hl = length/2
    complex_data = np.zeros((length, length), dtype="complex")
    values = np.arange(-hl, hl, 1)
    complex_data[:] = values
    complex_data += np.reshape(1j*values, (-1,1))
    #print(complex_data)
    return complex_data

def compute_fourier(image, nonzeroRows=0):
    fimage = image.astype("float64")
    return cv2.dft(fimage, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=nonzeroRows)

def compute_inverse_fourier(freq, nonzeroRows=0):
    return cv2.idft(freq, flags=cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE,
                    nonzeroRows=nonzeroRows)
    
def make_display_magnitude(mag):
    display_mag = cv2.log(mag + 1.0)
    cv2.normalize(display_mag, display_mag, norm_type=cv2.NORM_MINMAX)
    return display_mag

def get_optimal_dft_size(image):
    m = cv2.getOptimalDFTSize(image.shape[0])
    n = cv2.getOptimalDFTSize(image.shape[1])
    return m,n

def make_gaussian_filter(side_len):
    gaussianCol = cv2.getGaussianKernel(side_len, sigma=0)
    gaussianRow = np.transpose(gaussianCol)
    kernel = np.matmul(gaussianCol, gaussianRow)
    return kernel

def complex_image_to_display_mag(complex_image):
    complex_data = to_numpy_complex(complex_image)
    mag, _ = complex_to_polar(complex_data)
    display_mag = make_display_magnitude(mag)
    return display_mag
    
def filter_with_fourier(image, kernel):
    padded_rows = cv2.getOptimalDFTSize(image.shape[0] + kernel.shape[0] - 1)
    padded_cols = cv2.getOptimalDFTSize(image.shape[1] + kernel.shape[1] - 1)
    
    fimage = image.astype("float64")
    fkernel = kernel.astype("float64")
    
    padded_image = cv2.copyMakeBorder(fimage, 
                                      0, padded_rows - image.shape[0], 
                                      0, padded_cols - image.shape[1],
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    
    padded_kernel = cv2.copyMakeBorder(fkernel, 
                                      0, padded_rows - kernel.shape[0], 
                                      0, padded_cols - kernel.shape[1],
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    
    padded_image = image_space_shift(padded_image)
    padded_kernel = image_space_shift(padded_kernel)
    
    F = compute_fourier(padded_image)
    H = compute_fourier(padded_kernel)    
    G = cv2.mulSpectrums(F, H, 0, conjB=False)
    
    display_F = complex_image_to_display_mag(F)
    display_H = complex_image_to_display_mag(H)
    display_G = complex_image_to_display_mag(G)
    cv2.imshow("MAGNITUDE F", display_F)
    cv2.imshow("MAGNITUDE H", display_H)
    cv2.imshow("MAGNITUDE G", display_G)
    
    padded_output = compute_inverse_fourier(G, nonzeroRows=(
                                                image.shape[0] + kernel.shape[0]-1))
    padded_output = image_space_shift(padded_output)
    
    sr = int(kernel.shape[0]/2)
    er = sr + image.shape[0]
    sc = int(kernel.shape[1]/2)
    ec = sc + image.shape[1]
        
    output = np.copy(padded_output[sr:er, sc:ec])
    output /= 255.0
    return output

def do_frequency(image, mask_image, draw_radius):
    '''
    m,n = get_optimal_dft_size(image)   
    padded = cv2.copyMakeBorder(image, 
                                0, m - image.shape[0],
                                0, n - image.shape[1],
                                borderType=cv2.BORDER_CONSTANT,
                                value=0)
    fourier = compute_fourier(padded)
    fourier_data = to_numpy_complex(fourier)
    mag, phase = complex_to_polar(fourier_data)
    
    log_mag = make_display_magnitude(mag)
    display_phase = np.copy(phase)
    cv2.normalize(display_phase, display_phase, norm_type=cv2.NORM_MINMAX)
    
    log_mag = shift_complex(log_mag)
    mag = shift_complex(mag)
    
    mag_window_name = "MAGNITUDE"
    cv2.namedWindow(mag_window_name)
    mask_pack = [mask_image,draw_radius]
    cv2.setMouseCallback(mag_window_name, on_mouse, mask_pack)
    
    mag *= mask_image
    log_mag *= mask_image
    
    cv2.imshow(mag_window_name, log_mag)
    cv2.imshow("PHASE", display_phase)
    
    mag = unshift_complex(mag)
    complex_data = polar_to_complex(mag, phase)
    complex_image = to_complex_image(complex_data)
    
    output = compute_inverse_fourier(complex_image)
    output = output[:image.shape[0], :image.shape[1]]
    
    output /= 255.0    
    
    return output
    '''
    
    kernel = make_gaussian_filter(81)
    output = filter_with_fourier(image, kernel)
    return output

###############################################################################
# MAIN
###############################################################################

def main(): 
    
    '''
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
    '''
    '''
    simple1D = np.array([[1,2,4,4]])
    simple_fourier = compute_fourier(simple1D)
    simple_complex = to_numpy_complex(simple_fourier)
    print("ORIGINAL:", simple1D)
    print("FOURIER:", simple_complex)
    simple_recon = compute_inverse_fourier(simple_fourier)
    print("RECON:", simple_recon)
    exit()
    '''
       
    mask_image = None
    draw_radius = 50
           
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
        ESC_KEY = 27
        key = -1
        while key != ESC_KEY:
            # Get next frame from camera
            _, image = camera.read()
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if mask_image is None:
                m,n = get_optimal_dft_size(grayscale)  
                mask_image = np.ones((m,n), dtype="float64")
                        
            output = do_frequency(grayscale, mask_image, draw_radius)
            
            # Show the image
            cv2.imshow(windowName, grayscale)
            cv2.imshow("OUTPUT", output)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('c'):
                mask_image[:,:] = 1 # = np.ones(grayscale.shape, dtype="float64")
            elif key == ord('a'):
                draw_radius += 5
            elif key == ord('z'):
                draw_radius -= 5
                
            draw_radius = max(draw_radius, 5)
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
    