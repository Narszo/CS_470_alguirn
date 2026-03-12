import unittest
import General_Testing as GT
import os
import cv2
import numpy as np
import A02

base_dir = "assign02"
image_dir = os.path.join(base_dir, "images")
filter_dir = os.path.join(base_dir, "filters")
ground_dir = os.path.join(base_dir, "ground")
out_dir = os.path.join(base_dir, "output")

def create_gaussian(width, height):
    gaussian_row = cv2.getGaussianKernel(width, sigma=0).astype("float64")
    gaussian_col = cv2.getGaussianKernel(height, sigma=0).astype("float64")
    gaussian_2d = gaussian_row @ gaussian_col.T 
    return gaussian_2d    

ground_kernels = [
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64),
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
        np.array([[1],[0],[-1]], dtype=np.float64),
        np.array([[1, 0, -1]], dtype=np.float64),
        np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10], 
                  [11, 12, 13, 14, 15], 
                  [16, 17, 18, 19, 20], 
                  [21, 22, 23, 24, 25], 
                  [26, 27, 28, 29, 30], 
                  [31, 32, 33, 34, 35]], dtype=np.float64),        
        np.ones((35,35), dtype=np.float64),
        create_gaussian(35, 35)        
]

alphaBetaValues = [
    [0.125, 127],
    [0.125, 127],
    [0.0625, 0],
    [0.125, 127],
    [0.125, 127],
    [0.125, 127],
    [0.0015873015, 0],
    [1.0/(35*35), 0],
    [1, 0]    
]

class Test_A02(unittest.TestCase):
    def setUp(self):
        self.all_filenames = os.listdir(image_dir)
        self.all_filenames.sort()
    
        self.all_filters = os.listdir(filter_dir)
        self.all_filters.sort()
        
    def perform_test_one_read_kernel_file(self, findex):
        # Get filter filename
        filter_filename = self.all_filters[findex]
        # Load using function
        kernel = A02.read_kernel_file(os.path.join(filter_dir, filter_filename))
        # Get ground kernels
        ground = ground_kernels[findex]
        # Compare to data
        GT.check_for_unequal("Failed on filter", filter_filename, kernel, ground)
        
    def test_read_kernel_file(self):
        for i in range(len(self.all_filters)):
            with self.subTest(filename=self.all_filters[i]):
                self.perform_test_one_read_kernel_file(i)
            
    def perform_test_do_convolution_one_image(self, findex, image_index, convolution_func):        
        # Get ground kernels
        kernel = ground_kernels[findex]
        
        # Get kernel name
        kernel_name = self.all_filters[findex][:-4]
        
        # Load image
        image = cv2.imread(os.path.join(image_dir, self.all_filenames[image_index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       
                          
        # Get alpha/beta values
        alpha_beta = alphaBetaValues[findex]
        
        # Filter image
        output_noconvert = convolution_func(image=image, kernel=kernel, alpha=alpha_beta[0], beta=alpha_beta[1], convert_uint8=False)
        output_convertdef = convolution_func(image=image, kernel=kernel)
        output_convert = convolution_func(image=image, kernel=kernel, alpha=alpha_beta[0], beta=alpha_beta[1], convert_uint8=True)
        
        # Get ground images        
        flipped_kernel = cv2.flip(kernel, -1)
        ground = cv2.filter2D(image, ddepth=cv2.CV_64FC1, kernel=flipped_kernel, borderType=cv2.BORDER_CONSTANT)
                
        ground_convertdef = cv2.convertScaleAbs(ground, alpha=1.0, beta=0.0)  
        ground_convert = cv2.convertScaleAbs(ground, alpha=alpha_beta[0], beta=alpha_beta[1])  
               
        # Compare to data
        GT.check_for_unequal("(No convert) Failed on filter " + str(findex) + " with image", 
                             self.all_filenames[image_index], 
                             output_noconvert, ground)
        GT.check_for_unequal("(Convert with defaults) Failed on filter " + str(findex) + " with image", 
                             self.all_filenames[image_index], 
                             output_convertdef, ground_convertdef)
        GT.check_for_unequal("(Convert) Failed on filter " + str(findex) + " with image", 
                             self.all_filenames[image_index], 
                             output_convert, ground_convert)
       
    def perform_test_do_convolution(self, convolution_func, max_cnt=None):
        if max_cnt is None:
            max_cnt = len(self.all_filters)
            
        for i in range(max_cnt):            
            for j in range(len(self.all_filenames)):                
                with self.subTest(filter=self.all_filters[i], image=self.all_filenames[j]):
                    self.perform_test_do_convolution_one_image(i, j, convolution_func)    
    
    def test_do_convolution_fast(self):
        self.perform_test_do_convolution(A02.do_convolution_fast)
        
    def test_do_convolution_slow(self):
        self.perform_test_do_convolution(A02.do_convolution_slow, len(ground_kernels)-2)
    
    def test_do_convolution_fourier(self):
        self.perform_test_do_convolution(A02.do_convolution_fourier)
                           
def main():
    runner = unittest.TextTestRunner()
    test_cases = unittest.TestLoader().loadTestsFromTestCase(Test_A02)    
    runner.run(test_cases)

if __name__ == '__main__':    
    main()

