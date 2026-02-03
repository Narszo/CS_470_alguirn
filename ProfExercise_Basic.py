import cv2
import numpy as np

def main():
    print("HI")
    
    image = np.zeros((480,640,3), dtype="uint8")
    
    print(image.shape, image.dtype)
    width = image.shape[1]
    print("Width:", width)
    
    image[:,:,0] = 255
    image[:100, 30:60] = (0,255,255) # Yellow
    
    subimage = np.copy(image[:200, 10:100, :])
    
    image[:30, :30, 2] = 255
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    
    gray_c = np.expand_dims(gray, -1)
    print(gray_c.shape)
    gray_b_c = np.expand_dims(gray_c, 0)
    print(gray_b_c.shape)
    gray_only_c = np.squeeze(gray_b_c, 0)
    print(gray_only_c.shape)
    
    shades = np.zeros((480,256), dtype="uint8")
    grayrange = np.arange(256, dtype="uint8")
    shades[:] = grayrange
       
    float_image = shades.astype("float64")
    
    #shades = cv2.convertScaleAbs(float_image)
    shades = np.clip(np.round(float_image),0, 255).astype("uint8")
    
    disk_image = cv2.imread("images/test.png", cv2.IMREAD_COLOR)
    disk_image = cv2.cvtColor(disk_image, cv2.COLOR_BGR2GRAY)
    
    disk_image = np.where(disk_image > 200, 200, disk_image)
    disk_image = np.where(disk_image < 100, 100, disk_image)
    
    #cv2.imshow("Disk image", disk_image)
    #cv2.imshow("Shades", shades)
    #cv2.imshow("My Image", image)
    #cv2.imshow("Subimage", subimage)
    #cv2.imshow("Gray", gray)
    #cv2.imshow("Float", float_image)
    #cv2.waitKey(-1)
    
    eyes_orig = cv2.imread("images/eyes.png")
    sf = 10.0
    eyes_small = cv2.resize(eyes_orig, dsize=None, 
                            fx=1/sf, fy=1/sf)
    eyes_horror = cv2.resize(eyes_small, dsize=None,
                             fx=sf, fy=sf,
                             interpolation=cv2.INTER_NEAREST)
    #cv2.imshow("MY EYES!!!!!!!", eyes_horror)
    #cv2.waitKey(-1)    
    
    capture = cv2.VideoCapture("images/noice.mp4")
    
    key = -1
    sf = 1.0
    ghost = None
    MAX_CNT = 15
        
    while key == -1:
        _, image = capture.read()
        
        if ghost is not None:
            fimage = image.astype("float64")
            fghost = ghost.astype("float64")
            fimage = 0.5*fimage + 0.5*fghost
            combo = cv2.convertScaleAbs(fimage) 
        else:
            combo = image      
        
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_index, frame_cnt)
        
        if frame_index % MAX_CNT == 0:
            ghost = np.copy(image)
        
        if frame_index == frame_cnt:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)   
          
        '''    
        image = cv2.resize(image, dsize=None, 
                            fx=1/sf, fy=1/sf)
        image = cv2.resize(image, dsize=None,
                                fx=sf, fy=sf,
                                interpolation=cv2.INTER_NEAREST)
        sf += 0.2 
        '''                              
            
        cv2.imshow("NOICE", combo)
        
        key = cv2.waitKey(30)
        
        
    
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
