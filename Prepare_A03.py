import requests 
import os
import pandas as pd
import shutil
from General_A03 import *

def download_file(file_url, output_dir, output_filename):   
  
    r = requests.get(file_url, verify=True)     
    save_path = os.path.join(output_dir, output_filename)
    
    with open(save_path,'wb') as f:          
        f.write(r.content)       
        
def main():
    # Re-create output directory if it doesn't exist
    if os.path.exists(BCCD_DATA_DIR):
        shutil.rmtree(BCCD_DATA_DIR)        
    os.makedirs(BCCD_DATA_DIR)
        
    # Download the list of files...
    print("***********************************************************")
    print("Files provided by: https://github.com/Shenggan/BCCD_Dataset")
    print("***********************************************************")
    
    base_url = "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/refs/heads/master/"
    
    download_file(base_url + "test.csv",
                  BCCD_DATA_DIR,
                  BCCD_DATA_CSV)
    
    # Open CSV file with data
    data = pd.read_csv(os.path.join(BCCD_DATA_DIR, BCCD_DATA_CSV))
    print(data)
    
    # Load up the train and test splits
    base_split_url = base_url + "/BCCD/ImageSets/Main/"
    def download_split(split_file):
        download_file(os.path.join(base_split_url, split_file), BCCD_DATA_DIR, split_file) 
    download_split("train.txt")
    download_split("test.txt")
    download_split("trainval.txt")
    download_split("val.txt")
    
    # Get image base url
    base_image_url = base_url + "/BCCD/JPEGImages/"
    
    # Make image directory
    image_dir = os.path.join(BCCD_DATA_DIR, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Loop through each image file and download it
    all_image_filesnames = data["filename"].unique().tolist()
    for image_filename in all_image_filesnames:
        print("*", image_filename)
        full_image_url = base_image_url + image_filename
        download_file(full_image_url, image_dir, image_filename) 
                    
if __name__ == "__main__":
    main()
    