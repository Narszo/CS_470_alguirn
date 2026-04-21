###############################################################################
# IMPORTS
###############################################################################

import os
import sys
import cv2
import pandas as pd
import numpy as np
import shutil
from enum import Enum

# NOTE: Uncomment this line and comment the following to restore original data directory
BCCD_DATA_DIR = os.path.join(".", "data", "BCCD")
#BCCD_DATA_DIR = os.path.join("..", "..", "data", "BCCD")

BCCD_DATA_CSV = "data.csv"

base_dir = "assign03"

class BCCD_TYPES(Enum):   
    RBC = "RBC"  
    WBC = "WBC"
    PLATE = "Platelets"   
    
QUESTIONABLE_ACC_TESTS = [
    "TEST_025.png",
    "TEST_033.png",
    "TEST_035.png",
    "TEST_053.png",
    "TEST_060.png",
]

###############################################################################
# EXTRACTS BOUNDING BOXES FOR PARTICULAR CELL TYPE
###############################################################################
def unpack_one_cell_type_only(objects, cell_type):
    # Get the bounding boxes and labels from the dictionary
    bboxes = objects['bbox']
    labels = objects['label']
    
    # Create an empty list to hold the bounding boxes we want to keep
    cell_boxes = []

    # For each entry...
    for i in range(len(bboxes)):
        # Get one bounding box and label
        bb = bboxes[i]     
        label = labels[i]   
                
        # If this is the cell we're looking for...        
        if label == cell_type.value:            
            # Add to our list
            cell_boxes.append(bb)

    return cell_boxes

###############################################################################
# LOADS UP BCCD DATA
###############################################################################
def load_and_prepare_BCCD_data(cell_type):
    # Assumes that Prepare_A03 has already been run
    
    # Load up CSV file                
    data = pd.read_csv(os.path.join(BCCD_DATA_DIR, BCCD_DATA_CSV))
    
    # Load up split files
    def load_split_files(filename):
        all_files = []
        with open(os.path.join(BCCD_DATA_DIR, filename), "r") as f:
            all_files = f.readlines()
            
        for i in range(len(all_files)):
            all_files[i] = all_files[i].strip() + ".jpg"
            
        return all_files
    
    train_files = load_split_files("train.txt")
    test_files = load_split_files("test.txt")
    
    def load_objects(df):
        object_data = {
            "bbox": [],
            "label": []
        }
        # For each row
        for index, row in df.iterrows():
            # Get label
            label = row["cell_type"]   
            
            # Get the bounding box
            # y1, x1, y2, x2
            box = np.array([row["ymin"], row["xmin"], row["ymax"], row["xmax"]], dtype="int")
            
            # Add to lists
            object_data["bbox"].append(box)
            object_data["label"].append(label)
            
        return object_data        
            
    def prepare_dataset(filelist, cell_type):
        all_data = []
        # For each file
        for filename in filelist:  
            # If this is NOT in our questionable list...
            if filename not in QUESTIONABLE_ACC_TESTS:  
                # Load image
                image = cv2.imread(os.path.join(BCCD_DATA_DIR, "images", filename))
                # Load up bounding box and label info            
                objects = load_objects(data[(data["filename"] == filename) & (data["cell_type"] == cell_type.value)])
                # Add to list
                all_data.append((image, objects))
                
        return all_data
        
    # Create "datasets" for training and testing
    train_data = prepare_dataset(train_files, cell_type)    
    test_data = prepare_dataset(test_files, cell_type)
    
    # Number of items
    print("Number of training images:", len(train_data))
    print("Number of testing images:", len(test_data))  

    return train_data, test_data

###############################################################################
# COMPUTES INTERSECTION OVER UNION FOR BOUNDING BOXES
###############################################################################
def compute_one_IOU(predicted, ground):
    # Bounding box stored as (y1, x1, y2, x2)   
    def compute_area(left, right, top, bottom):
        width = right - left
        height = bottom - top
        width = max(0, width)
        height = max(0, height)        
        area = width * height
        return area

    # Get intersection
    left = max(predicted[1], ground[1])
    right = min(predicted[3], ground[3])
    top = max(predicted[0], ground[0])
    bottom = min(predicted[2], ground[2])    
    intersection = compute_area(left, right, top, bottom)
    
    # Get union
    area_pred = compute_area(predicted[1], predicted[3], predicted[0], predicted[2])
    area_ground = compute_area(ground[1], ground[3], ground[0], ground[2])
    union = area_pred + area_ground - intersection     

    # Get IOU
    iou = intersection / union
            
    return iou
    
def compute_IOU(all_predicted, all_ground):
    # For each ground box, find the nearest match
    all_IOU = 0.0
    for ground in all_ground:
        best_IOU = 0.0
        for predicted in all_predicted:
            one_IOU = compute_one_IOU(predicted, ground)
            if one_IOU < 0 or one_IOU > 1.0:
                print(one_IOU)   
                exit(1)         
            best_IOU = max(best_IOU, one_IOU)
        all_IOU += best_IOU
    
    # Average it out
    if len(all_ground) > 0:
        all_IOU /= len(all_ground)

    return all_IOU

###############################################################################
# DRAWS BOUNDING BOXES ON IMAGE
###############################################################################
def draw_bounding_boxes(image, bounding_boxes, color):
    # For each box...
    for bb in bounding_boxes:
        cv2.rectangle(image, (bb[1], bb[0]), (bb[3], bb[2]), color, thickness=2)
   
###############################################################################
# PREDICTS BOUNDING BOXES ON DATASET AND COMPUTES METRICS
###############################################################################
def predict_dataset(dataset, model_dir, prefix, out_dir, cell_type, cell_finder):  
    
    # Prepare metric dictionary
    metrics = {}
    metrics["Accuracy"] = 0.0    
    metrics["IOU"] = 0.0

    # Get total count
    total_cnt = len(dataset)       

    # Print starting
    print("Starting on", prefix, "(" + str(total_cnt) + " samples total)")
    
    # For each datapoint...
    image_index = 0
    acc_cnt = 0    
    iou_cnt = 0
    
    # Create cell finder
    cell_finder_obj = cell_finder(model_dir)
    if cell_type == BCCD_TYPES.WBC:
        find_cell_func = cell_finder_obj.find_WBC
    elif cell_type == BCCD_TYPES.RBC:
        find_cell_func = cell_finder_obj.find_RBC
    else:
        raise ValueError("Unsupported cell type for evaluation.")
    
    for data_pack in dataset:
        # Get filename first to check for questionable
        output_filename = "%s_%03d.png" % (prefix, image_index)
         
        # Each item is a tuple, so separate data into image and objects
        image = np.copy(data_pack[0])
        objects = data_pack[1]

        # Objects is a dictionary, so we'll unpack the bounding boxes
        # for specific cells only        
        true_bounding_boxes = unpack_one_cell_type_only(objects, cell_type)
        true_cell_count = len(true_bounding_boxes)
        
        # Calculate bounding boxes using your approach
        pred_bounding_boxes = find_cell_func(image)        
        pred_cell_count = len(pred_bounding_boxes)
        
        # Draw bounding boxes on image
        draw_bounding_boxes(image, true_bounding_boxes, (0,0,0))
        draw_bounding_boxes(image, pred_bounding_boxes, (0,255,0))

        # Show images (DEBUG)   
        #cv2.imshow("IMAGE", image)        
        #cv2.waitKey(-1)

        # Save image
        cv2.imwrite(os.path.join(out_dir, output_filename), image)
                
        # Is this correct in terms of the number of cells predicted?
        count_is_correct = (true_cell_count == pred_cell_count)
                    
        if count_is_correct:
            metrics["Accuracy"] += 1.0            
        acc_cnt += 1
        
        # Compute IOU
        iou = compute_IOU(pred_bounding_boxes, true_bounding_boxes)        
        metrics["IOU"] += iou
        iou_cnt += 1
                                            
        # Increment index
        image_index += 1

        # Print progress
        percent = 100.0*image_index / total_cnt
        print("%.1f%% complete...       " % percent, end="\r", flush=True)

    # Print complete
    print(prefix, "complete!                    ")
        
    # Average out metrics
    metrics["Accuracy"] /= acc_cnt    
    metrics["IOU"] /= iou_cnt

    # Return metrics
    return metrics

###############################################################################
# PRINTS METRICS (to STDOUT or file)
###############################################################################
def print_metrics(all_metrics, stream=sys.stdout):
    for data_type in all_metrics:
        print(data_type + ":", file=stream)
        for key in all_metrics[data_type]:
            print("\t", key, "=", all_metrics[data_type][key], file=stream)
            
###############################################################################
# RECREATE DIRECTORY
###############################################################################
def recreate_dir(out_dir):
    # Does directory exist?
    if os.path.exists(out_dir):
        check_overwrite = input("Folder exists: " 
                                + out_dir 
                                + "\nDo you wish to overwrite it? (y/n) ")
        if check_overwrite == "y":
            shutil.rmtree(out_dir)
        else:
            print("Exiting...")
            exit(1)
            
    # Create output directory
    os.makedirs(out_dir)
  
###############################################################################
# RUN EVALUATION
###############################################################################  
def evaluate(cell_finder, cell_type):
    # Load datasets
    train_data, test_data = load_and_prepare_BCCD_data(cell_type)
        
    # Prepare output directory
    out_dir = os.path.join(base_dir, "output_" + cell_type.value)
    recreate_dir(out_dir)
    
    # Also grab model directory
    model_dir = os.path.join(base_dir, "output_models")
        
    # Prepare metrics
    all_metrics = {}
        
    # Predict for training
    all_metrics["TRAINING"] = predict_dataset(train_data, model_dir, "TRAIN", out_dir, 
                                                cell_type, cell_finder)

    # Predict for testing
    all_metrics["TESTING"] = predict_dataset(test_data, model_dir, "TEST", out_dir, 
                                                cell_type, cell_finder)
    
    # Save metrics
    print_metrics(all_metrics)
    with open(out_dir + "/RESULTS_" + cell_type.value + ".txt", "w") as f:
        print_metrics(all_metrics, f)
        
###############################################################################
# RUN TRAINING
###############################################################################  
def train(cell_finder, cell_type): 
    # Load datasets
    train_data, test_data = load_and_prepare_BCCD_data(cell_type)
    
    # Prepare model directory (ONLY create if it doesn't exist)
    model_dir = os.path.join(base_dir, "output_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
            
    # Run desired training
    cell_finder_obj = cell_finder(model_dir)
    if cell_type == BCCD_TYPES.WBC:
        cell_finder_obj.train_WBC(train_data)
    elif cell_type == BCCD_TYPES.RBC:
        cell_finder_obj.train_RBC(train_data)
    else:
        raise ValueError("Unsupported cell type for training.")
      