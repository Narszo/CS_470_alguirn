import os
import cv2
import numpy as np
from skimage.segmentation import slic


class CellFinder:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def train_WBC(self, train_data):
        pass

    def find_WBC(self, image):
        # Get superpixel groups using SLIC
        segments = slic(image, n_segments=200, sigma=5, start_label=0, compactness=15)
        cnt = len(np.unique(segments))

        # Compute mean color per superpixel
        group_means = np.zeros((cnt, 3), dtype="float32")
        for specific_group in range(cnt):
            mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
            mask_image_3ch = np.expand_dims(mask_image, axis=-1)
            mean_color = cv2.mean(image, mask=mask_image)[0:3]
            group_means[specific_group] = mean_color

        # Use K-means on group mean colors to group into 4 color groups
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, bestLabels, centers = cv2.kmeans(
            group_means, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Find the k-means group closest to blue
        target_color = np.array([255, 0, 0], dtype="float32")
        distances = np.sqrt(np.sum((centers - target_color) ** 2, axis=1))
        closest_group = np.argmin(distances)

        # Set closest group to white, rest to black
        new_centers = np.zeros_like(centers)
        new_centers[closest_group] = [255, 255, 255]

        # Determine new colors for each superpixel group
        new_centers = np.uint8(new_centers)
        colors_per_clump = new_centers[bestLabels.flatten()]

        # Recolor superpixels with new group colors
        cell_mask = colors_per_clump[segments]
        cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kernel)
        cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, kernel)

        # Use connected components to get disjoint blobs
        retval, labels = cv2.connectedComponents(cell_mask, None, 8, cv2.CV_32S)

        # Extract bounding boxes for each blob (skip background 0)
        bounding_boxes = []
        img_h, img_w = image.shape[:2]
        min_area = (img_h * img_w) * 0.005  # Minimum blob area threshold
        max_area = (img_h * img_w) * 0.5    # Maximum blob area threshold

        for i in range(1, retval):
            coords = np.where(labels == i)
            if len(coords[0]) == 0:
                continue
            
            area = len(coords[0])
            if area < min_area or area > max_area:
                continue

            ymin = int(np.min(coords[0]))
            xmin = int(np.min(coords[1]))
            ymax = int(np.max(coords[0]))
            xmax = int(np.max(coords[1]))

            # Filter out very small or very elongated boxes
            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w < 20 or box_h < 20:
                continue
            aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)
            if aspect_ratio > 4.0:
                continue

            bounding_boxes.append((ymin, xmin, ymax, xmax))
        return bounding_boxes