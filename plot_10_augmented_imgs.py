from config import *
import numpy as np
import os
from dataset import SkijumpDataset
from visualization import plot_img_with_heatmaps

if __name__ == "__main__":
    dataset = "train"
    annotation_path = os.path.join(annotation_path, dataset + ".csv")

    paths, keypoints, bounding_boxes = load_dataset(annotation_path, image_base_path)

    # choose 10 random imgs from dataset
    idx = np.random.choice(len(keypoints), 10, replace=False)

    dataset = SkijumpDataset(paths,
                             keypoints,
                             bounding_boxes,
                             128,
                             use_heatmap=True,
                             heatmap_size=128,
                             use_augment=True)

    for id in idx:
        img, heatmaps = dataset.__getitem__(id)
        plot_img_with_heatmaps(img, heatmaps)
