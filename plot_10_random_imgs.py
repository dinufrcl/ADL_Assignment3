from config import *
import numpy as np
import os
from visualization import plot_img_with_keypoints

if __name__ == "__main__":
    # select dataset randomly
    dataset = np.random.choice(np.array(["train", "val"]))
    annotation_path = os.path.join(annotation_path, dataset + ".csv")

    paths, keypoints, _ = load_dataset(annotation_path, image_base_path)
    idx = np.random.choice(len(keypoints), 10, replace=False)

    for id in idx:
        keypoints_selected = keypoints[id]

        img_selected = paths[id]
        img_selected = Image.open(img_selected)

        plot_img_with_keypoints(img_selected, keypoints_selected)




