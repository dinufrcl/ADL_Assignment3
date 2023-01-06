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

        # # select visible keypoints
        # visibility = np.where(keypoints_selected[:,2]==2)[0]
        # keypoints_selected = keypoints_selected[visibility][:,[0,1]]
        #
        # #c = np.random.randint(1, 5, size=visibility.shape[0], replace=False)
        # #colors = np.array(list(keypoint_colors.values()))[visibility]
        # classes = np.array(list(keypoint_colors.keys()))[visibility]
        #
        # img_selected = paths[id]
        # img_selected = Image.open(img_selected)
        # plt.imshow(img_selected)
        # plt.scatter(keypoints_selected[:,0], keypoints_selected[:,1], c=np.arange(0,keypoints_selected.shape[0]), alpha=0.2)
        # plt.legend(classes,
        #           loc="lower left", title="Classes")
        # plt.show()
        # print("test")





