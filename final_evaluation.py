import os.path

import torch
from models import *
from config import *
from dataset import SkijumpDataset, seed_worker
from torch.utils.data import DataLoader
import sys
from visualization import *
import pandas as pd

model_name = "Resnet18ASPPAugProb0_5RandomScale"

device = 'cpu'

if sys.gettrace() is None:
    num_workers = 16
else:
    num_workers = 0

model = ResNet18ASPPModel().to(device)
model.load_state_dict(torch.load('data/Resnet18ASPPAugProb0_5RandomScale' + "_model_weights.pth"))
model.eval()

paths, keypoints_orig, bounding_boxes = load_dataset(os.path.join(annotation_path, "val.csv"),
                                                image_base_path)

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(0)

    dataset = SkijumpDataset(paths,
                                   keypoints_orig.copy(),
                                   bounding_boxes.copy(),
                                   img_size=128,
                                   heatmap_size=64,
                                   use_augment=False,
                                   use_heatmap=False,
                                   return_bounding_box_and_resize_factor=True)

    idx = np.random.choice(len(keypoints_orig), 10, replace=False)

    j = 0
    with torch.no_grad():
        for i in range(10):
            id = idx[i]
            img, _, bounding_box, resize_factor = dataset.__getitem__(id)

            # show original img
            plt.imshow(dataset.img_orig[j])
            plt.show()

            y_hat = model.forward(img.unsqueeze(0))

            # show heatmaps output from model
            plot_img_with_heatmaps(img, y_hat.squeeze(0))

            max_act = torch.amax(y_hat, dim=[2, 3]).to(device)
            visible = torch.where(max_act > 0.2)

            y_hat_flatten = y_hat.flatten(start_dim=2).to(device)
            _, max_ind = y_hat_flatten.max(-1)
            y_hat_coords = torch.stack([max_ind // 64, max_ind % 64], -1).to(device)  # ???

            # transform coordinates back to the original image size

            # output from dataset size
            y_hat_coords[:, :, [0, 1]] = y_hat_coords[:, :, [0, 1]] * 2  # ????

            # before resize
            resize = torch.repeat_interleave(torch.tensor(resize_factor), 17).to(device)

            y_hat_coords = y_hat_coords.double()
            y_hat_coords.flatten(0, 1)[:, 0] = y_hat_coords.flatten(0, 1)[:, 0] / resize
            y_hat_coords.flatten(0, 1)[:, 1] = y_hat_coords.flatten(0, 1)[:, 1] / resize

            pad_func = torch.nn.ConstantPad1d((0, 1), 0)
            y_hat_coords = pad_func(y_hat_coords)
            y_hat_coords[visible[0], visible[1], 2] = 1
            y_hat_coords = y_hat_coords[:, :, [1, 0, 2]]

            # before bounding box
            y_hat_coords.flatten(0, 1)[:, [0, 1]] += torch.tensor(bounding_box)[[0,2]]

            # show img with original keypoints and predicted keypoints
            plot_img_with_keypoints(dataset.img_orig[j], y_hat_coords.squeeze(0))

            keypoints_selected = keypoints_orig[id].copy()
            plot_img_with_keypoints(dataset.img_orig[j], keypoints_selected)

            j += 1
