import os.path

import torch
from models import *
from config import *
from dataset import SkijumpDataset, seed_worker
from torch.utils.data import DataLoader
import sys
from visualization import *
import pandas as pd

test_model = "ResNet18"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if sys.gettrace() is None:
    num_workers = 16
else:
    num_workers = 0

if test_model == "ResNet18":
    model = ResNet18Model().to(device)
    model.load_state_dict(torch.load('data/ResNet18Model' + "_model_weights.pth"))
else:
    model = HRNetw32Model().to(device)
    model.load_state_dict(torch.load('data/HRNetModel' + "_model_weights.pth"))

model.eval()

paths, keypoints, bounding_boxes = load_dataset(os.path.join(annotation_path, "test.csv"), image_base_path, offset_columns=2)

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(0)

    dataset = SkijumpDataset(paths,
                                   keypoints.copy(),
                                   bounding_boxes.copy(),
                                   img_size=128,
                                   heatmap_size=64,
                                   use_augment=False,
                                   use_heatmap=False,
                             return_bounding_box_and_resize_factor=True)

    data_loader = DataLoader(dataset=dataset,
                                   batch_size=16,
                                   num_workers=num_workers,
                                   shuffle=True,
                                   drop_last=True,
                                   worker_init_fn=seed_worker,
                                   generator=g
                                   )

    output_coords = None
    with torch.no_grad():
        for x_batch, y_batch, bounding_box_list, resize_factor in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model.forward(x_batch)

            # img = x_batch[1, :, :, :].to("cpu")
            # npimg = convert_tensor_numpy(img)
            # heatmaps = y_hat[1, :, :, :].to("cpu")
            #
            # plot_img_with_heatmaps(img, heatmaps)

            max_act = torch.amax(y_hat, dim=[2, 3]).to(device)
            visible = torch.where(max_act > 0.2)

            y_hat_flatten = y_hat.flatten(start_dim=2).to(device)
            _, max_ind = y_hat_flatten.max(-1)
            y_hat_coords = torch.stack([max_ind // 64, max_ind % 64], -1).to(device)  # ???

            # transform coordinates back to the original image size

            # output from dataset size
            y_hat_coords[:, :, [0, 1]] = y_hat_coords[:, :, [0, 1]] * 2  # ????

            # before resize
            resize = torch.repeat_interleave(resize_factor, 17).to(device)

            # resize = resize.unsqueeze(1).repeat(1, 2)
            y_hat_coords = y_hat_coords.double()
            y_hat_coords.flatten(0, 1)[:, 0] = y_hat_coords.flatten(0, 1)[:, 0] / resize.squeeze(0)
            y_hat_coords.flatten(0, 1)[:, 1] = y_hat_coords.flatten(0, 1)[:, 1] / resize.squeeze(0)

            pad_func = torch.nn.ConstantPad1d((0, 1), 0)
            y_hat_coords = pad_func(y_hat_coords)
            y_hat_coords[visible[0], visible[1], 2] = 1
            y_hat_coords = y_hat_coords[:, :, [1, 0, 2]]

            # before bounding box
            bb = torch.stack(bounding_box_list).transpose(0, 1).to(device)
            bb = torch.repeat_interleave(bb, 17, 0)
            y_hat_coords.flatten(0, 1)[:, [0, 1]] += bb[:, [0, 2]]

            # img = Image.open(paths[0])
            # plot_img_with_keypoints(data_loader.dataset.img_orig[16+3], y_hat_coords[3,:,:].to("cpu"))

            if output_coords is None:
                output_coords = y_hat_coords
            else:
                output_coords = torch.cat([output_coords, y_hat_coords], dim=0)

    output_coords = output_coords.to("cpu").long()
    paths = data_loader.dataset.img_paths

    output = pd.DataFrame({"image_name": paths,
                           "head_x": output_coords[:, 0, 0].numpy(),
                           "head_y": output_coords[:, 0, 1].numpy(),
                            "head_s": output_coords[:,0,2].numpy(),
                           "rsho_x": output_coords[:, 1, 0].numpy(),
                           "rsho_y": output_coords[:, 1, 1].numpy(),
                           "rsho_s": output_coords[:, 1, 2].numpy(),
                           "relb_x": output_coords[:, 2, 0].numpy(),
                           "relb_y": output_coords[:, 2, 1].numpy(),
                           "relb_s": output_coords[:, 2, 2].numpy(),
                           "rhan_x": output_coords[:, 3, 0].numpy(),
                           "rhan_y": output_coords[:, 3, 1].numpy(),
                           "rhan_s": output_coords[:, 3, 2].numpy(),
                           "lsho_x": output_coords[:, 4, 0].numpy(),
                           "lsho_y": output_coords[:, 4, 1].numpy(),
                           "lsho_s": output_coords[:, 4, 2].numpy(),
                           "lelb_x": output_coords[:, 5, 0].numpy(),
                           "lelb_y": output_coords[:, 5, 1].numpy(),
                           "lelb_s": output_coords[:, 5, 2].numpy(),
                           "lhan_x": output_coords[:, 6, 0].numpy(),
                           "lhan_y": output_coords[:, 6, 1].numpy(),
                           "lhan_s": output_coords[:, 6, 2].numpy(),
                           "rhip_x": output_coords[:, 7, 0].numpy(),
                           "rhip_y": output_coords[:, 7, 1].numpy(),
                           "rhip_s": output_coords[:, 7, 2].numpy(),
                           "rkne_x": output_coords[:, 8, 0].numpy(),
                           "rkne_y": output_coords[:, 8, 1].numpy(),
                           "rkne_s": output_coords[:, 8, 2].numpy(),
                           "rank_x": output_coords[:, 9, 0].numpy(),
                           "rank_y": output_coords[:, 9, 1].numpy(),
                           "rank_s": output_coords[:, 9, 2].numpy(),
                           "lhip_x": output_coords[:, 10, 0].numpy(),
                           "lhip_y": output_coords[:, 10, 1].numpy(),
                           "lhip_s": output_coords[:, 10, 2].numpy(),
                           "lkne_x": output_coords[:, 11, 0].numpy(),
                           "lkne_y": output_coords[:, 11, 1].numpy(),
                           "lkne_s": output_coords[:, 11, 2].numpy(),
                           "lank_x": output_coords[:, 12, 0].numpy(),
                           "lank_y": output_coords[:, 12, 1].numpy(),
                           "lank_s": output_coords[:, 12, 2].numpy(),
                           "rsti_x": output_coords[:, 13, 0].numpy(),
                           "rsti_y": output_coords[:, 13, 1].numpy(),
                           "rsti_s": output_coords[:, 13, 2].numpy(),
                           "rsta_x": output_coords[:, 14, 0].numpy(),
                           "rsta_y": output_coords[:, 14, 1].numpy(),
                           "rsta_s": output_coords[:, 14, 2].numpy(),
                           "lsti_x": output_coords[:, 15, 0].numpy(),
                           "lsti_y": output_coords[:, 15, 1].numpy(),
                           "lsti_s": output_coords[:, 15, 2].numpy(),
                           "lsta_x": output_coords[:, 16, 0].numpy(),
                           "lsta_y": output_coords[:, 16, 1].numpy(),
                           "lsta_s": output_coords[:, 16, 2].numpy()
                           })

    output["image_name"] = output["image_name"].str[-14:]
    output["image_name"] = output["image_name"].str.replace("/", "")

    output.to_csv("output/" + test_model + "_output", sep=";", index=False)










