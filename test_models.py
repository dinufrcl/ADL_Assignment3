import torch
from models import *
from config import *
from dataset import SkijumpDataset, seed_worker
from torch.utils.data import DataLoader
import sys


if sys.gettrace() is None:
    num_workers = 16
else:
    num_workers = 0

resnet18model = ResNet18Model()
resnet18model.load_state_dict(torch.load('data/ResNet18Model' + "_model_weights.pth"))
resnet18model.eval()

paths, keypoints, bounding_boxes = load_dataset(os.path.join(annotation_path, "test.csv"), image_base_path)

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(0)

    dataset = SkijumpDataset(paths,
                                   keypoints,
                                   bounding_boxes,
                                   img_size=128,
                                   heatmap_size=64,
                                   use_augment=False,
                                   use_heatmap=False)

    data_loader = DataLoader(dataset=dataset,
                                   batch_size=16,
                                   num_workers=num_workers,
                                   shuffle=True,
                                   drop_last=True,
                                   worker_init_fn=seed_worker,
                                   generator=g
                                   )

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model.forward(x_batch)



