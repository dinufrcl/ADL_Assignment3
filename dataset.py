import numpy as np
import torch
from config import *
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from visualization import plot_img_with_heatmaps
import math
import random

class SkijumpDataset(torch.utils.data.Dataset):
    coords_hflip = list(keypoint_ids_r_s_swapped.values())

    def __init__(self,
                 img_paths,
                 keypoints,
                 bounding_boxes,
                 img_size,
                 use_heatmap=False,
                 use_augment=False,
                 heatmap_size=None):
        self.img_paths = img_paths
        self.keypoints = keypoints
        self.bounding_boxes = bounding_boxes
        self.img_size = img_size
        self.use_heatmap = use_heatmap
        self.use_augment = use_augment
        self.heatmap_size = heatmap_size
        self.num_examples = len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        bounding_box = self.bounding_boxes[idx]
        keypoints = self.keypoints[idx]

        # normalize img
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        # crop image with bounding box
        # todo: h und w richtig?
        img = img[:, bounding_box[2]:bounding_box[3], bounding_box[0]:bounding_box[1]]

        height = img.size()[1]
        width = img.size()[2]

        # crop keypoints with bounding box
        keypoints[:, 0] -= bounding_box[0]
        keypoints[:, 1] -= bounding_box[2]

        # from visualization import *
        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        # determine how much padding is needed
        # determine shorter side
        ls = torch.argmax(torch.tensor(img.size())[1:3])+1 # long side
        ss = torch.argmin(torch.tensor(img.size())[1:3])+1 # short side

        dx = torch.tensor(img.size())[ls] - torch.tensor(img.size())[ss]

        if ls == 1:
            img = TF.pad(img, (0, 0, dx, 0))
        else:
            img = TF.pad(img, (0, 0, 0, dx))

        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        if self.use_augment:
            # color augmentation
            color_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                    hue=0.2)
            img = color_transform(img)

            rot = random.randint(-30, 30)
            trans_x = random.randint(0, int(0.3 * img.size()[1]))
            trans_y = random.randint(0, int(0.3 * img.size()[1]))

            if torch.rand(1) > 0.5:
                flip = True
            else:
                flip = False

            img, keypoints = SkijumpDataset.augment(img=img,
                                                label=keypoints,
                                                trans_x=trans_x,
                                                trans_y=trans_y,
                                                flip=flip,
                                                h=height,
                                                w=width,
                                                rot=rot)

        # resize to img_size
        # todo: Koordinaten nicht mit resize anpassen ??
        resize_factor = self.img_size / img.shape[1]
        img = TF.resize(img, [self.img_size, self.img_size])

        keypoints[:, 0:2] = keypoints[:, 0:2] * resize_factor

        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        if self.use_heatmap:
            joints = keypoints
            joints[:, 0:2] = (self.heatmap_size / self.img_size) * keypoints[:, 0:2]
            heatmaps = create_heatmaps(joints, self.heatmap_size)

            return img, heatmaps
        else:
            keypoints = torch.tensor(keypoints).long()

            # from visualization import *
            # npimg = convert_tensor_numpy(img)
            # plot_img_with_keypoints(npimg, keypoints)

            return img, keypoints



    @classmethod
    def augment(cls, img, label, rot, trans_x, trans_y, flip, h, w):
        # todo: cls für flip
        # wie flip ist wird als Klassenattribut festgelegt
        # wird benötigt damit man danach noch keypoints richtig zuordnen kann
        # dafür für Klasse festlegen welche keypoints wo sind --> static Attribute
        # hier so ähnlich wie static Methode

        img_size = img.size()[1]

        """ apply rotation & translation to img """
        #center = [int(h/2), int(w/2)]
        # todo: richiter Mittelpunkt?
        center = [int(w/2), int(h/2)]
        img = TF.affine(img, angle=rot, translate=[trans_x, trans_y], center=center, scale=1, shear=0)

        """ apply horizontal flip to img """
        if flip:
            img = TF.hflip(img)

        """ apply rotation to coords"""
        x = np.copy(label[:, 0])
        y = np.copy(label[:, 1])

        # todo: math bib hier ok?
        rot_rad = math.radians(rot)

        label[:, 0] = (x - center[0]) * np.cos(rot_rad) - (y - center[1]) * np.sin(rot_rad) + center[0]
        label[:, 1] = (x - center[0]) * np.sin(rot_rad) + (y - center[1]) * np.cos(rot_rad) + center[1]

        """ apply translation to coords """
        label[:, 0] += trans_x
        label[:, 1] += trans_y

        """ apply horizontal flip to coords """
        if flip:
            label[:, 0] = img_size - label[:, 0] - 1
            # todo: fehlt hier noch was? sh. oben --> make sure to change left and right?
            label = label[cls.coords_hflip, :]

        """ set coords to invisible that moved out of image """
        label[:, 2] = np.where((label[:, 0] > img_size) |
                               (label[:, 0] < 0) |
                               (label[:, 1] > img_size) |
                               (label[:, 1] < 0), 0, label[:, 2])

        # from visualization import *
        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, label)

        # todo: um 30% der Bildgröße verschieben

        return img, label

    def __len__(self):
        return self.num_examples

def create_heatmaps(joints, output_size, sigma=2):
    blob_size = 6 * sigma + 3
    heatmaps = []

    for joint in joints:
        hm = torch.zeros((output_size, output_size))
        if joint[2] == 0:
            heatmaps.append(hm)
        else:
            # create blob
            blob = torch.vstack([torch.repeat_interleave(torch.arange(blob_size), blob_size), torch.tile(torch.arange(blob_size), (blob_size,))])
            x = (blob[0, :] - 7) ** 2
            y = (blob[1, :] - 7) ** 2
            g = torch.exp(- (x + y) / (2 * sigma ** 2))

            # determine location of blob in hm
            x_coords = torch.repeat_interleave(torch.arange(joint[0] - 7, joint[0] + 8), blob_size)
            y_coords = torch.tile(torch.arange(joint[1] - 7, joint[1] + 8), (blob_size,))

            y_coords = y_coords[x_coords < output_size]
            g = g[x_coords < output_size]
            x_coords = x_coords[x_coords < output_size]

            x_coords = x_coords[y_coords < output_size]
            g = g[y_coords < output_size]
            y_coords = y_coords[y_coords < output_size]

            # put blob in heatmap
            hm[y_coords, x_coords] = g

            heatmaps.append(hm)

    return torch.stack(heatmaps)





    # for joint in joints:
    #     hm = np.zeros((output_size, output_size))
    #     if joint[2] == 0:
    #         heatmaps.append(hm)
    #     else:
    #         # create blob
    #         blob = np.vstack([np.repeat(np.arange(blob_size), blob_size), np.tile(np.arange(blob_size), blob_size)])
    #         x = (blob[0, :] - 7) ** 2
    #         y = (blob[1, :] - 7) ** 2
    #         g = np.exp(- (x + y) / (2 * sigma ** 2))
    #
    #         # determine location of blob in hm
    #         x_coords = np.repeat(np.arange(joint[0] - 7, joint[0] + 8), blob_size)
    #         y_coords = np.tile(np.arange(joint[1] - 7, joint[1] + 8), blob_size)
    #
    #         y_coords = y_coords[x_coords < output_size]
    #         g = g[x_coords < output_size]
    #         x_coords = x_coords[x_coords < output_size]
    #
    #         x_coords = x_coords[y_coords < output_size]
    #         g = g[y_coords < output_size]
    #         y_coords = y_coords[y_coords < output_size]
    #
    #         # put blob in heatmap
    #         hm[y_coords, x_coords] = g[np.where((y_coords < output_size) & (x_coords < output_size))]
    #
    #         heatmaps.append(hm)
    #
    # return torch.tensor(np.asarray(heatmaps))









    # determine which keypoints are visible
    # visibile = np.where(joints[:, 2] != 0)[0]
    # non_visibile = np.where(joints[:, 2] == 0)[0]
    # keypoints_non_visible = joints[non_visibile][:, [0, 1]]
    # keypoints_visible = joints[visibile][:, [0, 1]]
    #
    # # empty heatmaps for non visible keypoints
    #
    # hm_coord = np.vstack([np.arange((6 * sigma + 3)), np.arange((6 * sigma + 3))])
    # hm_coord = np.transpose(hm_coord)
    # hm_coord = np.expand_dims(hm_coord, axis=0)
    # hm_coord = np.repeat(hm_coord, keypoints_visible.shape[0], axis=0)
    # hm_coord = np.transpose(hm_coord)
    # keypoints_visible = np.transpose(keypoints_visible)
    #
    # # todo: x0 und y0 sind jeweils 7 und 7 da Mitte von Blobs
    #
    # x = (hm_coord[0,:,:] - keypoints_visible[0,:]) ** 2
    # y = (hm_coord[1, :, :] - keypoints_visible[1, :]) ** 2
    #
    # # todo: hier kommen nur 0en raus --> x und y Ergebnisse zu hoch? Warum?
    # g = np.exp(- (x+y) / (2 * sigma ** 2))

    # todo: g Werte zu heatmap ändern?
    # todo: Rückgabewert ist Listen von heatmaps? --> np Array mit Dim = Anzahl Keypoints
    # todo: wie macht man overlay heatmaps?



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


if __name__ == "__main__":
    annotation_path = os.path.join(annotation_path, "train.csv")
    ds = load_dataset(annotation_path, image_base_path)

    train_dataset = SkijumpDataset(ds[0], ds[1], ds[2], 200, use_augment=True)

    img, keypoints = train_dataset.__getitem__(20)

    from visualization import *

    npimg = convert_tensor_numpy(img)
    plot_img_with_keypoints(npimg, keypoints)
