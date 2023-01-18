from config import *
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
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
                 heatmap_size=None,
                 use_random_scale=False,
                 return_bounding_box_and_resize_factor=False,
                 augment_prob=1):
        self.img_paths = img_paths
        self.keypoints = keypoints.copy()
        self.bounding_boxes = bounding_boxes.copy()
        self.img_size = img_size
        self.use_heatmap = use_heatmap
        self.use_augment = use_augment
        self.use_random_scale = use_random_scale
        self.augment_prob = augment_prob
        self.heatmap_size = heatmap_size
        self.num_examples = len(self.img_paths)
        self.return_bounding_box = return_bounding_box_and_resize_factor
        self.img_orig = []
        self.paths_used =[]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        self.paths_used.append(img_path)
        img = Image.open(img_path)
        img_orig = img.copy()
        self.img_orig.append(img_orig)
        bounding_box = self.bounding_boxes[idx]
        keypoints = self.keypoints[idx].copy()
        keypoints = keypoints.astype('float64')

        # normalize img
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        # crop image with bounding box
        img = img[:, bounding_box[2]:bounding_box[3], bounding_box[0]:bounding_box[1]]

        # todo: h und w richtig?
        height = img.size()[1]
        width = img.size()[2]

        # crop keypoints with bounding box
        keypoints[:, 0] -= bounding_box[0]
        keypoints[:, 1] -= bounding_box[2]

        # from visualization import *
        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        if self.use_random_scale:
            if torch.rand(1) > self.augment_prob:
                scale = random.uniform(0.3, 2)
                img, keypoints = random_scale(img, keypoints, scale, height, width)

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

        # from visualization import *
        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, keypoints)

        if self.use_augment:
            if torch.rand(1) <= self.augment_prob:
                # color augmentation
                color_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                        hue=0.2)
                img = color_transform(img)

                rot = random.randint(-30, 30)
                trans_x = random.randint(0, int(0.3 * img.size()[1])) # todo: img_size hier?
                trans_y = random.randint(0, int(0.3 * img.size()[1]))

                if torch.rand(1) <= 0.5:
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
            keypoints = torch.tensor(keypoints).float()

            # from visualization import *
            # npimg = convert_tensor_numpy(img)
            # plot_img_with_keypoints(npimg, keypoints)

            if self.return_bounding_box:

                return img, keypoints, bounding_box, resize_factor
            else:
                return img, keypoints

    @classmethod
    def augment(cls, img, label, rot, trans_x, trans_y, flip, h, w):
        img_size = img.size()[1]

        """ apply rotation & translation to img """
        center = [int(w/2), int(h/2)]
        img = TF.affine(img, angle=rot, translate=[trans_x, trans_y], center=center, scale=1, shear=0)

        """ apply horizontal flip to img """
        if flip:
            img = TF.hflip(img)

        """ apply rotation to coords"""
        x = np.copy(label[:, 0])
        y = np.copy(label[:, 1])

        rot_rad = np.radians(rot)

        label[:, 0] = (x - center[0]) * np.cos(rot_rad) - (y - center[1]) * np.sin(rot_rad) + center[0]
        label[:, 1] = (x - center[0]) * np.sin(rot_rad) + (y - center[1]) * np.cos(rot_rad) + center[1]

        """ apply translation to coords """
        label[:, 0] += trans_x
        label[:, 1] += trans_y

        """ apply horizontal flip to coords """
        if flip:
            label[:, 0] = img_size - label[:, 0] - 1
            label = label[cls.coords_hflip, :]

        # label[:, [0,1]] = label[:, [0,1]] * 2

        """ set coords to invisible that moved out of image """
        label[:, 2] = np.where((label[:, 0] > img_size) |
                               (label[:, 0] < 0) |
                               (label[:, 1] > img_size) |
                               (label[:, 1] < 0), 0, label[:, 2])

        # from visualization import *
        # npimg = convert_tensor_numpy(img)
        # plot_img_with_keypoints(npimg, label)


        return img, label

    def __len__(self):
        return self.num_examples

def random_scale(img, keypoints, scale, height, width):
    if torch.rand(1) > 0.5:
        img = TF.resize(img, [height, int(width * scale)])
        keypoints[:, 0] = keypoints[:, 0] * scale
    else:
        img = TF.resize(img, [int(height * scale), width])
        keypoints[:, 1] = keypoints[:, 1] * scale

    # from visualization import *
    # npimg = convert_tensor_numpy(img)
    # plot_img_with_keypoints(npimg, keypoints)

    return img, keypoints


def create_heatmaps(joints, output_size, sigma=2):
    blob_size = 6 * sigma + 3
    heatmaps = []



    #
    # hm = torch.zeros((joints.shape[0], output_size, output_size))
    # blob = torch.vstack([torch.repeat_interleave(torch.arange(blob_size), blob_size),
    #                      torch.tile(torch.arange(blob_size), (blob_size,))])
    # x = (blob[0, :] - 7) ** 2
    # y = (blob[1, :] - 7) ** 2
    # g = torch.exp(- (x + y) / (2 * sigma ** 2))
    #
    #
    #
    #
    # x_coords = torch.repeat_interleave(torch.arange(joints[:, 0] - 7, joints[:, 0] + 8), blob_size)
    # y_coords = torch.tile(torch.arange(joints[:, 1] - 7, joints[:, 1] + 8), (blob_size,))




    for joint in joints:
        hm = torch.zeros((output_size, output_size))
        if joint[2] == 0:
            heatmaps.append(hm)
        else:
            # create blob
            blob = torch.vstack([torch.repeat_interleave(torch.arange(blob_size), blob_size),
                                 torch.tile(torch.arange(blob_size), (blob_size,))])
            x = (blob[0, :] - 7) ** 2
            y = (blob[1, :] - 7) ** 2
            g = torch.exp(- (x + y) / (2 * sigma ** 2))

            # determine location of blob in hm
            x_coords = torch.repeat_interleave(torch.arange(int(joint[0]) - 7, int(joint[0]) + 8), blob_size)
            y_coords = torch.tile(torch.arange(int(joint[1]) - 7, int(joint[1]) + 8), (blob_size,))

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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


if __name__ == "__main__":
    annotation_path = os.path.join(annotation_path, "train.csv")
    ds = load_dataset(annotation_path, image_base_path)

    train_dataset = SkijumpDataset(ds[0], ds[1], ds[2], 200, use_heatmap=True, heatmap_size=200)

    img, keypoints = train_dataset.__getitem__(20)

    from visualization import *

    npimg = convert_tensor_numpy(img)
    plot_img_with_keypoints(npimg, keypoints)
