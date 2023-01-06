import numpy as np
import os
from glob import glob
from PIL import Image


# keypoint_colors = {'head': ( 70, 70, 70),
#                        'rsho': (100, 40, 40),
#                        'relb': (220, 20, 60),
#                        'rhan': (153,153,153),
#                        'lsho': (157,234, 50),
#                        'lelb': (128, 64,128),
#                        'lhan': (244, 35,232),
#                        'rhip': (107,142, 35),
#                        'rkne': (  0,  0,142),
#                        'rank': (102,102,156),
#                        'lhip': (220,220,  0),
#                        'lkne': ( 70,130,180),
#                        'lank': (180,165,180),
#                        'rsti': (250,170, 30),
#                        'rsta': (145,170,100),
#                        'lsti': (200, 20,100),
#                        'lsta': (85, 266, 5)}

keypoint_colors = {'head': "darkblue",
                       'rsho': "meadiumseagreen",
                       'relb': "darksalmon",
                       'rhan': "mediumpurple",
                       'lsho': "orchid",
                       'lelb': "midnightblue",
                       'lhan': "firebrick",
                       'rhip': "orange",
                       'rkne': "darkolivegreen",
                       'rank': "darkblue",
                       'lhip': "dimgrey",
                       'lkne': "palevioletred",
                       'lank': "indigo",
                       'rsti': "orangered",
                       'rsta': "yellowgreen",
                       'lsti': "coral",
                       'lsta': "paleturquoise"}


keypoint_ids = {'head': 0,
                       'rsho': 1,
                       'relb': 2,
                       'rhan': 3,
                       'lsho': 4,
                       'lelb': 5,
                       'lhan': 6,
                       'rhip': 7,
                       'rkne': 8,
                       'rank': 9,
                       'lhip': 10,
                       'lkne': 11,
                       'lank': 12,
                       'rsti': 13,
                       'rsta': 14,
                       'lsti': 15,
                       'lsta': 16}

keypoint_ids_r_s_swapped = {'head': 0,
                       'rsho': 4,
                       'relb': 5,
                       'rhan': 6,
                       'lsho': 1,
                       'lelb': 2,
                       'lhan': 3,
                       'rhip': 10,
                       'rkne': 11,
                       'rank': 12,
                       'lhip': 7,
                       'lkne': 8,
                       'lank': 9,
                       'rsti': 15,
                       'rsta': 16,
                       'lsti': 13,
                       'lsta': 14}

annotation_path = os.path.join(os.getcwd(), "dataset/annotations")
image_base_path = os.path.join(os.getcwd(), "dataset/annotated_frames")

def load_dataset(annotation_path, image_base_path, offset_columns=4):
    annotations = open(annotation_path)

    paths = []
    keypoints_list = []
    bounding_boxes = []
    i = 0
    for line in annotations:
        tmp = line.split(";")
        if i > 1:
            while len(tmp[1]) < 5:
                tmp[1] = "0" + tmp[1]
            path = os.path.join(image_base_path, tmp[0], tmp[0]+"_("+tmp[1]+").jpg")
            paths.append(path)

            keypoints = np.empty((0,3))
            for j in np.arange(offset_columns, len(tmp), 3):
                arr = np.array([tmp[j:j+3]], dtype=int)
                keypoints = np.append(keypoints.astype(int), arr, axis=0)
            keypoints_list.append(keypoints)

            # extract all visible keypoints
            kp_vis = keypoints[np.where(keypoints[:,2]!=0)[0]][:,[0,1]]

            # load image to determine image boundaries
            img = np.array(Image.open(path))
            x_boundary = img.shape[1]
            y_boundary = img.shape[0]

            min_x = min(kp_vis[:, 0])
            max_x = max(kp_vis[:, 0])
            min_y = min(kp_vis[:, 1])
            max_y = max(kp_vis[:, 1])

            height = max_y-min_y
            width = max_x-min_x

            if int(max_x + 0.21*width) <= x_boundary:
                max_x = int(max_x + 0.21*width)
            else:
                max_x = x_boundary

            if int(max_y + 0.21*height) <= y_boundary:
                max_y = int(max_y + 0.21*height)
            else:
                max_y = y_boundary

            if int(min_x - 0.21*width) >= 0:
                min_x = int(min_x - 0.21*width)
            else:
                min_x = 0

            if int(min_y - 0.21*height) >= 0:
                min_y = int(min_y - 0.21*height)
            else:
                min_y = 0

            bounding_box = [min_x, max_x, min_y, max_y]
            bounding_boxes.append(bounding_box)
        i += 1

    return paths, keypoints_list, bounding_boxes

    # events = np.loadtxt(annotation_path, skiprows=2, delimiter=";",
    #                          usecols=0, dtype=int).tolist()
    # frame_num = np.loadtxt(annotation_path, skiprows=2, delimiter=";",
    #                          usecols=1, dtype=int).tolist()
    # paths = os.path.join(image_base_path*len(events),
    #                      [str(r) for r in events],
    #                      [str(r) for r in events] + "_("*len(events) +
    #                      [str(r) for r in frame_num] + ").jpg"*len(events))





if __name__ == "__main__":
    # coordinates = np.loadtxt("/home/mmc-user/ass3/dataset/annotations/train.csv", skiprows=2, delimiter=";",
    #            usecols=range(4, 55), dtype=int)
    # print("test")
    annotation_path = os.path.join(annotation_path, "train.csv")
    ds = load_dataset(annotation_path, image_base_path)
    print("test")

