from config import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# todo: keypoints mit verschiedenen Farben ausgeben
def plot_img_with_keypoints(img, keypoints):
    num_keypoints = keypoints.shape[0]

    color = iter(cm.rainbow(np.linspace(0, 1, num_keypoints)))
    classes = np.array(list(keypoint_colors.keys()))
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(num_keypoints):
        if keypoints[i, 2] != 0:
            c = next(color)
            ax.scatter(keypoints[i, 0], keypoints[i, 1], c=c, label=classes[i], edgecolors='none')
    ax.legend()

    plt.show()





    # # select visible keypoints
    # visibility = np.where(keypoints[:, 2] != 0)[0]
    # keypoints_selected = keypoints[visibility][:, [0, 1]]
    #
    # classes = np.array(list(keypoint_colors.keys()))[visibility]
    # colors = np.array(list(keypoint_colors.values()))[visibility]
    # colors = np.random.choice(range(0,9999), classes.shape[0], replace=False)
    #
    # color = iter(cm.rainbow(np.linspace(0, 1, n)))
    # for i in range(n):
    #     c = next(color)
    #     plt.plot(x, y, c=c)
    #
    # i = 0
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # color = iter(cm.rainbow(np.linspace(0, 1, n)))
    # for cl in classes:
    #     ax.scatter(keypoints_selected[i,0], keypoints_selected[i,1], c=colors[i], label=cl, edgecolors='none') # alpha=0.3
    #     i += 1
    # ax.legend()


def convert_tensor_numpy(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    return npimg

def plot_img_with_heatmaps(img, heatmaps):
    npimg = convert_tensor_numpy(img)

    if heatmaps.shape[1] < npimg.shape[1]:
        # todo: Heatmap Punkte werden sehr groß
        heatmaps = TF.resize(heatmaps, [npimg.shape[1], npimg.shape[1]], interpolation=InterpolationMode.NEAREST)

    heatmaps = heatmaps.numpy()
    heatmaps = np.transpose(heatmaps, (1, 2, 0))

    # fig = plt.figure()
    # fig.add_subplot(4, 5, 1)
    # plt.imshow(npimg)
    # plt.axis('off')
    #
    # for i in range(heatmaps.shape[2]):
    #     fig.add_subplot(4, 5, i+2)
    #     plt.imshow(heatmaps[:, :, i], 'jet', interpolation='none', alpha=0.5)
    #     plt.axis('off')
    #
    # plt.show()


    plt.figure()
    plt.imshow(npimg)
    for i in range(heatmaps.shape[2]):
        plt.imshow(heatmaps[:, :, i], 'jet', interpolation='none', alpha=0.1)
    plt.show()
