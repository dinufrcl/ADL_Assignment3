import torch.nn

from training_config_2_5_2 import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from PCK_metric import *
from PCK_metric import compute_PCK
from visualization import *

def train_step(x_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    if device == 'cuda':
        with torch.autocast(device):
            y_hat = model.forward(x_batch)

            # with torch.no_grad():
            #     img = x_batch[0, :, :, :].to("cpu")
            #     npimg = convert_tensor_numpy(img)
            #     heatmaps = y_hat[0, :, :, :].to("cpu")
            #
            #
            #     plot_img_with_heatmaps(img, heatmaps)
            #
            #     max_act = torch.amax(y_hat, dim=[2, 3])
            #
            #     visible = torch.where(max_act > 0.2)
            #
            #     y_hat_flatten = y_hat.flatten(start_dim=2)
            #     _, max_ind = y_hat_flatten.max(-1)
            #     y_hat_coords = torch.stack([max_ind // 64, max_ind % 64], -1)  # ???
            #
            #     # transform coordinates back to the original image size
            #     y_hat_coords[:, :, [0, 1]] = y_hat_coords[:, :, [0, 1]] * 2  # ????
            #
            #     pad_func = torch.nn.ConstantPad1d((0, 1), 0)
            #     y_hat_coords = pad_func(y_hat_coords)
            #
            #     y_hat_coords[visible[0], visible[1], 2] = 1
            #
            #     y_hat_coords = y_hat_coords[:, :, [1, 0, 2]]
            #
            #     img = x_batch[0, :, :, :].to("cpu")
            #     coords = y_hat_coords[0,:,:].to("cpu")
            #     npimg = convert_tensor_numpy(img)
            #     plot_img_with_keypoints(npimg, coords)

            loss = loss_function(y_hat, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        y_hat = model.forward(x_batch)
        loss = loss_function(y_hat, y_batch)

        loss.backward()
        optimizer.step()

    # model.train()
    # optimizer.zero_grad()
    # x_batch = x_batch.to(device)
    # y_batch = y_batch.to(device)
    #
    # if device == 'cuda':
    #     with torch.autocast(device): # run with mixed precision
    #         y_hat = model.forward(x_batch)
    #         # https://stackoverflow.com/questions/69518359/pytorch-argmax-across-multiple-dimensions
    #         y_hat_flatten = y_hat.view(16, 16, -1)
    #         _, max_ind = y_hat_flatten.max(-1)
    #         y_hat = torch.stack([max_ind // 64, max_ind % 64], -1)  # ???
    #
    #
    #         y_hat[:, :, [0, 1]] = y_hat[:, :, [0, 1]] * 2  # ????
    #
    #         pad_func = torch.nn.ConstantPad1d((0,1), 1)
    #         y_hat = pad_func(y_hat)
    #
    #         loss = loss_function(y_hat.float(), y_batch.float())
    #
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    # else:
    #     y_hat = model.forward(x_batch)
    #     loss = loss_function(y_hat, y_batch)
    #
    #     loss.backward()
    #     optimizer.step()

def test_model(data_loader):
    model.eval()
    optimizer.zero_grad()
    pck_values = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model.forward(x_batch)

            # img = x_batch[0, :, :, :].to("cpu")
            # npimg = convert_tensor_numpy(img)
            # heatmaps = y_hat[0, :, :, :].to("cpu")
            # plot_img_with_heatmaps(img, heatmaps)

            #heatmaps = TF.resize(y_hat, [x_batch.shape[2], x_batch.shape[2]], interpolation=InterpolationMode.NEAREST)

            max_act = torch.amax(y_hat, dim=[2, 3]).to(device)
            visible = torch.where(max_act > 0.2)

            y_hat_flatten = y_hat.flatten(start_dim=2).to(device)
            _, max_ind = y_hat_flatten.max(-1)
            y_hat_coords = torch.stack([max_ind // 64, max_ind % 64], -1).to(device)  # ???

            # transform coordinates back to the original image size
            y_hat_coords[:, :, [0, 1]] = y_hat_coords[:, :, [0, 1]] * 2  # ????

            pad_func = torch.nn.ConstantPad1d((0, 1), 0)
            y_hat_coords = pad_func(y_hat_coords)

            y_hat_coords[visible[0], visible[1], 2] = 1

            y_hat_coords = y_hat_coords[:, :, [1,0,2]]

            # img = x_batch[0, :, :, :].to("cpu")
            # coords = y_hat_coords[0,:,:].to("cpu")
            # npimg = convert_tensor_numpy(img)
            # plot_img_with_keypoints(npimg, coords)

            _, pck = compute_PCK(y_hat_coords, y_batch, device=device)

            pck_values.append(pck)

    return torch.mean(torch.stack(pck_values))


def train(n_epochs, train_data_loader, validation_data_loader):
    for epoch in tqdm(range(n_epochs)):
        for x_batch, y_batch in train_data_loader:
            train_step(x_batch, y_batch)
        pck = test_model(validation_data_loader)
        writer.add_scalar('PCK/validation', pck, epoch+performed_epochs)

        training_history.append(pck)

        # save model weights and optimizer state if best performance on validation set
        if pck == max(training_history):
            torch.save(model.state_dict(), save_path + "_model_weights.pth")
            torch.save(optimizer.state_dict(), save_path + "_optimizer_state.pth")

        # save for backup
        torch.save(model.state_dict(), save_path2 + "_model_weights.pth")
        torch.save(optimizer.state_dict(), save_path2 + "_optimizer_state.pth")

    writer.close()

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(0)

    paths, keypoints, bounding_boxes = load_dataset(os.path.join(annotation_path, "train.csv"), image_base_path)
    train_dataset = SkijumpDataset(paths,
                                   keypoints.copy(),
                                   bounding_boxes.copy(),
                                   img_size=128,
                                   heatmap_size=64,
                                   use_augment=True,
                                   use_heatmap=True,
                                   use_random_scale=use_random_scale)

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=16,
                                   num_workers=num_workers,
                                   shuffle=True,
                                   drop_last=True,
                                   worker_init_fn=seed_worker,
                                   generator=g
                                   )

    paths, keypoints, bounding_boxes = load_dataset(os.path.join(annotation_path, "val.csv"), image_base_path)
    val_dataset = SkijumpDataset(paths,
                                 keypoints,
                                 bounding_boxes,
                                 img_size=128,
                                 use_heatmap=False,
                                 use_augment=False)

    validation_data_loader = DataLoader(dataset=val_dataset,
                                   batch_size=16,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   drop_last= False,
                                   worker_init_fn=seed_worker,
                                   generator=g)

    if continue_training:
        # load from backup
        model.load_state_dict(torch.load(save_path2 + "_model_weights.pth"))
        optimizer.load_state_dict(torch.load(save_path2 + "_optimizer_state.pth"))

    train(n_epochs=n_epochs, train_data_loader=train_data_loader, validation_data_loader=validation_data_loader)

