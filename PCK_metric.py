from config import *

def compute_PCK(predictions, annotations, threshold=0.1, device="cpu"):
    num_obs = predictions.shape[0]
    num_keypoints = predictions.shape[1]

    # compute each dmax
    lhip_id = keypoint_ids["lhip"]
    rsho_id = keypoint_ids["rsho"]

    lhip_x = annotations[:, lhip_id, 0]
    lhip_y = annotations[:, lhip_id, 1]
    rsho_x = annotations[:, rsho_id, 0]
    rsho_y = annotations[:, rsho_id, 1]

    d_max = threshold * ((lhip_x-rsho_x).pow(2) + (lhip_y-rsho_y).pow(2)).sqrt()
    d_max = d_max.repeat_interleave(num_keypoints)
    d_max = d_max.reshape(num_obs, num_keypoints)

    num_all_keypoints = torch.full((num_keypoints, ), annotations.size()[0]).to(device)
    tmp = torch.count_nonzero((annotations[:, :, 2] == 0), 0)
    num_all_keypoints -= tmp

    dist = (predictions[:, :, 0:2] - annotations[:, :, 0:2]).pow(2).sum(2).sqrt()
    dist -= d_max
    dist = torch.where(annotations[:,:,2]==0, abs(dist), dist)
    dist = torch.where((annotations[:,:,2]!=0) & (predictions[:,:,2]==0), abs(dist), dist)

    pck_per_class = torch.count_nonzero(dist < 0, dim = 0) / num_all_keypoints
    overall_pck = torch.count_nonzero(dist < 0) / num_all_keypoints.sum()

    return pck_per_class, overall_pck

if __name__ == "__main__":
    _, predictions, _ = load_dataset(os.path.join(os.getcwd(),"predictions.csv"), image_base_path, 2)
    _, true_vals, _ = load_dataset(os.path.join(annotation_path, "val.csv"), image_base_path, 4)

    predictions = torch.tensor(predictions).long()
    true_vals = torch.tensor(true_vals).long()

    print(compute_PCK(predictions, true_vals))
    #print(compute_PCK(predictions, true_vals, 0.2))





