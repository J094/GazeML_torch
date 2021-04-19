import torch
import numpy as np

from src.datasources.unityeyes import UnityEyesDataset
import src.models.gaze_modelbased as GM
import src.utils.gaze as gaze_util
import torch.utils.data as D


if __name__ == "__main__":

    elg_model = torch.load('./models/v0.2/model-v0.2-(36, 60)-epoch-89-loss-0.7151.pth')
    elg_model.eval()

    val_root = "E:/Datasets/UnityEyes_Windows/800x600/val"
    val_dataset = UnityEyesDataset(val_root, eye_image_shape=(36, 60), generate_heatmaps=True, random_difficulty=False)

    val_dataloader = D.DataLoader(val_dataset, batch_size=1)

    total_angular_loss = 0.0
    i = 0
    while i <= 40000:
        entry = next(iter(val_dataloader))
        eye_input = entry['eye'].cuda()

        landmarks_label = entry['landmarks'].cuda()
        radius_label = entry['radius'].cuda()
        gaze_label = entry['gaze'].cuda()

        heatmaps_predict, ldmks_predict, radius_predict = elg_model(eye_input)

        ldmks = ldmks_predict.cpu().detach().numpy()
        iris_ldmks = np.array(ldmks[0][0:8])
        iris_center = np.array(ldmks[0][-2])
        eyeball_center = np.array(ldmks[0][-1])
        eyeball_radius = radius_predict.cpu().detach().numpy()[0]
        radius_predict - torch.unsqueeze(radius_label, dim=0)

        gaze_predict = GM.estimate_gaze_from_landmarks(iris_ldmks, iris_center, eyeball_center, eyeball_radius)
        predict = gaze_predict.reshape(1, 2)
        label = gaze_label.cpu().detach().numpy()

        loss = np.mean(gaze_util.angular_error(predict, label))

        total_angular_loss += loss
        i += 1
        if i % 50 == 0:
            print(f'gaze_angular_loss_average_{i}_samples: ', total_angular_loss / (i))

    print('gaze_angular_loss_average: ', total_angular_loss / 40000)
