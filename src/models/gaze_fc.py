import torch
import torch.nn as nn

import src.models.submodules as S


class GazeFC(torch.nn.Module):
    """Build elg model"""
    def __init__(self, first_layer_stride=1, num_rbs=1, num_hgs=4, num_modules=3,
                 num_feature_maps=32, num_landmarks=18):
        super(GazeFC, self).__init__()

        self.flatten = nn.Flatten()
        self.linearGazeBefore = S.Linear(36, 100)
        self.linearGaze = nn.ModuleDict()
        for i in range(3):
            self.linearGaze[f'{i}'] = S.Linear(100, 100)
        self.linearGazeAfter = S.Linear(100, 2)

    def forward(self, x):
        # Replace iris_center as gaze_prior by (iris_center - eyeball_center)
        for e in range(len(x)):
            eye_width = torch.sqrt(torch.mean(torch.square(x[e][0] - x[e][4])))
            x[e] = x[e] / eye_width
            gaze_prior = x[e][-2] - x[e][-1]
            x[e][-2] = gaze_prior
        x = self.flatten(x)
        x = self.linearGazeBefore(x)
        for i in range(3):
            x = self.linearGaze[f'{i}'](x)
        gaze = self.linearGazeAfter(x)
        return gaze
