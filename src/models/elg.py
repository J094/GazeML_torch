import torch
import torch.nn as nn

import src.models.submodules as S


class ELG(torch.nn.Module):
    """Build elg model"""
    def __init__(self, first_layer_stride=1, num_rbs=1, num_hgs=4, num_modules=3,
                 num_feature_maps=32, num_landmarks=18):
        super(ELG, self).__init__()
        self.num_modules = num_modules

        self.convPre = S.Conv(1, num_feature_maps, 7, first_layer_stride)
        self.residualBlockPre1 = S.ResidualBlock(num_feature_maps, 2*num_feature_maps)
        self.residualBlockPre2 = S.ResidualBlock(2*num_feature_maps, num_feature_maps)

        self.hourglassBlock = nn.ModuleDict()
        self.hourglassAfter = nn.ModuleDict()
        for i in range(num_modules):
            self.hourglassBlock[f'{i}'] = S.HourglassBlock(num_hgs, num_feature_maps, num_rbs)
            self.hourglassAfter[f'{i}'] = S.HourglassAfter(num_feature_maps, num_landmarks, num_rbs)

        self.calLandmarks = S.CalLandmarks(num_landmarks)
        self.flatten = nn.Flatten()
        self.linearRadiusBefore = S.Linear(2*num_landmarks, 100)
        self.linearRadius = nn.ModuleDict()
        for i in range(3):
            self.linearRadius[f'{i}'] = S.Linear(100, 100)
        self.linearRadiusAfter = nn.Linear(100, 1)

        # self.linearGazeBefore = S.Linear(2*num_landmarks+1, 100)
        # self.linearGaze = nn.ModuleDict()
        # for i in range(3):
        #     self.linearGaze[f'{i}'] = S.Linear(100, 100)
        # self.linearGazeAfter = S.Linear(100, 2)

    def forward(self, x):
        x = self.convPre(x)
        x = self.residualBlockPre1(x)
        x = self.residualBlockPre2(x)
        x_prev = x
        h = None
        for i in range(self.num_modules):
            x = self.hourglassBlock[f'{i}'](x)
            x, h = self.hourglassAfter[f'{i}'](x_prev, x, do_merge=(i < (self.num_modules - 1)))
            x_prev = x
        heatmaps = h
        ldmks = self.calLandmarks(heatmaps)
        # Don't need transpose, it's already NCHW
        # x = torch.transpose(ldmks, 1, 2)
        x = self.flatten(ldmks)
        x = self.linearRadiusBefore(x)
        for i in range(3):
            x = self.linearRadius[f'{i}'](x)
        radius = self.linearRadiusAfter(x)

        # Don't need transpose, it's already NCHW
        # x = torch.transpose(ldmks, 1, 2)
        # x = self.flatten(ldmks)
        # x = torch.cat((x, radius), dim=1)
        # x = self.linearGazeBefore(x)
        # for i in range(3):
        #     x = self.linearGaze[f'{i}'](x)
        # gaze = self.linearGazeAfter(x)
        return heatmaps, ldmks, radius
