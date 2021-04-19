import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batchNorm2d = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.batchNorm2d(x)
        x = self.conv(x)
        x = F.relu(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.batchNorm1d = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x = self.batchNorm1d(x)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        half_out_channels = max(int(out_channels/2), 1)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convLow1 = Conv(in_channels, half_out_channels, 1, 1)
        self.convLow2 = Conv(half_out_channels, half_out_channels, 3, 1)
        self.convLow3 = Conv(half_out_channels, out_channels, 1, 1)
        self.convUp = Conv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        # Lower
        c = x
        c = self.convLow1(c)
        c = self.convLow2(c)
        c = self.convLow3(c)
        # Upper
        s = x
        if self.in_channels != self.out_channels:
            s = self.convUp(s)
        x = c + s
        return x


class HourglassBlock(nn.Module):
    def __init__(self, steps_to_go, num_channels, num_rbs):
        super(HourglassBlock, self).__init__()
        self.steps_to_go = steps_to_go
        self.num_channels = num_channels
        self.num_rbs = num_rbs

        self.residualBlock = nn.ModuleDict()
        self.build_hg(self.steps_to_go, self.num_channels, self.num_rbs)

    def build_hg(self, steps_to_go, num_channels, num_rbs):
        # Upper
        self.residualBlock[f'{steps_to_go}'] = nn.ModuleDict()
        self.residualBlock[f'{steps_to_go}']['up'] = nn.ModuleDict()
        for i in range(num_rbs):
            self.residualBlock[f'{steps_to_go}']['up'][f'{i}'] = ResidualBlock(num_channels, num_channels)
        # Lower
        # RB1
        self.residualBlock[f'{steps_to_go}']['low'] = nn.ModuleDict()
        self.residualBlock[f'{steps_to_go}']['low'][f'{1}'] = nn.ModuleDict()
        for i in range(num_rbs):
            self.residualBlock[f'{steps_to_go}']['low'][f'{1}'][f'{i}'] = ResidualBlock(num_channels, num_channels)
        # Recursive
        if steps_to_go > 1:
            self.build_hg(steps_to_go-1, num_channels, num_rbs)
        else:
            # RB2
            self.residualBlock[f'{steps_to_go}']['low'][f'{2}'] = nn.ModuleDict()
            for i in range(num_rbs):
                self.residualBlock[f'{steps_to_go}']['low'][f'{2}'][f'{i}'] = ResidualBlock(num_channels, num_channels)
        # RB3
        self.residualBlock[f'{steps_to_go}']['low'][f'{3}'] = nn.ModuleDict()
        for i in range(num_rbs):
            self.residualBlock[f'{steps_to_go}']['low'][f'{3}'][f'{i}'] = ResidualBlock(num_channels, num_channels)

    def forward(self, x):
        return self.forward_hg(x, self.steps_to_go, self.num_channels, self.num_rbs)

    def forward_hg(self, x, steps_to_go, num_channels, num_rbs):
        # Upper
        up1 = x
        for i in range(num_rbs):
            up1 = self.residualBlock[f'{steps_to_go}']['up'][f'{i}'](up1)
        # Lower
        low1 = F.max_pool2d(x, kernel_size=2)
        # RB1
        for i in range(num_rbs):
            low1 = self.residualBlock[f'{steps_to_go}']['low'][f'{1}'][f'{i}'](low1)
        # Recursive
        if steps_to_go > 1:
            low2 = self.forward_hg(low1, steps_to_go-1, num_channels, num_rbs)
        else:
            # RB2
            low2 = low1
            for i in range(num_rbs):
                low2 = self.residualBlock[f'{steps_to_go}']['low'][f'{2}'][f'{i}'](low2)
        # RB3
        low3 = low2
        for i in range(num_rbs):
            low3 = self.residualBlock[f'{steps_to_go}']['low'][f'{3}'][f'{i}'](low3)
        # Upsampling
        low4 = F.interpolate(low3, size=up1.shape[2:], mode='bilinear', align_corners=True)
        # Add upper and lower branch
        x = up1 + low4
        return x


class HourglassAfter(nn.Module):
    def __init__(self, num_feature_maps, num_landmarks, num_rbs):
        super(HourglassAfter, self).__init__()
        self.num_rbs = num_rbs

        self.residualBlock = nn.ModuleDict()
        for i in range(num_rbs):
            self.residualBlock[f'{i}'] = ResidualBlock(num_feature_maps, num_feature_maps)
        self.convF = Conv(num_feature_maps, num_feature_maps, 1, 1)
        self.convH = Conv(num_feature_maps, num_landmarks, 1, 1)
        self.convM1 = Conv(num_landmarks, num_feature_maps, 1, 1)
        self.convM2 = Conv(num_feature_maps, num_feature_maps, 1, 1)

    def forward(self, x_prev, x_now, do_merge=True):
        # After hourglass
        for i in range(self.num_rbs):
            x_now = self.residualBlock[f'{i}'](x_now)
        # Feature_maps
        x_now = self.convF(x_now)
        # Heatmaps
        h = self.convH(x_now)
        # Save feature_maps for next stack of hourglass
        x_next = x_now
        # Merge heatmaps and feature_maps and prev_feature_maps
        if do_merge:
            h_merge_1 = self.convM1(h)
            h_merge_2 = self.convM2(x_now)
            h_merged = h_merge_1 + h_merge_2
            x_next = x_prev + h_merged
        return x_next, h


class CalLandmarks(nn.Module):
    def __init__(self, num_landmarks):
        super(CalLandmarks, self).__init__()
        self.num_landmarks = num_landmarks

    def forward(self, x):
        _, _, h, w = x.shape
        # Assume normalized coordinate [0, 1] for numeric stability
        ref_ys, ref_xs = torch.meshgrid(torch.linspace(0, 1.0, steps=h),
                                        torch.linspace(0, 1.0, steps=w))
        ref_xs = torch.reshape(ref_xs, (-1, h*w)).cuda()
        ref_ys = torch.reshape(ref_ys, (-1, h*w)).cuda()
        # Assuming NHWC, for PyTorch it's NCHW, don't need transpose
        beta = 1e2
        # Transpose x from NHWC to NCHW
        # x = torch.transpose(x, 1, 3)
        # x = torch.transpose(x, 2, 3)
        x = torch.reshape(x, (-1, self.num_landmarks, h*w))
        x = F.softmax(beta*x, dim=-1)
        lmrk_xs = torch.sum(ref_xs * x, dim=2)
        lmrk_ys = torch.sum(ref_ys * x, dim=2)
        # Return to actual coordinates ranges
        # The label heatmaps + 0.5px, so we only need + 0.5 here
        return torch.stack([
            lmrk_xs * (w - 1.0) + 0.5,
            lmrk_ys * (h - 1.0) + 0.5
        ], dim=2)  # N x 18 x 2
