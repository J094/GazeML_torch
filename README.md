# GazeML_torch

Reproduction of https://github.com/swook/GazeML (paper: https://arxiv.org/abs/1805.04771v1) by PyTorch.

## Requirement

```
pytorch (cuda)
tensorboard
numpy
opencv
imutils
dlib
scipy
```

## Example

![GazeML_torch_v0.2_jun](exp1.gif)

![GazeML_torch_v0.2_jiabin](exp2.gif)

## Webcam Demo

```
python demo_webcam.py
```

dlib models: https://github.com/davisking/dlib-models

I use dlib mmod_human_face_detector to detect face region.

Then, i use dlib shape_predictor_5_landmarks to get eye landmarks for clippling eye region.

After that, i use the model_based model from https://arxiv.org/abs/1805.04771v1 to estimate Gaze.
