import sys


def train(batch_size, eye_image_shape, epochs, version):

    import torch
    from src.models.gaze_fc import GazeFC
    gaze_model = GazeFC().cuda()
    # elg_model = torch.load("./models/...")

    from src.datasources.unityeyes import UnityEyesDataset
    train_root = "E:/Datasets/UnityEyes_Windows/800x600/train"
    # train_root = "/home/junguo/Datasets/UnityEyes/800x600/train"
    train_dataset = UnityEyesDataset(train_root, eye_image_shape=eye_image_shape, generate_heatmaps=True, random_difficulty=True)
    val_root = "E:/Datasets/UnityEyes_Windows/800x600/val"
    # val_root = "/home/junguo/Datasets/UnityEyes/800x600/val"
    val_dataset = UnityEyesDataset(val_root, eye_image_shape=eye_image_shape, generate_heatmaps=True, random_difficulty=True)

    start_epoch = 1
    initial_learning_rate = 1e-4

    from src.trainers.gaze_trainer import GazeTrainer
    gaze_trainer = GazeTrainer(model=gaze_model,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               initial_learning_rate=initial_learning_rate,
                               epochs=epochs,
                               start_epoch=start_epoch,
                               batch_size=batch_size,
                               version=version)

    gaze_trainer.run()


if __name__ == "__main__":
    batch_size = eval(sys.argv[1])
    shape_multiplier = eval(sys.argv[2])
    epochs = eval(sys.argv[3])

    eye_image_shape = (36*shape_multiplier, 60*shape_multiplier)

    train(batch_size, eye_image_shape, epochs, version=f'v0.2-Gaze')