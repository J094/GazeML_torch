
import torch
from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np

import src.models.gaze_modelbased as GM
import src.utils.gaze as gaze_util


def clip_eye_region(eye_region_landmarks, image):
    # Output size.
    oh, ow = 72, 120

    def process_coords(coords_list):
        return np.array([(x, y) for (x, y) in coords_list])

    def process_rescale_clip(eye_landmarks):
        eye_width = 1.5 * abs(eye_landmarks[0][0] - eye_landmarks[1][0])
        eye_middle = (eye_landmarks[0] + eye_landmarks[1]) / 2

        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = -eye_middle[0] + 0.5 * eye_width
        recentre_mat[1, 2] = -eye_middle[1] + 0.5 * oh / ow * eye_width

        scale_mat = np.asmatrix(np.eye(3))
        np.fill_diagonal(scale_mat, ow / eye_width)

        transform_mat = recentre_mat * scale_mat

        eye = cv.warpAffine(image, transform_mat[:2, :3], (ow, oh))
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        return eye, np.asarray(transform_mat)

    left_eye_landmarks = process_coords(eye_region_landmarks[2:4])
    right_eye_landmarks = process_coords(eye_region_landmarks[0:2])
    left_eye_image, left_transform_mat = process_rescale_clip(left_eye_landmarks)
    right_eye_image, right_transform_mat = process_rescale_clip(right_eye_landmarks)
    
    return [left_eye_image, left_transform_mat], [right_eye_image, right_transform_mat]

def estimate_gaze(eye_image, transform_mat, model, is_left: bool):
    eye_image = np.expand_dims(eye_image, -1)
    # Change format to NCHW.
    eye_image = np.transpose(eye_image, (2, 0, 1))
    eye_image = torch.unsqueeze(torch.Tensor(eye_image), dim=0)
    eye_input = eye_image.cuda()
    # Do predict by elg_model.
    heatmaps_predict, ldmks_predict, radius_predict = model(eye_input)
    # Get parameters for model_based gaze estimator.
    ldmks = ldmks_predict.cpu().detach().numpy()
    iris_ldmks = np.array(ldmks[0][0:8])
    iris_center = np.array(ldmks[0][-2])
    eyeball_center = np.array(ldmks[0][-1])
    eyeball_radius = radius_predict.cpu().detach().numpy()[0]
    # Predict gaze.
    gaze_predict = GM.estimate_gaze_from_landmarks(iris_ldmks, iris_center, eyeball_center, eyeball_radius)
    predict = gaze_predict.reshape(1, 2)
    iris_center = ldmks_predict[0].cpu().detach().numpy()[16]
    if is_left:
        iris_center[0] = 120 - iris_center[0]
    iris_center = (iris_center - [transform_mat[0][2], transform_mat[1][2]]) / transform_mat[0][0]
    return predict, iris_center




if __name__ == "__main__":
    # initialize dlib's face detector (mmod) and then create
    # the facial landmark predictor
    d = "./src/models/mmod_human_face_detector.dat"
    p = "./src/models/shape_predictor_5_face_landmarks.dat"
    detector = dlib.cnn_face_detection_model_v1(d)
    predictor = dlib.shape_predictor(p)
    elg_model = torch.load('./models/v0.2/model-v0.2-(36, 60)-epoch-89-loss-0.7151.pth')
    elg_model.eval()

    cap = cv.VideoCapture(0)

    while True:
        # load the input image and convert it to grayscale
        _, image = cap.read()
        image = cv.flip(image, 1)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # detect face in the grayscale image
        faceRects = detector(gray, 0)

        # loop over the face detections
        for (i, faceRect) in enumerate(faceRects):
            # show the face region
            # x1 = faceRect.rect.left()
            # y1 = faceRect.rect.top()
            # x2 = faceRect.rect.right()
            # y2 = faceRect.rect.bottom()
            # cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # determine the facial landmarks for the face ragion, then
            # convert the facial landmarks (x, y) to a NumPy array
            shape = predictor(gray, faceRect.rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y) for the eye-region landmarks
            # and draw them on the image
            for (j, (x, y)) in enumerate(shape):
                if j in range(0, 4):
                    cv.circle(image, (x, y), 2, (0, 255, 0), -1)
            # 0-1:Right to Left in Right Eye.
            # 2-3:Left to Right in Left Eye.
            eye_region_landmarks = shape[0:4]
            left_eye, right_eye = clip_eye_region(eye_region_landmarks, gray)
            # As this elg_model only train for right eyes, so need to do flip for left eyes before estimate.
            left_gaze, left_iris_center = estimate_gaze(
                cv.flip(left_eye[0], 1), 
                transform_mat=left_eye[1],
                model=elg_model,
                is_left=True,
            )
            # Change gaze respect to left eyes.
            left_gaze[0][1] = -left_gaze[0][1]
            right_gaze, right_iris_center = estimate_gaze(
                right_eye[0],
                transform_mat=right_eye[1], 
                model=elg_model,
                is_left=False,
            )
            image = gaze_util.draw_gaze(image, left_iris_center, left_gaze[0])
            image = gaze_util.draw_gaze(image, right_iris_center, right_gaze[0])

        # Show the output image with gaze direction.
        cv.imshow("Output", image)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
