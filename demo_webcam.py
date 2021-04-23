
import torch
from imutils import face_utils
import dlib
import cv2 as cv
import numpy as np

import src.models.gaze_modelbased as GM
import src.utils.gaze as gaze_util

def clip_eye_region(eye_region_landmarks, image, image_shape):
    # Input size.
    ih, iw = image_shape
    # Output size.
    oh, ow = 36, 60

    def process_coords(coords_list):
        return np.array([(x, y) for (x, y) in coords_list])

    def process_rescale_clip(eye_landmarks):
        eye_width = 1.5 * abs(left_eye_landmarks[0][0] - left_eye_landmarks[1][0])
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
        return eye, eye_middle

    left_eye_landmarks = process_coords(eye_region_landmarks[2:4])
    right_eye_landmarks = process_coords(eye_region_landmarks[0:2])
    left_eye_image, left_middle = process_rescale_clip(left_eye_landmarks)
    right_eye_image, right_middle = process_rescale_clip(right_eye_landmarks)
    
    return [left_eye_image, left_middle], [right_eye_image, right_middle]

def estimate_gaze(eye_image):
    elg_model = torch.load('./models/v0.2/model-v0.2-(36, 60)-epoch-89-loss-0.7151.pth')
    elg_model.eval()
    eye_image = np.expand_dims(eye_image, -1)
    # Change format to NCHW.
    eye_image = np.transpose(eye_image, (2, 0, 1))
    eye_image = torch.unsqueeze(torch.Tensor(eye_image), dim=0)
    eye_input = eye_image.cuda()
    # Do predict by elg_model.
    heatmaps_predict, ldmks_predict, radius_predict = elg_model(eye_input)
    # Get parameters for model_based gaze estimator.
    ldmks = ldmks_predict.cpu().detach().numpy()
    iris_ldmks = np.array(ldmks[0][0:8])
    iris_center = np.array(ldmks[0][-2])
    eyeball_center = np.array(ldmks[0][-1])
    eyeball_radius = radius_predict.cpu().detach().numpy()[0]
    # Predict gaze.
    gaze_predict = GM.estimate_gaze_from_landmarks(iris_ldmks, iris_center, eyeball_center, eyeball_radius)
    predict = gaze_predict.reshape(1, 2)
    return predict




if __name__ == "__main__":
    # initialize dlib's face detector (mmod) and then create
    # the facial landmark predictor
    d = "./src/models/mmod_human_face_detector.dat"
    p = "./src/models/shape_predictor_5_face_landmarks.dat"
    detector = dlib.cnn_face_detection_model_v1(d)
    predictor = dlib.shape_predictor(p)

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
            # for (j, (x, y)) in enumerate(shape):
            #     if j in range(0, 4):
            #         cv.circle(image, (x, y), 2, (0, 255, 0), -1)
            # 0-1:Right to Left in Right Eye.
            # 2-3:Left to Right in Left Eye.
            eye_region_landmarks = shape[0:4]
            left_eye, right_eye = clip_eye_region(eye_region_landmarks, gray, gray.shape)
            # As this elg_model only train for right eyes, so need to do flip for left eyes before estimate.
            left_gaze = estimate_gaze(cv.flip(left_eye[0], 1))
            # Change gaze respect to left eyes.
            left_gaze[0][1] = -left_gaze[0][1]
            right_gaze = estimate_gaze(right_eye[0])
            image = gaze_util.draw_gaze(image, left_eye[1], left_gaze[0])
            image = gaze_util.draw_gaze(image, right_eye[1], right_gaze[0])

        # Show the output image with gaze direction.
        cv.imshow("Output", image)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
