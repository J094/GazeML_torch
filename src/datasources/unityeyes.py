"""UnityEyes data source for gaze estimation."""
import os
import threading
import time

import torch
import cv2 as cv
import numpy as np
import json

import src.utils.gaze as gaze_util
import src.utils.heatmap as heatmap_util


class UnityEyesDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 root,
                 eye_image_shape=(36, 60),
                 generate_heatmaps=False,
                 heatmaps_scale=1.0,
                 random_difficulty=False):
        super(UnityEyesDataset).__init__()

        self.root = root
        self.current_index = 0
        self.file_stems = sorted([p[:-5] for p in os.listdir(root)
                                  if p.endswith('.json')], key=lambda x: int(x))
        self.num_entries = len(self.file_stems)

        self.eye_image_shape = eye_image_shape
        self.heatmaps_scale = heatmaps_scale
        self.generate_heatmaps = generate_heatmaps

        self.difficulty = 0.0  # 0.0 to 1.0
        self.augmentation_ranges = {  # (easy, hard)
            'translation': (2.0, 10.0),
            'rotation': (0.1, 2.0),
            'intensity': (0.5, 20.0),
            'blur': (0.1, 1.0),
            'scale': (0.01, 0.1),
            'rescale': (1.0, 0.5),
            'num_line': (0.0, 2.0),
            'heatmap_sigma': (7.5, 2.5),
        }

        self.random_difficulty = random_difficulty

        self.lock = threading.Lock()

    def __iter__(self):
        # To see if yield can resolve memory leaky????
        return self.entry_generator()

    def entry_generator(self):
        """Read entry from UnityEyes."""
        try:
            while True:
                # Give random difficulty for every entry.
                if self.random_difficulty:
                    difficulty = np.random.rand()
                    self.set_difficulty(difficulty)
                # start_time = time.clock()

                self.lock.acquire()

                idx = self.current_index
                self.current_index += 1

                # Make it an endless data generator.
                if self.current_index >= len(self.file_stems):
                    self.current_index = 0

                file_stem = self.file_stems[idx]
                jpg_path = os.path.join(self.root, f'{file_stem}.jpg')
                json_path = os.path.join(self.root, f'{file_stem}.json')
                if not os.path.isfile(jpg_path) or not os.path.isfile(json_path):
                    continue
                with open(json_path, 'r') as f:
                    json_data = json.load(f)  # json_data is a dict type data
                entry = {
                    # Single channel gray image
                    'full_image': cv.imread(jpg_path, cv.IMREAD_GRAYSCALE),
                    'json_data': json_data,
                }
                assert entry['full_image'] is not None

                self.lock.release()

                # thr = threading.Thread(target=self.preprocess_entry, args=(entry,))
                # entry = thr.start()
                # thr.join()
                entry = self.preprocess_entry(entry)

                # print(time.clock() - start_time)

                if entry is None:
                    continue
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def set_difficulty(self, difficulty):
        """Set difficulty of training data."""
        assert isinstance(difficulty, float)
        assert 0.0 <= difficulty <= 1.0
        self.difficulty = difficulty

    def set_augmentation_range(self, augmentation_type, easy_value, hard_value):
        """Set 'range' for a known augmentation type."""
        assert isinstance(augmentation_type, str)
        assert augmentation_type in self.augmentation_ranges
        assert isinstance(easy_value, float) or isinstance(easy_value, int)
        assert isinstance(hard_value, float) or isinstance(hard_value, int)
        self.augmentation_ranges[augmentation_type] = (easy_value, hard_value)

    # Try to use PyTorch to preprocess, and get GPU acceleration.
    def preprocess_entry(self, entry):
        """Use annotations to segment eyes and calculate gaze direction."""
        full_image = entry['full_image']
        json_data = entry['json_data']
        del entry['full_image']
        del entry['json_data']

        ih, iw = full_image.shape
        iw_2, ih_2 = 0.5 * iw, 0.5 * ih
        oh, ow = self.eye_image_shape

        # The original y-coord is using normal coordinate system, not for image.
        def process_coords(coords_list):
            # eval(l) receive a str-expression and return the value of this expression.
            coords = [eval(e) for e in coords_list]
            return np.array([(x, ih-y, z) for (x, y, z) in coords])
        interior_landmarks = process_coords(json_data['interior_margin_2d'])
        caruncle_landmarks = process_coords(json_data['caruncle_2d'])
        iris_landmarks = process_coords(json_data['iris_2d'])

        random_multipliers = []

        def value_from_type(augmentation_type):
            # Scale to be in range
            easy_value, hard_value = self.augmentation_ranges[augmentation_type]
            value = (hard_value - easy_value) * self.difficulty + easy_value
            # Ensure value in reasonable range
            value = (np.clip(value, easy_value, hard_value)
                     if easy_value < hard_value
                     else np.clip(value, hard_value, easy_value))
            return value

        def noisy_value_from_type(augmentation_type):
            # Get normal distributed random value
            if len(random_multipliers) == 0:
                # If random_multipliers is empty, add another 8 multipliers in random_multipliers.
                random_multipliers.extend(
                    list(np.random.normal(size=(len(self.augmentation_ranges),))))
            # Every time pop one normal distribution value as multipliers.
            return random_multipliers.pop() * value_from_type(augmentation_type)

        # Only select almost frontal images
        h_pitch, h_yaw, _ = eval(json_data['head_pose'])
        if h_pitch > 180.0:  # Need to correct pitch
            h_pitch -= 360.0
        h_yaw -= 180.0  # Need to correct yaw
        if abs(h_pitch) > 30 or abs(h_yaw) > 30:
            return None

        # Prepare to segment eye image
        left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
        right_corner = interior_landmarks[8, :2]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                              np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

        # Centre axes to eyeball centre
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-iw_2], [-ih_2]]

        # Rotate eye image if requested
        rotate_mat = np.asmatrix(np.eye(3))
        rotation_noise = noisy_value_from_type('rotation')
        if rotation_noise > 0:
            # Rotation respect to z-axis.
            rotate_angle = np.radians(rotation_noise)
            cos_rotate = np.cos(rotate_angle)
            sin_rotate = np.sin(rotate_angle)
            rotate_mat[0, 0] = cos_rotate
            rotate_mat[0, 1] = -sin_rotate
            rotate_mat[1, 0] = sin_rotate
            rotate_mat[1, 1] = cos_rotate

        # Scale image to fit output dimensions (with a little bit of noise)
        scale_mat = np.asmatrix(np.eye(3))
        scale = 1. + noisy_value_from_type('scale')
        scale_inv = 1. / scale
        np.fill_diagonal(scale_mat, ow / eye_width * scale)
        original_eyeball_radius = 71.7593
        eyeball_radius = original_eyeball_radius * scale_mat[0, 0]  # See: https://goo.gl/ZnXgDE
        entry['radius'] = np.float32(eyeball_radius)

        # Re-centre eye image such that eye fits (based on determined `eye_middle`)
        recentre_mat = np.asmatrix(np.eye(3))
        # Re-centre need to respect original coordinate system.
        recentre_mat[0, 2] = iw/2 - eye_middle[0] + 0.5 * eye_width * scale_inv
        recentre_mat[1, 2] = ih/2 - eye_middle[1] + 0.5 * oh / ow * eye_width * scale_inv
        recentre_mat[0, 2] += noisy_value_from_type('translation')  # x
        recentre_mat[1, 2] += noisy_value_from_type('translation')  # y

        # Apply transforms
        transform_mat = recentre_mat * scale_mat * rotate_mat * translate_mat
        eye = cv.warpAffine(full_image, transform_mat[:2, :3], (ow, oh))

        # Convert look vector to gaze direction in polar angles
        look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3]
        # Change the direction of x-axis to convenient the calculation of (pitch, yaw)
        # In image coordinate system
        look_vec[0] = -look_vec[0]
        original_gaze = gaze_util.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        look_vec = rotate_mat * look_vec.reshape(3, 1)
        gaze = gaze_util.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        if gaze[1] > 0.0:
            gaze[1] = np.pi - gaze[1]
        elif gaze[1] < 0.0:
            gaze[1] = -(np.pi + gaze[1])
        entry['gaze'] = gaze.astype(np.float32)

        # Draw line randomly
        num_line_noise = int(np.round(noisy_value_from_type('num_line')))
        if num_line_noise > 0:
            line_rand_nums = np.random.rand(5 * num_line_noise)
            for i in range(num_line_noise):
                j = 5 * i
                lx0, ly0 = int(ow * line_rand_nums[j]), oh
                lx1, ly1 = ow, int(oh * line_rand_nums[j + 1])
                direction = line_rand_nums[j + 2]
                if direction < 0.25:
                    lx1 = ly0 = 0
                elif direction < 0.5:
                    lx1 = 0
                elif direction < 0.75:
                    ly0 = 0
                line_colour = int(255 * line_rand_nums[j + 3])
                eye = cv.line(eye, (lx0, ly0), (lx1, ly1),
                              color=(line_colour, line_colour, line_colour),
                              thickness=max(1, int(6*line_rand_nums[j + 4])),
                              lineType=cv.LINE_AA)

        # Rescale image if required
        rescale_max = value_from_type('rescale')
        if rescale_max < 1.0:
            rescale_noise = np.random.uniform(low=rescale_max, high=1.0)
            interpolation = cv.INTER_CUBIC
            eye = cv.resize(eye, dsize=(0, 0), fx=rescale_noise, fy=rescale_noise,
                            interpolation=interpolation)
            eye = cv.equalizeHist(eye)
            eye = cv.resize(eye, dsize=(ow, oh), interpolation=interpolation)

        # Add rgb noise to eye image (I think it's gray noise?)
        intensity_noise = int(value_from_type('intensity'))
        if intensity_noise > 0:
            eye = eye.astype(np.int16)
            eye += np.random.randint(low=-intensity_noise, high=intensity_noise,
                                     size=eye.shape, dtype=np.int16)
            cv.normalize(eye, eye, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            eye = eye.astype(np.uint8)

        # Add blur to eye image
        blur_noise = noisy_value_from_type('blur')
        if blur_noise > 0:
            eye = cv.GaussianBlur(eye, (7, 7), 0.5 + np.abs(blur_noise))

        # Histogram equalization and preprocessing for NN
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1)
        # Transfer to channels first for PyTorch
        entry['eye'] = np.transpose(eye, (2, 0, 1))

        # Select and transform landmark coordinates
        iris_centre = np.asarray([
            iw_2 + original_eyeball_radius * -np.cos(original_gaze[0]) * np.sin(original_gaze[1]),
            ih_2 + original_eyeball_radius * -np.sin(original_gaze[0]),
            ])
        landmarks = np.concatenate([interior_landmarks[::2, :2],  # 8
                                    iris_landmarks[::4, :2],  # 8
                                    iris_centre.reshape((1, 2)),
                                    [[iw_2, ih_2]],  # Eyeball centre
                                    ])  # 18 in total
        # Pad 1 dimension for matrix multiplication.
        landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant',
                                       constant_values=1))
        # Do a matrix multiplication with transform_mat.T to get landmarks in new coordinate system.
        landmarks = np.asarray(landmarks * transform_mat.T)
        landmarks = landmarks[:, :2]  # We only need x, y
        entry['landmarks'] = landmarks.astype(np.float32)

        # Generate heatmaps if necessary
        if self.generate_heatmaps:
            # Should be half-scale (compared to eye image)
            entry['heatmaps'] = np.asarray([
                heatmap_util.gaussian_2d(
                    shape=(self.heatmaps_scale*oh, self.heatmaps_scale*ow),
                    centre=self.heatmaps_scale*landmark,
                    sigma=value_from_type('heatmap_sigma'),
                )
                for landmark in entry['landmarks']
            ]).astype(np.float32)
            # c x h x w, we need to transfer it to 'NHWC' channels last, for PyTorch don't need
            # entry['heatmaps'] = np.transpose(entry['heatmaps'], (1, 2, 0))
        return entry
