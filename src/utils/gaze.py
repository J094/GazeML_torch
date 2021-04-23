"""Utility methods for gaze angle and error calculations."""
import cv2 as cv
import numpy as np

"""
输入：包含 (pitch, yaw) 的 np.array，shape 为 (n, 2)
输出：包含 (gx, gy, gz) 的 np.array，shape 为 (n, 3)，即针对每一对 (pitch, yaw) 的 gaze estimation
"""
def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


"""
输入：包含 (gx, gy, gz) 的 np.array，shape 为 (n, 3)
输出：包含 (pitch, yaw) 的 np.array，shape 为 (n, 2)，即通过 gaze direction 来计算 (pitch, yaw)
"""
def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    # 将向量标准化
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # phi
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # theta
    return out


"""
radians 转变成实际角度
"""
radians_to_degrees = 180.0 / np.pi

"""
利用余弦相似性来计算角误差
"""
def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return np.mean(angular_error(a, b))

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out
