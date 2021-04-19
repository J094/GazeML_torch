import numpy as np
import scipy.optimize


def estimate_gaze_from_landmarks(iris_landmarks, iris_centre, eyeball_centre, eyeball_radius,
                                 initial_gaze=None):
    """Given iris edge landmarks and other coordinates, estimate gaze direction.
    More correctly stated, estimate gaze from iris edge landmark coordinates, iris centre
    coordinates, eyeball centre coordinates, and eyeball radius in pixels.
    """
    e_x0, e_y0 = eyeball_centre
    i_x0, i_y0 = iris_centre

    if initial_gaze is not None:
        theta, phi = initial_gaze
        # theta = -theta
    else:
        theta = np.arcsin(np.clip((i_y0 - e_y0) / (-eyeball_radius), -1.0, 1.0))
        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))

    delta = 0.1 * np.pi
    if iris_landmarks[0, 0] < iris_landmarks[4, 0]:  # flipped
        alphas = np.flip(np.arange(0.0, 2.0 * np.pi, step=np.pi/4.0), axis=0)
    else:
        alphas = np.arange(-np.pi, np.pi, step=np.pi/4.0) + np.pi/4.0
    sin_alphas = np.sin(alphas)
    cos_alphas = np.cos(alphas)

    def gaze_fit_loss_func(inputs):
        theta, phi, delta, phase = inputs
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        # sin_alphas_shifted = np.sin(alphas + phase)
        sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = np.cos(alphas + phase)
        cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        # x = -np.cos(theta + delta * sin_alphas_shifted)
        x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x *= np.sin(phi + delta * cos_alphas_shifted)
        x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        x = x1 * x2
        # y = np.sin(theta + delta * sin_alphas_shifted)
        y1 = sin_theta * cos_delta_sin
        y2 = cos_theta * sin_delta_sin
        y = y1 + y2

        ix = e_x0 + eyeball_radius * x
        iy = e_y0 + eyeball_radius * y
        dx = ix - iris_landmarks[:, 0]
        dy = iy - iris_landmarks[:, 1]
        out = np.mean(dx ** 2 + dy ** 2)

        # In addition, match estimated and actual iris centre
        iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        out += iris_dx ** 2 + iris_dy ** 2

        # sin_alphas_shifted = sin_alphas * cos_phase + cos_alphas * sin_phase
        # cos_alphas_shifted = cos_alphas * cos_phase - sin_alphas * sin_phase
        dsin_alphas_shifted_dphase = -sin_alphas * sin_phase + cos_alphas * cos_phase
        dcos_alphas_shifted_dphase = -cos_alphas * sin_phase - sin_alphas * cos_phase

        # sin_delta_sin = np.sin(delta * sin_alphas_shifted)
        # sin_delta_cos = np.sin(delta * cos_alphas_shifted)
        # cos_delta_sin = np.cos(delta * sin_alphas_shifted)
        # cos_delta_cos = np.cos(delta * cos_alphas_shifted)
        dsin_delta_sin_ddelta = cos_delta_sin * sin_alphas_shifted
        dsin_delta_cos_ddelta = cos_delta_cos * cos_alphas_shifted
        dcos_delta_sin_ddelta = -sin_delta_sin * sin_alphas_shifted
        dcos_delta_cos_ddelta = -sin_delta_cos * cos_alphas_shifted
        dsin_delta_sin_dphase = cos_delta_sin * delta * dsin_alphas_shifted_dphase
        dsin_delta_cos_dphase = cos_delta_cos * delta * dcos_alphas_shifted_dphase
        dcos_delta_sin_dphase = -sin_delta_sin * delta * dsin_alphas_shifted_dphase
        dcos_delta_cos_dphase = -sin_delta_cos * delta * dcos_alphas_shifted_dphase

        # x1 = -cos_theta * cos_delta_sin + sin_theta * sin_delta_sin
        # x2 = sin_phi * cos_delta_cos + cos_phi * sin_delta_cos
        dx1_dtheta = sin_theta * cos_delta_sin + cos_theta * sin_delta_sin
        dx2_dtheta = 0.0
        dx1_dphi = 0.0
        dx2_dphi = cos_phi * cos_delta_cos - sin_phi * sin_delta_cos
        dx1_ddelta = -cos_theta * dcos_delta_sin_ddelta + sin_theta * dsin_delta_sin_ddelta
        dx2_ddelta = sin_phi * dcos_delta_cos_ddelta + cos_phi * dsin_delta_cos_ddelta
        dx1_dphase = -cos_theta * dcos_delta_sin_dphase + sin_theta * dsin_delta_sin_dphase
        dx2_dphase = sin_phi * dcos_delta_cos_dphase + cos_phi * dsin_delta_cos_dphase

        # y1 = sin_theta * cos_delta_sin
        # y2 = cos_theta * sin_delta_sin
        dy1_dtheta = cos_theta * cos_delta_sin
        dy2_dtheta = -sin_theta * sin_delta_sin
        dy1_dphi = 0.0
        dy2_dphi = 0.0
        dy1_ddelta = sin_theta * dcos_delta_sin_ddelta
        dy2_ddelta = cos_theta * dsin_delta_sin_ddelta
        dy1_dphase = sin_theta * dcos_delta_sin_dphase
        dy2_dphase = cos_theta * dsin_delta_sin_dphase

        # x = x1 * x2
        # y = y1 + y2
        dx_dtheta = dx1_dtheta * x2 + x1 * dx2_dtheta
        dx_dphi = dx1_dphi * x2 + x1 * dx2_dphi
        dx_ddelta = dx1_ddelta * x2 + x1 * dx2_ddelta
        dx_dphase = dx1_dphase * x2 + x1 * dx2_dphase
        dy_dtheta = dy1_dtheta + dy2_dtheta
        dy_dphi = dy1_dphi + dy2_dphi
        dy_ddelta = dy1_ddelta + dy2_ddelta
        dy_dphase = dy1_dphase + dy2_dphase

        # ix = w_2 + eyeball_radius * x
        # iy = h_2 + eyeball_radius * y
        dix_dtheta = eyeball_radius * dx_dtheta
        dix_dphi = eyeball_radius * dx_dphi
        dix_ddelta = eyeball_radius * dx_ddelta
        dix_dphase = eyeball_radius * dx_dphase
        diy_dtheta = eyeball_radius * dy_dtheta
        diy_dphi = eyeball_radius * dy_dphi
        diy_ddelta = eyeball_radius * dy_ddelta
        diy_dphase = eyeball_radius * dy_dphase

        # dx = ix - iris_landmarks[:, 0]
        # dy = iy - iris_landmarks[:, 1]
        ddx_dtheta = dix_dtheta
        ddx_dphi = dix_dphi
        ddx_ddelta = dix_ddelta
        ddx_dphase = dix_dphase
        ddy_dtheta = diy_dtheta
        ddy_dphi = diy_dphi
        ddy_ddelta = diy_ddelta
        ddy_dphase = diy_dphase

        # out = dx ** 2 + dy ** 2
        dout_dtheta = np.mean(2 * (dx * ddx_dtheta + dy * ddy_dtheta))
        dout_dphi = np.mean(2 * (dx * ddx_dphi + dy * ddy_dphi))
        dout_ddelta = np.mean(2 * (dx * ddx_ddelta + dy * ddy_ddelta))
        dout_dphase = np.mean(2 * (dx * ddx_dphase + dy * ddy_dphase))

        # iris_dx = e_x0 + eyeball_radius * -cos_theta * sin_phi - i_x0
        # iris_dy = e_y0 + eyeball_radius * sin_theta - i_y0
        # out += iris_dx ** 2 + iris_dy ** 2
        dout_dtheta += 2 * eyeball_radius * (sin_theta * sin_phi * iris_dx + cos_theta * iris_dy)
        dout_dphi += 2 * eyeball_radius * (-cos_theta * cos_phi * iris_dx)

        return out, np.array([dout_dtheta.item(), dout_dphi.item(), dout_ddelta, dout_dphase])

    phase = 0.02
    result = scipy.optimize.minimize(gaze_fit_loss_func, x0=np.array([theta, phi, delta, phase]),
                                     bounds=(
                                         (-0.4*np.pi, 0.4*np.pi),
                                         (-0.4*np.pi, 0.4*np.pi),
                                         (0.01*np.pi, 0.5*np.pi),
                                         (-np.pi, np.pi),
                                     ),
                                     jac=True,
                                     tol=1e-6,
                                     method='TNC',
                                     options={
                                         # 'disp': True,
                                         'gtol': 1e-6,
                                         'maxiter': 100,
                                     })
    if result.success:
        theta, phi, delta, phase = result.x

    return np.array([theta, phi])
