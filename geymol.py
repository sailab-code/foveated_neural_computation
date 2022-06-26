import numpy as np
import cv2
from math import isnan, sqrt
from random import randint, uniform
from scipy.integrate import odeint
import time
import os
import torch
import torch.nn.functional as F


class GEymol:

    def __init__(self, parameters, device=None):
        self.parameters = parameters
        self.h = parameters['h']
        self.w = parameters['w']
        self.t = 0
        if 'y' in parameters and parameters['y'] is not None:
            self.y = np.array(parameters['y'])  # y = [x (row), y (col), velocity_x, velocity_y]
        else:
            self.y = GEymol.__generate_initial_conditions(self.h, self.w)
        self.is_online = parameters['is_online'] if 'is_online' in parameters else False
        self.saccades_per_second = float(3.0)  # it must be float
        self.real_time_last_saccade = time.process_time()
        self.first_call = True
        self.static_image = "static_image" in parameters and parameters["static_image"]
        self.gradient_norm_t_static = None
        self.of_norm_t_static = None
        self.virtual_mass_t = None
        self.gradient_norm_t = None
        self.of_norm_t = None

        if parameters['max_distance'] <= 0 or parameters['max_distance'] % 2 == 0:
            raise ValueError("Invalid filter size, it must be odd! (max_distance)")

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # sobel filter processor
        self.sobel_kernel = GEymol.__get_sobel_kernel(self.device)

        # precomputed IOR mask
        if 'ior_ray' not in parameters.keys():
            parameters['ior_ray'] = 0.02 * min(self.h, self.w)
        if 'ior_blur' not in parameters.keys():
            parameters['ior_blur'] = 0.15 * min(self.h, self.w)
        self.centered_gaussian_2d_approx = \
            self.__generate_approximation_of_2d_gaussian([self.h // 2, self.w // 2],
                                                         ray=round(parameters['ior_ray']),
                                                         blur=round(parameters['ior_blur'])
                                                         # blur=self.parameters['max_distance']
                                                         # blur=7
                                                         )

        # generating the distance matrix
        self.gravitational_filter = \
            torch.from_numpy(GEymol.__create_gravitational_filter(parameters['max_distance'])).to(self.device)

        # matrix to mark pixels to which inhibit returns
        self.IOR_matrix = torch.zeros((self.h, self.w), dtype=torch.float32, device=self.device)

        # face detector and related map
        base_path = os.path.dirname(cv2.__file__) + os.sep + "data"
        self.face_detector = cv2.CascadeClassifier(base_path + os.sep + 'haarcascade_frontalface_default.xml')
        self.face_map_t = torch.zeros((self.h, self.w), dtype=torch.float32, device=self.device)
        self.feature_maps = None

    def reset(self, y=[], t=0):
        self.t = t

        if y is None:
            self.y = GEymol.__generate_initial_conditions(self.h, self.w)
        else:
            self.y = np.array(y)

        self.first_call = True

    def reset_inhibition_of_return(self):
        self.IOR_matrix *= 0.

    def next_location(self, frame_t, of_t=None, lock=None, frame_gray_uint8_cpu=None,
                      virtualmass_xy=None, virtualmass_vxvy=None, virtualmass=None):

        # ensuring data is well shaped (expected 4D tensors, 1 x c x h x w in [0,1] -frame_t- and 1 x 2 x h x w -of_t)
        if frame_t.ndim != 4 or frame_t.shape[0] != 1 or frame_t.shape[1] != 1:
            raise ValueError("Unsupported tensor format for the input frame: " + str(frame_t.shape) +
                             " (expected 1 x 1 x h x w)")
        if of_t is not None and (of_t.ndim != 4 or of_t.shape[0] != 1 or of_t.shape[1] != 2):
            raise ValueError("Unsupported tensor format for the optical flow data: " + str(of_t.shape) +
                             " (expected 1 x 2 x h x w)")
        if virtualmass_xy is not None and virtualmass is not None:
            raise ValueError("You can specify either a virtualmass_xy or a virtualmass, not both of them!")

        # let's start from the initial position
        if self.first_call:
            if self.static_image:

                # if processing a static image - to speed up computations
                self.gradient_norm_t_static = GEymol.__get_gradient_norm(frame_t, self.sobel_kernel)
                self.of_norm_t_static = GEymol.__get_opticalflow_norm(of_t) if of_t is not None else None

        # computing features
        if not self.static_image:
            self.gradient_norm_t = GEymol.__get_gradient_norm(frame_t, self.sobel_kernel)
            self.of_norm_t = GEymol.__get_opticalflow_norm(of_t)
        else:
            self.gradient_norm_t = self.gradient_norm_t_static
            self.of_norm_t = self.of_norm_t_static

        if self.parameters['alpha_fm'] > 0.0:
            self.__update_face_map(frame_t, frame_gray_uint8_cpu=frame_gray_uint8_cpu)

        virtualmass_t = GEymol.__build_map_from_xy(frame_t, virtualmass_xy) if virtualmass_xy else None
        self.virtual_mass_t = virtualmass if virtualmass_t is None and virtualmass is not None else virtualmass_t

        if virtualmass_xy is not None and virtualmass_vxvy is not None:
            self.y[2] = virtualmass_vxvy[0]
            self.y[3] = virtualmass_vxvy[1]

        if self.first_call:
            self.first_call = False
            return self.y, False

        gradient_norm_t = self.gradient_norm_t * (1.0 - self.IOR_matrix)
        of_norm_t = self.of_norm_t
        face_map_t = self.face_map_t * (1.0 - self.IOR_matrix) if self.parameters['alpha_fm'] > 0.0 else None
        virtualmass_t = self.virtual_mass_t * (1.0 - self.IOR_matrix) if self.virtual_mass_t is not None else None

        # stacking features
        feature_maps = (gradient_norm_t, of_norm_t, face_map_t, virtualmass_t)
        self.feature_maps = feature_maps

        # integrating ODE
        y_prev = self.y
        if lock is not None:
            with lock:
                y = odeint(GEymol.__my_ode, self.y, np.arange(self.t, self.t + 1, .1),  # instants to integrate (10)
                           args=(feature_maps, self.parameters, self.gravitational_filter),
                           mxstep=00, rtol=0.1, atol=0.1
                           )
                self.y = y[-1, :]  # picking up the latest integrated time instant
        else:
            y = odeint(GEymol.__my_ode, self.y, np.arange(self.t, self.t + 1, .1),  # instants to integrate (10)
                       args=(feature_maps, self.parameters, self.gravitational_filter),
                       mxstep=100, rtol=0.1, atol=0.1
                       )
            self.y = y[-1, :]  # picking up the latest integrated time instant

        # next time instant
        self.t += 1

        # avoid predicting out-of-frame locations
        foa_xy_and_velxy = self.y
        foa_xy_and_velxy[0], foa_xy_and_velxy[1] = \
            GEymol.__stay_inside_fix_nans_round_to_int((self.h, self.w), foa_xy_and_velxy[0:2])

        vel_norm = sqrt((float(foa_xy_and_velxy[0]) - y_prev[0]) ** 2 + (float(foa_xy_and_velxy[1]) - y_prev[1]) ** 2)
        saccade = vel_norm > self.parameters["fixation_threshold_speed"]

        # add pixel coordinates to the inhibition of return matrix
        if not self.is_online:
            if self.t % max(int(float(self.parameters['fps']) / self.saccades_per_second), 1) == 0:
                self.IOR_matrix = self.__inhibit_return_in(self.IOR_matrix, row_col=foa_xy_and_velxy[0:2])
        else:
            if time.process_time() - self.real_time_last_saccade >= (1.0 / self.saccades_per_second):
                self.IOR_matrix = self.__inhibit_return_in(self.IOR_matrix, row_col=foa_xy_and_velxy[0:2])
                self.real_time_last_saccade = time.process_time()  # update real time of the last saccade

        return foa_xy_and_velxy, saccade

    @staticmethod
    def __torch_float_01_to_np_uint8(torch_img):
        if torch_img.ndim == 2:
            return (torch_img * 255.0).cpu().numpy().astype(np.uint8)
        elif torch_img.ndim == 3:
            return (torch_img * 255.0).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        elif torch_img.ndim == 4:
            return (torch_img * 255.0).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        else:
            raise ValueError("Unsupported image type.")

    def __update_face_map(self, frame_t, updating_factor=.3, frame_gray_uint8_cpu=None):

        # add potential in locations of faces
        if frame_gray_uint8_cpu is not None:
            faces = self.face_detector.detectMultiScale(frame_gray_uint8_cpu)
        else:
            if frame_t.dtype == torch.float32:
                faces = self.face_detector.detectMultiScale(
                    GEymol.__torch_float_01_to_np_uint8(frame_t), 1.3, 5)
            else:
                raise ValueError("Unsupported tensor type (expected torch.float32, values in [0,1]).")

        face_map_new = torch.zeros_like(self.face_map_t)
        for (y, x, h, w) in faces:
            face_map_new[x:x + w, y:y + h] = 1.0  # tested

        # update as weighted sum
        self.face_map_t = (1.0 - updating_factor) * self.face_map_t + updating_factor * face_map_new

    def __generate_approximation_of_2d_gaussian(self, row_col, ray=25, blur=151):
        if blur % 2 == 0:
            blur += 1

        row, col = row_col
        blank_image_with_circle = np.zeros((self.h, self.w), dtype=np.float32)
        cv2.circle(blank_image_with_circle, (col, row), ray, 1.0, -1)  # draw a filled circle (setting it to 1.0)
        gaussian_2d_approx = cv2.GaussianBlur(blank_image_with_circle, (blur, blur), 0)  # blur the whole image
        max_val = np.max(gaussian_2d_approx)

        if max_val < 1.0:
            gaussian_2d_approx = gaussian_2d_approx / max_val  # normalize in [0,1]
        return torch.from_numpy(gaussian_2d_approx).to(self.device)

    def __inhibit_return_in(self, frame, row_col):
        cx = (self.h // 2)
        cy = (self.w // 2)
        ox = cx - row_col[0]
        oy = cy - row_col[1]
        gaussian_2d_approx = \
            GEymol.__extract_patch(self.centered_gaussian_2d_approx, [cx + ox, cy + oy], [self.h, self.w])

        frame = 0.9 * frame + gaussian_2d_approx
        frame = torch.min(frame, torch.tensor(1.0, dtype=torch.float32, device=self.device))
        return frame

    @staticmethod
    def __generate_initial_conditions(h, w):
        init_ray = int(min(h, w) * 0.17)  # arbitrary (it should be improved)
        x1_init = int(h / 2) + randint(-init_ray, init_ray)  # arbitrary (it should be improved)
        x2_init = int(w / 2) + randint(-init_ray, init_ray)  # arbitrary (it should be improved)
        v1_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))  # arbitrary (it should be improved)
        v2_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))  # arbitrary (it should be improved)
        return np.array([x1_init, x2_init, v1_init, v2_init])

    @staticmethod
    def __create_gravitational_filter(filter_size):
        filter_matrix = np.zeros((2, filter_size, filter_size), dtype=np.float32)  # size: 2 x filter_size x filter_size
        center_x, center_y = (filter_size // 2), (filter_size // 2)

        for i in range(filter_size):
            for j in range(filter_size):
                if not (i == center_x and j == center_y):  # avoid mid of the filter (set it to zero)
                    filter_matrix[0, i, j] = (filter_size // 10.0 + 1.0) * float(i - center_x) / (
                            ((i - center_x) ** 2 + (j - center_y) ** 2) + (filter_size // 10.0))

        for i in range(filter_size):
            for j in range(filter_size):
                if not (i == center_x and j == center_y):  # avoid mid of the filter (set it to zero)
                    filter_matrix[1, i, j] = (filter_size // 10.0 + 1.0) * float(j - center_y) / (
                            ((i - center_x) ** 2 + (j - center_y) ** 2) + (filter_size // 10.0))

        return filter_matrix

    @staticmethod
    def __get_sobel_kernel(device):
        kernel = torch.tensor([[[[-1., -4., -6., -4., -1.],
                                 [-2., -8., -12., -8., -2.],
                                 [0., 0., 0., 0., 0.],
                                 [2., 8., 12., 8., 2.],
                                 [1., 4., 6., 4., 1.]]],
                               [[[-1., -2., 0., 2., 1.],
                                 [-4., -8., 0., 8., 4.],
                                 [-6., -12., 0., 12., 6.],
                                 [-4., -8., 0., 8., 4.],
                                 [-1., -2., 0., 2., 1.]]]]
                              , dtype=torch.float32, device=device)  # 2 x 1 x 5 x 5
        return kernel / (0.25 * torch.sum(torch.abs(kernel)))

    @staticmethod
    def __get_gradient_norm(frame_t, sobel_kernel):
        frame_t = F.pad(frame_t, pad=(2, 2, 2, 2), mode='replicate')
        # frame_t = F.pad(frame_t, pad=(2, 2, 2, 2), mode="constant", value=0.42)
        # sobel_xy = F.conv2d(frame_t, sobel_kernel, bias=None, padding=(2, 2), groups=1)
        # sobel_xy = F.conv2d(frame_t, sobel_kernel, bias=None, padding='same', groups=1)
        sobel_xy = F.conv2d(frame_t, sobel_kernel, bias=None)

        # getting norm
        grad_norm = torch.squeeze(torch.sum(sobel_xy ** 2, dim=1)) # h x w (max = 1)

        return grad_norm

    @staticmethod
    def __get_opticalflow_norm(of_t):

        # getting norm (optical flow is expected to be 1 x 2 x h x w)
        of_norm = torch.squeeze(torch.sqrt(torch.sum(of_t ** 2, dim=1)))  # h x w

        # get outliers (it solves ego-motion)
        of_norm = of_norm - torch.mean(of_norm)
        of_norm = torch.abs(of_norm)

        return of_norm

    @staticmethod
    def __build_map_from_xy(frame_t, xy):
        z = torch.zeros((frame_t.shape[2], frame_t.shape[3]), dtype=torch.float, device=frame_t.device)
        z[int(xy[0]), int(xy[1])] = 1.0
        return z

    @staticmethod
    def __stay_inside_fix_nans_round_to_int(frame_hw, row_col, ray=5):
        row, col = row_col

        if isnan(row) or isnan(col):
            row, col = 0, 0
        else:
            row, col = int(row), int(col)

        if row - ray < 0:
            row = ray
        else:
            if row + ray >= frame_hw[0]:
                row = frame_hw[0] - ray - 1
        if col - ray < 0:
            col = ray
        else:
            if col + ray >= frame_hw[1]:
                col = frame_hw[1] - ray - 1

        return row, col

    @staticmethod
    def __extract_patch(frame, patch_center_xy, patch_size_xy, normalize=False):
        x, y = patch_center_xy
        odd_x = patch_size_xy[0] % 2
        odd_y = patch_size_xy[1] % 2
        d_x = patch_size_xy[0] // 2
        d_y = patch_size_xy[1] // 2
        h, w = frame.shape[0], frame.shape[1]

        # avoid extracting patches that are centered at out-of-the-frame-coordinates
        if x < 0:
            x = 0
        elif x >= h:
            x = h - 1

        if y < 0:
            y = 0
        elif y >= w:
            y = w - 1

        # integer coordinates
        x = int(x)
        y = int(y)

        # handling borders
        f_x = x - d_x
        t_x = x + d_x
        f_y = y - d_y
        t_y = y + d_y
        if t_x >= h or f_x < 0 or t_y >= w or f_y < 0:
            patch = torch.zeros((patch_size_xy[0], patch_size_xy[1]), dtype=frame.dtype, device=frame.device)

            sf_x = 0
            st_x = 0
            sf_y = 0
            st_y = 0
            cf_x = f_x
            cf_y = f_y
            ct_x = t_x
            ct_y = t_y

            if f_x < 0:
                cf_x = 0
                sf_x = -f_x
            if t_x >= h:
                ct_x = h - 1
                st_x = t_x - ct_x
            if f_y < 0:
                cf_y = 0
                sf_y = -f_y
            if t_y >= w:
                ct_y = w - 1
                st_y = t_y - ct_y

            patch[sf_x:patch_size_xy[0] - st_x, sf_y:patch_size_xy[1] - st_y] = \
                frame[cf_x:ct_x + odd_x, cf_y:ct_y + odd_y]
        else:
            patch = frame[f_x:t_x + 1, f_y:t_y + 1]

        # normalizing
        if normalize:
            max_val = torch.max(patch)
            if max_val > 0.0:
                return patch / max_val
            else:
                return patch
        else:
            return patch

    @staticmethod
    def __my_ode(y, t, feature_maps, parameters, gravitational_filter):
        dissipation = parameters['dissipation']
        alpha_c = parameters['alpha_c']
        alpha_of = parameters['alpha_of']
        alpha_fm = parameters['alpha_fm']
        alpha_virtual = parameters['alpha_virtual']
        filter_size = parameters['max_distance']
        filter_sizes = [filter_size, filter_size]

        # extracting patches from the considered features
        gradient_norm_t_patch = GEymol.__extract_patch(feature_maps[0], y[0:2], filter_sizes, normalize=True)

        # computing gravitational fields contributions
        # (broadcasting the product between 2 x filter_size x filter_size and 1 x filter_size x filter_size)
        gravitational_grad = alpha_c * torch.sum(gravitational_filter * gradient_norm_t_patch.unsqueeze(0), dim=(1, 2))

        if feature_maps[1] is not None:
            of_norm_t_patch = GEymol.__extract_patch(feature_maps[1], y[0:2], filter_sizes, normalize=True)
            gravitational_of = alpha_of * torch.sum(gravitational_filter * of_norm_t_patch.unsqueeze(0), dim=(1, 2))
        else:
            gravitational_of = torch.zeros(1)

        if feature_maps[2] is not None:
            face_map_t_patch = GEymol.__extract_patch(feature_maps[2], y[0:2], filter_sizes, normalize=False)
            gravitational_faces = alpha_fm * torch.sum(gravitational_filter * face_map_t_patch.unsqueeze(0), dim=(1, 2))
        else:
            gravitational_faces = torch.zeros(1)

        if feature_maps[3] is not None:
            virtualmass_t_patch = GEymol.__extract_patch(feature_maps[3], y[0:2], filter_sizes, normalize=False)
            gravitational_virtualmass = alpha_virtual * torch.sum(gravitational_filter *
                                                                  virtualmass_t_patch.unsqueeze(0), dim=(1, 2))
        else:
            gravitational_virtualmass = torch.zeros(1)

        # building the system of differential equations (4 equations)
        # y[2]
        # y[3]
        # gravitational_grad[0] + gravitational_of[0] + gravitational_faces[0] + gravitational_virtualmass[0]
        #     - dissipation * y[2]
        # gravitational_grad[1] + gravitational_of[1] + gravitational_faces[1] + gravitational_virtualmass[0]
        #     - dissipation * y[3]
        dy = np.concatenate([np.array(y[2:]),
                             gravitational_grad.cpu().numpy() +
                             gravitational_of.cpu().numpy() +
                             gravitational_faces.cpu().numpy() +
                             gravitational_virtualmass.cpu().numpy() -
                             dissipation * np.array(y[2:])])

        return dy
