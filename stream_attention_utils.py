import numpy as np
import matplotlib.pyplot as plt
from typing import List
import csv
from sklearn.metrics import confusion_matrix
from skimage import color
import cv2
import torchvision
import os
import json
import math
import seaborn

colors = [[255, 0, 0], [255, 128, 0], [255, 255, 9], [128, 255, 0], [0, 255, 0], [0, 255, 128], [0, 255, 255],
          [0, 128, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127], [128, 128, 128], [255, 204, 204],
          [255, 229, 204], [255, 255, 204], [229, 255, 204], [204, 255, 204], [204, 255, 229], [204, 229, 255],
          [0, 0, 0]]
colors = list(np.array(colors) / 255.0)


def plot(data1_per_stimulus: list, x1_label: str, y1_label: str,
         data2_per_stimulus: list = None, x2_label: str = None, y2_label: str = None,
         data3_per_stimulus: list = None, x3_label: str = None, y3_label: str = None,
         normalize=False, ignore_last_class=False, stimuli_ids=None, path=None):
    num_plots = 1 + (data2_per_stimulus is not None) + (data3_per_stimulus is not None)
    data_per_stimulus = [data1_per_stimulus, data2_per_stimulus, data3_per_stimulus]
    x_labels = [x1_label, x2_label, x3_label]
    y_labels = [y1_label, y2_label, y3_label]

    if ignore_last_class:
        for j in range(0, num_plots):
            for s in range(len(data_per_stimulus[j])):
                data_per_stimulus[j][s] = data_per_stimulus[j][s][0: (data_per_stimulus[j][s].shape[0] - 1)]

    if normalize:
        for j in range(0, num_plots):
            for s in range(len(data_per_stimulus[j])):
                den = np.sum(data_per_stimulus[j][s])
                if den == 0:
                    den = 1
                data_per_stimulus[j][s] = data_per_stimulus[j][s] / den

    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 7))

    if num_plots == 1:
        axs = [axs]

    for j in range(0, num_plots):
        data = np.array(data_per_stimulus[j]).transpose()  # number-of-classes x number-of-stimuli
        X = np.arange(data.shape[1])  # number-of-stimuli

        wi = 1. / (3. + data.shape[0])
        for i in range(0, data.shape[0]):
            axs[j].bar(X + i * wi + wi / 2., data[i], color=colors[i], width=wi)
            axs[j].bar(X + i * wi + wi / 2., -0.025, color=colors[i], width=wi)

        axs[j].set_xlabel(x_labels[j])
        axs[j].set_ylabel(y_labels[j])
        axs[j].set_xticks(X)

        if stimuli_ids is not None:
            axs[j].set_xticklabels(stimuli_ids)

        axs[j].set_xlim((0, data.shape[1]))
        axs[j].set_ylim((-0.025, max(np.max(data), 0.1)))

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    return plt


def accuracies_per_class_to_string(acc_per_class):
    s = ""
    for i in range(0, acc_per_class.shape[0]):
        if i > 0:
            s += " "
        s += "{:.2f}".format(acc_per_class[i]) if acc_per_class[i] != -1.0 else "-.--"
    return s


def compute_accuracies_per_class(conf_matrix: np.ndarray):
    den = np.sum(conf_matrix, axis=1)
    num = conf_matrix.diagonal()
    dz = den == 0
    den[dz] = 1.
    acc_per_class = num / den
    acc_per_class[dz] = -1.0  # classes with no examples at all
    return acc_per_class


def compute_accuracies(conf_matrix_total: np.ndarray,
                       conf_matrices_per_stimulus: List[np.ndarray]):
    den = np.sum(conf_matrix_total)
    if den == 0:
        den = 1
    acc_total_unbalanced = np.sum(conf_matrix_total.diagonal()) / den
    acc_total_per_class = compute_accuracies_per_class(conf_matrix_total)
    acc_total_balanced = np.mean(acc_total_per_class)

    acc_per_stimulus_unbalanced = [None] * len(conf_matrices_per_stimulus)
    acc_per_stimulus_per_class = [None] * len(conf_matrices_per_stimulus)
    acc_per_stimulus_balanced = [None] * len(conf_matrices_per_stimulus)

    for i in range(0, len(conf_matrices_per_stimulus)):
        den = np.sum(conf_matrices_per_stimulus[i])
        if den == 0:
            den = 1
        acc_per_stimulus_unbalanced[i] = np.sum(conf_matrices_per_stimulus[i].diagonal()) / den
        acc_per_stimulus_per_class[i] = compute_accuracies_per_class(conf_matrices_per_stimulus[i])
        acc_per_stimulus_balanced[i] = np.mean(acc_per_stimulus_per_class[i])

    return acc_total_unbalanced, acc_total_balanced, acc_total_per_class, \
           acc_per_stimulus_unbalanced, acc_per_stimulus_balanced, acc_per_stimulus_per_class


def save_result_log(log_file: str, logs: List[dict], append=False):
    with open(log_file, 'w' if not append else 'a') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=";", quoting=csv.QUOTE_NONE, fieldnames=list(logs[0].keys()))
        if not append:
            writer.writeheader()
        writer.writerows(logs)


def parse_conf_matrix(text: str, num_classes: int) -> np.ndarray:
    return np.fromstring(text.replace('[', '').replace(']', ''),
                         dtype=np.long, sep=', ').reshape(num_classes, num_classes)


def read_result_log(log_file: str, test_only=False):
    conf_matrix_total = None
    conf_matrices_per_stimulus = []

    conf_matrix_focussed_total = None
    conf_matrices_focussed_per_stimulus = []

    focussed_classes_total = None
    focussed_classes_per_stimulus = []

    focussed_moving_classes_total = None
    focussed_moving_classes_per_stimulus = []

    average_time = 0.
    tot_frames = 0
    processed_stimuli = []

    foa_xs_per_stimulus = []
    foa_ys_per_stimulus = []

    stimulus_id_per_stimulus = []

    # reading file
    with open(log_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";", quoting=csv.QUOTE_NONE)

        # reading a single row
        for row in reader:

            # reading basic data from the current row (and updating some statistics)
            is_test = row['is_test'] == 'True'
            if not is_test and test_only:
                continue

            if row['processed_stimulus_id'] not in processed_stimuli:
                processed_stimuli.append(row['processed_stimulus_id'])
            s = processed_stimuli.index(row['processed_stimulus_id'])

            num_classes = int(row['classes'])
            focussed_class = int(row['focussed_class'])
            focussed_moving_obj = int(row['focussed_moving_obj'])
            average_time += float(row['time'])
            foa_x = int(row['foa_x'])
            foa_y = int(row['foa_y'])
            stimulus_id = int(row['stimulus_id'])
            tot_frames += 1

            # accumulating data from the confusion matrix of this row
            conf_matrix = parse_conf_matrix(row['confusion_matrix'], num_classes)
            if conf_matrix_total is None:
                conf_matrix_total = np.array(conf_matrix, copy=True)
            else:
                conf_matrix_total += conf_matrix
            if s >= len(conf_matrices_per_stimulus):
                conf_matrices_per_stimulus.append(np.array(conf_matrix, copy=True))
            else:
                conf_matrices_per_stimulus[s] += conf_matrix

            # accumulating data from the confusion matrix of this row (focussed areas)
            conf_matrix_focussed = parse_conf_matrix(row['confusion_matrix_focussed'], num_classes)
            if conf_matrix_focussed_total is None:
                conf_matrix_focussed_total = np.array(conf_matrix_focussed, copy=True)
            else:
                conf_matrix_focussed_total += conf_matrix_focussed
            if s >= len(conf_matrices_focussed_per_stimulus):
                conf_matrices_focussed_per_stimulus.append(np.array(conf_matrix_focussed, copy=True))
            else:
                conf_matrices_focussed_per_stimulus[s] += conf_matrix_focussed

            # accumulating data from statistics on focussed classes (or moving only focussed classes)
            if s >= len(focussed_classes_per_stimulus):
                focussed_classes_per_stimulus.append(np.zeros(num_classes, dtype=np.long))
                focussed_moving_classes_per_stimulus.append(np.zeros(num_classes, dtype=np.long))
            if focussed_class >= 0:  # when it is < 0, it is an unknown class
                if focussed_classes_total is None:
                    focussed_classes_total = np.zeros(conf_matrix.shape[0], dtype=np.long)
                if focussed_moving_classes_total is None:
                    focussed_moving_classes_total = np.zeros(conf_matrix.shape[0], dtype=np.long)

                focussed_classes_total[focussed_class] += 1
                if focussed_moving_obj == 1:
                    focussed_moving_classes_total[focussed_class] += 1
                focussed_classes_per_stimulus[s][focussed_class] += 1
                if focussed_moving_obj == 1:
                    focussed_moving_classes_per_stimulus[s][focussed_class] += 1

            # FOA coordinates per stimulus
            if s >= len(foa_xs_per_stimulus):
                foa_xs_per_stimulus.append([])
            foa_xs_per_stimulus[s].append(foa_x)

            if s >= len(foa_ys_per_stimulus):
                foa_ys_per_stimulus.append([])
            foa_ys_per_stimulus[s].append(foa_y)

            # ID of the stimuli
            if s >= len(stimulus_id_per_stimulus):
                stimulus_id_per_stimulus.append(stimulus_id)

    # normalizing time
    average_time /= tot_frames

    # post-computing the confusion matrix on the peripheral areas
    conf_matrix_peri_total = conf_matrix_total - conf_matrix_focussed_total
    conf_matrices_peri_per_stimulus = []
    for i in range(0, len(conf_matrices_per_stimulus)):
        conf_matrices_peri_per_stimulus.append(conf_matrices_per_stimulus[i] -
                                               conf_matrices_focussed_per_stimulus[i])

    return conf_matrix_total, conf_matrices_per_stimulus, \
           conf_matrix_focussed_total, conf_matrices_focussed_per_stimulus, \
           conf_matrix_peri_total, conf_matrices_peri_per_stimulus, \
           focussed_classes_total, focussed_classes_per_stimulus, \
           focussed_moving_classes_total, focussed_moving_classes_per_stimulus, \
           average_time, tot_frames, foa_xs_per_stimulus, foa_ys_per_stimulus, stimulus_id_per_stimulus


def compute_confusion_matrices(predictions: np.ndarray, sup_labels: np.ndarray,
                               h: int, w: int, foa_x: int, foa_y: int,
                               conf_mat_labels: List[int],
                               bg_thres: float = 0.5):
    num_classes = predictions.shape[0]
    bg_decisions = np.amax(predictions, axis=0)
    bg_decisions = bg_decisions <= bg_thres
    decision = np.argmax(predictions, axis=0).astype(np.uint8) * np.logical_not(bg_decisions) \
               + num_classes * bg_decisions

    # ensuring shapes are fine
    decision = decision.reshape(h, w)
    sup_labels = sup_labels.reshape(h, w)

    # metrics (per frame)
    m = int(14 / 2.0)  # focussed area is defined to be (2m + 1) x (2m + 1)
    top = foa_x - m
    top = top if top >= 0 else 0
    bottom = foa_x + m
    bottom = bottom if bottom < h else h - 1
    left = foa_y - m
    left = left if left >= 0 else 0
    right = foa_y + m
    right = right if right < w else w - 1

    conf_matrix = confusion_matrix(sup_labels.flatten(), decision.flatten(), labels=conf_mat_labels)
    conf_matrix_focussed = confusion_matrix(sup_labels[top:(bottom + 1), left:(right + 1)].flatten(),
                                            decision[top:(bottom + 1), left:(right + 1)].flatten(),
                                            labels=conf_mat_labels)

    return decision, conf_matrix, conf_matrix_focussed


# this places an external crop into an image, given the coordinates of its barycenter
def place_crop(_img, _crop, _mid):
    _h = _img.shape[0]
    _w = _img.shape[1]
    __crop = np.tile(np.expand_dims(_crop, 2), (1, 1, 3))

    top = (_mid[0] - (_crop.shape[0] // 2))
    c_top = top
    bottom = top + _crop.shape[0]
    c_bottom = bottom
    d_top = 0
    d_bottom = d_top + _crop.shape[0]
    if top < 0:
        c_top = 0
        d_top = -top
    if bottom > _h:
        c_bottom = _h
        d_bottom -= (bottom - _h)

    left = (_mid[1] - (_crop.shape[1] // 2))
    c_left = left
    right = left + _crop.shape[1]
    c_right = right
    d_left = 0
    d_right = d_left + _crop.shape[1]
    if left < 0:
        c_left = 0
        d_left = -left
    if right > _w:
        c_right = _w
        d_right -= (right - _w)

    _img[c_top:c_bottom, c_left:c_right, :] = __crop[d_top:d_bottom, d_left:d_right]


def create_legend(class_id_to_label, siz):
    def draw_text(_h, _w, _text, _scale=1, _tick=2):
        # draw text of fake image
        _img = np.zeros((_h, _w), np.uint8)
        _img = cv2.putText(_img, _text, (_w // 2, _h // 2), cv2.FONT_HERSHEY_SIMPLEX, _scale, 255, _tick, cv2.LINE_AA)

        # finding the bounding box of the drawn text
        _idx = cv2.findNonZero(_img)
        _box = cv2.boundingRect(_idx)

        # cropping
        __crop = _img[_box[1]:_box[1] + _box[3], _box[0]:_box[0] + _box[2]]

        return __crop

    _c = len(class_id_to_label)
    ss = 100
    bg = np.zeros((ss, ss * _c, 3), dtype=np.uint8)
    bg_labels = np.zeros((ss, ss * _c), dtype=np.uint8)
    for _i in range(0, _c):
        bg_labels[:, ss * _i:ss * (_i + 1)] = _i
        _crop = draw_text(ss, ss, class_id_to_label[_i], _scale=2, _tick=2)
        place_crop(bg, _crop, (ss // 2, ((ss * _c) // (2 * _c)) + ((2 * _i * (ss * _c)) // (2 * _c))))

    _legend = cv2.resize(bg, siz, interpolation=cv2.INTER_LINEAR)
    _fake_predictions = cv2.resize(bg_labels, siz, interpolation=cv2.INTER_NEAREST)
    return _legend, _fake_predictions.astype(np.uint8)


def build_basic_drawing_panel(h: int, w: int, legend: np.ndarray):
    np_image = np.zeros((h, w, 3), dtype=np.float)
    data = np.concatenate([np_image, legend], axis=0)
    data = np.concatenate([data, data * 0, data * 0], axis=1)
    data = np.concatenate([data, np.concatenate([np_image * 0, np_image * 0, np_image * 0], axis=1)], axis=0)
    return data


def update_drawing_panel(drawing_panel: np.ndarray,
                         image: np.ndarray, predicted_labels: np.ndarray,
                         legend: np.ndarray, legend_labels: np.ndarray,
                         num_classes: int,
                         foa_x: int, foa_y: int,
                         other_data_to_draw: List[np.ndarray],
                         focussed_region_size: int = None,
                         highlight_frame: bool = False):
    # guessing image sizes
    h = image.shape[0]
    w = image.shape[1]

    # switching label associated to the background class (needed due to a boring issue of the drawing routine)
    predicted_labels[predicted_labels == num_classes] = 255

    # predictions (with highlighting)
    out_pic_pred = color.label2rgb(np.concatenate([predicted_labels, legend_labels], axis=0),
                                   np.concatenate([image, legend], axis=0), bg_label=255, bg_color=None,
                                   image_alpha=1.0, kind='overlay', channel_axis=2)

    # depicting FOA and focussed areas over predictions
    cv2.drawMarker(out_pic_pred, (foa_y, foa_x), color=(1.0, 0, 0) if not highlight_frame else (0.0, 1.0, 0.0),
                   markerType=cv2.MARKER_CROSS, markerSize=5 if not highlight_frame else 8,
                   thickness=1 if not highlight_frame else 2)

    if focussed_region_size is not None and focussed_region_size > 0:
        cv2.circle(out_pic_pred, (foa_y, foa_x),
                   focussed_region_size // 2, color=(1.0, 0, 0) if not highlight_frame else (0.0, 1.0, 0.0),
                   thickness=1)

    # drawing the first sector
    drawing_panel[0:h + legend.shape[0], 0:w, :] = out_pic_pred

    # drawing all the other 5 sectors
    for i in range(0, 5):
        if other_data_to_draw[i] is None:
            continue

        data = np.clip(other_data_to_draw[i], 0, 1)
        if data.ndim == 2:
            data = np.tile(np.expand_dims(data, axis=2), (1, 1, 3))

        if i == 0:
            drawing_panel[0:h, w:2 * w, :] = data
        elif i == 1:
            drawing_panel[0:h, 2 * w:, :] = data
        elif i == 2:
            drawing_panel[h + legend.shape[0]:, 0:w, :] = data
        elif i == 3:
            drawing_panel[h + legend.shape[0]:, w:2 * w, :] = data
        elif i == 4:
            drawing_panel[h + legend.shape[0]:, 2 * w:, :] = data


def load_dict_from_json(json_file, force_int_keys=False, force_int_values=False):
    _loaded = json.load(open(json_file, 'r'))
    _dict = {}
    if force_int_keys:
        for k, v in _loaded.items():
            _dict[int(k)] = v if not force_int_values else int(v)
    else:
        for k, v in _loaded.items():
            _dict[k] = v if not force_int_values else int(v)
    return _dict


def create_dataset(data_folder, h, w):
    # 'letters' and 'mnist', both from E-MNIST collection
    digits = torchvision.datasets.EMNIST('.', 'mnist', train=True, download=True)
    letters = torchvision.datasets.EMNIST('.', 'byclass', train=True, download=True)

    # selected class names and the corresponding class indices (accordingly to the considered datasets)
    inv_class_map_digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    inv_class_map_letters = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 19,
                             'K': 20}

    # class name to index, accordingly to the classifier
    inv_class_map_classifier = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                                'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18,
                                'K': 19}

    # inverted maps
    class_map_digits = {v: k for k, v in inv_class_map_digits.items()}
    class_map_letters = {v: k for k, v in inv_class_map_letters.items()}
    class_map_classifier = {v: k for k, v in inv_class_map_classifier.items()}

    # class to dataset
    class_name_to_dataset = {}
    for k, _ in inv_class_map_digits.items():
        class_name_to_dataset[k] = digits
    for k, _ in inv_class_map_letters.items():
        class_name_to_dataset[k] = letters

    # indexing data samples by class index (dataset class index)
    index_by_class = [None] * len(class_map_classifier)
    i = 0
    for s, l in digits:
        if l in class_map_digits.keys():
            if i in [2, 6, 10, 11, 12, 15, 17, 20, 22, 26, 27, 30, 34, 35, 37, 38, 39, 40,
                     41, 43, 45, 47, 48, 50, 52, 54, 57, 58, 59, 60, 61, 63, 64, 68, 71, 75,
                     76, 77, 78, 79, 82, 83, 85, 87, 88, 90, 91, 96, 98, 99, 101, 103, 106,
                     107, 108, 109, 117, 121, 123, 125, 127, 130, 131, 135, 138, 140, 143,
                     144, 145, 146, 147, 148, 155, 156, 160, 162, 164, 166, 167, 168, 171,
                     173, 175, 178, 183, 185, 187, 189, 192, 194, 195, 197, 198, 199, 201,
                     203, 204, 205, 206, 208, 210, 212, 217, 218, 219, 221, 226, 227, 229,
                     231, 232, 240, 242, 244, 245, 246, 247, 248, 249, 251, 252, 256, 258,
                     259, 260, 261, 262, 263, 268, 269, 270, 271, 276, 280, 281, 282, 283,
                     285, 287, 288, 289, 290, 294, 295, 296, 300, 303, 304, 309, 311, 317,
                     320, 321, 323, 326, 333, 334, 347, 348, 357, 360, 361, 368, 369, 373,
                     377, 385, 386, 387, 388, 389, 391, 403, 404, 405, 412, 415, 417, 420,
                     421, 422, 423, 424, 433, 435, 436, 438, 443, 445, 448, 452, 453, 461,
                     465, 467, 468, 474, 479, 483, 485, 491, 512, 514, 515, 520, 523, 524,
                     525, 526, 535, 536, 538, 541, 543, 547, 548, 550, 553, 555, 557, 558,
                     559, 561, 562, 565, 567, 571, 572, 573, 574, 576, 580, 586, 591, 592,
                     593, 600]:
                i += 1
                continue
            class_name = class_map_digits[l]
            class_index = inv_class_map_classifier[class_name]
            if index_by_class[class_index] is not None:
                index_by_class[class_index].append(i)
            else:
                index_by_class[class_index] = [i]
        i += 1
    i = 0
    for s, l in letters:
        if l in class_map_letters.keys():
            if i in [27, 32, 40, 77, 101, 109, 131, 139, 147, 177, 184, 244, 296, 300, 305, 309, 347, 358, 405,
                     428, 433, 459, 483, 523, 563, 581, 600, 619, 655, 673, 674, 677, 688, 689, 696, 761,
                     772, 832, 929, 960, 967, 980, 985, 1036, 1050, 1058, 1109, 1117, 1125, 1175, 1243,
                     1302, 1330, 1437, 1464, 1472, 1506, 1548, 1656, 1740, 1765, 1815, 1819, 1820,
                     1835, 1849, 1875, 1943, 1997, 2026, 2033, 2045, 2120, 2143, 2149, 2199, 2203,
                     2230, 2233, 2249, 2276, 2287, 2315, 2321, 2343, 2371, 2481, 2486, 2497, 2516,
                     2539, 2578, 2580, 2586, 2592, 2599, 2608, 2623, 2671, 2676, 2722, 2781, 2788,
                     2802, 2838, 2899, 2912, 2928, 2978, 2990, 3036, 3051, 3062, 3084, 3094, 3115,
                     3169, 3211, 3228, 3295, 3340, 3509, 3570, 3634, 3691, 3737, 3785, 3828, 3895,
                     3907, 3948, 3957, 3978, 4001, 4007, 4031, 4041, 4067, 4105, 4173, 4177, 4220,
                     4333, 4355, 4358, 4367, 4384, 4406, 4434, 4457, 4495, 4511, 4529, 4550, 4554,
                     4609, 4727, 4762, 4773, 4849, 4877, 4927, 4940, 4964, 4999, 5065, 5105, 5137,
                     5155, 5164, 5227, 5277, 5286, 5329, 5371, 5441, 5480, 5506, 5516, 5529, 5538,
                     5572, 5582, 5606, 5651, 5762, 5849, 5866, 5873, 5966, 6046, 6070, 6108, 6155,
                     6162, 6187, 6242, 6283, 6337, 6349, 6414, 6417, 6435, 6473, 6508, 6587, 6645,
                     6683, 6701, 6715, 6754, 6821, 6876, 6892, 6911, 6986, 7100, 7165, 7174, 7229,
                     7266, 7274, 7320, 7373, 7400, 7433, 7434, 7468, 7478, 7480, 7518, 7633, 7673,
                     7778, 7801, 7809, 7843, 7911, 7950, 8125, 8252, 8263, 8378, 8449, 8536, 8563,
                     8588, 8604, 8628, 8703, 8722, 8784, 8804, 8820, 8823, 8916, 8993, 9069, 9124,
                     9135, 9250, 9279, 9292, 9299, 9351, 9381, 9421, 9459, 9644, 9722, 9809, 9916,
                     9967, 10018, 10091, 10120, 10585]:
                i += 1
                continue
            class_name = class_map_letters[l]
            class_index = inv_class_map_classifier[class_name]
            if index_by_class[class_index] is not None:
                index_by_class[class_index].append(i)
            else:
                index_by_class[class_index] = [i]
        i += 1

    # sampling from the merged collection
    def sample_next(_class_name, _class_name_to_dataset, _inv_class_map_classifier, _index_by_class):
        _i = 0
        _id = -1
        _dataset = _class_name_to_dataset[_class_name]
        _class_index = _inv_class_map_classifier[_class_name]
        while True:
            if _index_by_class[_class_index][_i] < 0:
                _i += 1
            else:
                _id = _index_by_class[_class_index][_i]
                break

        _index_by_class[_class_index][_i] *= -1  # masking

        _s, _ = _dataset[_id]
        return np.asarray(_s.convert("L")).transpose()

    # creating a black image
    image = np.zeros((h, w, 3), np.uint8)

    # creating storage for supervisions
    sup_labels = np.ones((h, w), dtype=np.long) * len(inv_class_map_classifier)

    # creating video
    wanna_plot = False
    if wanna_plot:
        plt.ion()

    ax_data = None
    entering = True
    c = len(inv_class_map_classifier)
    crops = [None] * c
    np.random.seed(0)
    k = 1
    passes = 0
    stimulus_id = -1
    stimulus_id_to_files = {}

    while True:
        oh = ow = 0

        # order
        if entering:
            order = np.random.permutation(c)
            stimulus_id += 1

        # loop that fills the screen with all the classes or that clears the screen
        for z in range(0, c):
            zz = order[z]
            class_name = class_map_classifier[zz]

            if entering:

                # sampling a new crop and caching it
                crop = sample_next(class_name, class_name_to_dataset, inv_class_map_classifier, index_by_class)
                crops[z] = crop

                # from out-of-screen to a certain (vertical) destination
                from_h = -14
                dest_h = 7 * (h // 8) - oh
                dest_w = w // 10 + ow

            elif not entering:

                # picking up the cached crop
                crop = crops[z]

                # clearing the area where the crop was drawn last time
                cur_h = 7 * (h // 8) - oh
                cur_w = w // 10 + ow
                image[cur_h - 14:cur_h + 14, cur_w - 14:cur_w + 14, :] = 0
                sup_labels[cur_h - 14:cur_h + 14, cur_w - 14:cur_w + 14] = len(inv_class_map_classifier)

                # from where the crop is to out-of-screen (vertical direction)
                from_h = cur_h
                dest_h = h + 14
                dest_w = cur_w

            # saving the initial image (before the animation) and initial labels (before the animation)
            prev_image = np.array(image, copy=True)
            prev_sup_labels = np.array(sup_labels, copy=True)

            # animation length
            step_size = round(h * 0.01)
            steps = int((dest_h - from_h) / step_size)

            # animation loop for a single crop
            for f in range(0, steps):

                # placing crop
                image = np.array(prev_image, copy=True)
                place_crop(image, crop, (f * step_size + from_h if f != steps - 1 else dest_h, dest_w))

                # saving labels
                moving_pixels = np.sum(np.abs(prev_image - image), axis=2) > 0
                sup_labels = np.array(prev_sup_labels, copy=True)
                sup_labels[moving_pixels] = inv_class_map_classifier[class_name]

                # saving flow
                of = np.zeros((2, h, w), dtype=np.float)
                of[0, moving_pixels] = step_size  # vertical movement

                # checking if the moving thing is fully inside the scene
                object_fully_visible = True
                if np.abs(of[:, 0, :]).sum() > 0. or np.abs(of[:, -1, :]).sum() > 0. \
                        or np.abs(of[:, :, 0]).sum() > 0. or np.abs(of[:, :, -1]).sum() > 0.:
                    object_fully_visible = False

                # saving info
                info = {'stimulus_id': stimulus_id,
                        'first_of_stimulus': (z == 0 and f == 0),
                        'last_of_stimulus': (z == c and f == steps),
                        'object_id': z,
                        'first_of_object': f == 0,
                        'last_of_object': f == steps,
                        'object_fully_visible': object_fully_visible}

                # plotting
                if wanna_plot:
                    if ax_data is None:
                        ax_data = plt.imshow(image)
                        plt.show()
                    else:
                        ax_data.set_data(image)
                    plt.pause(0.04)

                # saving data to disk
                path = data_folder + os.sep
                file = "frame_and_sup_" + "{:06d}".format(k)
                if k == 1:
                    json.dump(inv_class_map_classifier, open(path + "inv_class_map_classifier.json", 'w'))
                    json.dump(class_map_classifier, open(path + "class_map_classifier.json", 'w'))
                np.savez_compressed(path + file,
                                    frame=image, sup_labels=sup_labels, of=of, info=info)
                print(path + "frame_sup_of_" + "{:06d}".format(k) + ".npz")

                # organize map from stimulus ID to involved files
                if stimulus_id not in stimulus_id_to_files.keys():
                    stimulus_id_to_files[stimulus_id] = []
                stimulus_id_to_files[stimulus_id].append(file)

                k += 1

            # updating offsets for next crop
            ow += 2 * (w // 10)
            if z == 4 or z == 9 or z == 14:
                ow = 0
                oh += 2 * (h // 8)

        # switching from entering to exiting (and vice-versa)
        entering = not entering

        # counting
        passes += 0.5
        if passes == 20:
            json.dump(stimulus_id_to_files, open(path + "stimulus_id_to_files.json", 'w'))
            break


def set_border_to(tensor, border, value):
    h = tensor.shape[-2]
    w = tensor.shape[-1]
    tensor[0:border, :] = value
    tensor[h - border:h, :] = value
    tensor[:, 0:border] = value
    tensor[:, w - border:w] = value


def ignore_motion_if_on_border(_of):
    of = _of.squeeze(0)
    of_mask = of.abs().sum(dim=0) != 0.
    if of[:, 0, :].abs().sum() > 0. or of[:, -1, :].abs().sum() > 0. \
            or of[:, :, 0].abs().sum() > 0. or of[:, :, -1].abs().sum() > 0.:
        of *= 0.
        is_on_border = True
    else:
        is_on_border = False
    return is_on_border, of_mask


def plot_saliency_maps(saliency_map_per_stimulus, stimuli_ids=None, path=None):
    # a = math.ceil(math.sqrt(len(saliency_map_per_stimulus)))
    # fig, axs = plt.subplots(round(len(saliency_map_per_stimulus) / float(a)), a, figsize=(10, 7))
    fig, axs = plt.subplots(6, 3, figsize=(7, 15))
    a = 3
    s = 0
    t = 0
    for j in range(0, len(saliency_map_per_stimulus)):
        frame = saliency_map_per_stimulus[j][0]
        xy = saliency_map_per_stimulus[j][1]
        seaborn.kdeplot(x=xy[:, 1], y=xy[:, 0],
                        fill=True, color='red', alpha=0.5, legend=False, ax=axs[s, t], zorder=2)
        axs[s, t].imshow(frame, zorder=1)
        axs[s, t].set_xlabel("Stimulus " + (str(stimuli_ids[j]) if stimuli_ids is not None else str(j)))
        axs[s, t].xaxis.set_ticklabels([])
        axs[s, t].yaxis.set_ticklabels([])
        axs[s, t].set_xticks([])
        axs[s, t].set_yticks([])
        t += 1
        if t >= a:
            t = 0
            s += 1
    if path is not None:
        plt.savefig(path + ".pdf", bbox_inches='tight')
    plt.show()
    return plt


def plot_saliency_maps_split(saliency_map_per_stimulus, stimuli_ids=None, path=None, seed=None):
    # fig, axs = plt.subplots(6, 3, figsize=(7, 15))
    # a = 3
    # s = 0
    # t = 0
    for j in range(0, len(saliency_map_per_stimulus)):
        fig, axs = plt.subplots()
        frame = saliency_map_per_stimulus[j][0]
        xy = saliency_map_per_stimulus[j][1]
        seaborn.kdeplot(x=xy[:, 1], y=xy[:, 0],
                        fill=True, color='red', alpha=0.5, legend=False, ax=axs, zorder=2)
        axs.imshow(frame, zorder=1)
        #axs.set_xlabel("Stimulus " + (str(stimuli_ids[j]) if stimuli_ids is not None else str(j)))
        axs.xaxis.set_ticklabels([])
        axs.yaxis.set_ticklabels([])
        axs.set_xticks([])
        axs.set_yticks([])

        if path is not None:
            plt.savefig(path + f"_{j}_{seed}.pdf", bbox_inches='tight')
        plt.show()
    return plt
