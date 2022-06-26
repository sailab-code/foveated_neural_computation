import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from fov_conv2d_reg import FovConv2dReg
from geymol import GEymol
import time
import socket
import random
import sys
import gc
from collections import OrderedDict
from stream_attention_utils import plot, compute_accuracies, update_drawing_panel, \
    read_result_log, save_result_log, compute_confusion_matrices, build_basic_drawing_panel, \
    create_legend, create_dataset, load_dict_from_json, compute_accuracies_per_class, accuracies_per_class_to_string, \
    plot_saliency_maps, plot_saliency_maps_split
from stream_attention_nets import create_networks
import argparse
import wandb
from pathlib import Path
import copy

# ======================================================================================================================
# configuration (misc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--net', type=str, default="fov_reg_all_2")
    parser.add_argument('--w', type=int, default=200)
    parser.add_argument('--fixed_seed', type=str, default="True",
                        help='For logging the model in tensorboard')
    parser.add_argument('--seed', type=int, default=12146543)
    parser.add_argument('--print_every', type=int, default=5000)
    parser.add_argument('--eval', type=str, default="false")
    parser.add_argument('--wandb', type=str, default="False",
                        help='Log the model in wandb?')

    args = parser.parse_args()

    args.fixed_seed = args.fixed_seed in {'True', 'true'}
    args.wandb = args.wandb in {'True', 'true'}

    if args.fixed_seed:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    root_exp_folder = 'results_exp_pane'
    os.makedirs(root_exp_folder, exist_ok=True)
    log_file_root = f'{args.net}__lr_{args.lr}__w_{args.w}'
    log_dir = os.path.join(root_exp_folder, log_file_root)
    if args.fixed_seed:
        log_dir += "__seed_" + str(args.seed)

    if os.path.exists(log_dir):
        print(f"Experiment {log_dir} already available!")
        if args.eval:
            pass
        else:
            exit()
    else:
        os.makedirs(log_dir, exist_ok=True)

    log_file = log_file_root + '.log'
    # save to disk the arguments
    dict_params = vars(args)
    json.dump(dict_params, open(os.path.join(log_dir, "config.json"), 'w'))

    # configuration (dataset)
    h = args.w
    w = args.w

    data_folder = os.path.join('data', 'pane_dataset')
    if args.w != 200:
        data_folder = data_folder + f"_{args.w}"
    # data_folder = os.path.join('pane_dataset')
    net_output_classes = 20

    # configuration (buffered supervisions)
    buffer_device = device
    max_buffer_size = 1000
    mini_batch_size = 5
    learn_from_buffered_data_only = True

    # configuration (experiment)
    remove_bg_supervisions = False
    weight_of_bg_pixels = 1e-5
    process_only_the_first_channel_of_the_images = True

    architecture_type = args.net
    lr = args.lr

    # update logdir

    geymol_opts = {'alpha_c': 1.0,
                   'alpha_of': 5.0,
                   'alpha_virtual': 10.0,
                   'max_distance': int(0.5 * (w + h)) if int(0.5 * (w + h)) % 2 == 1 else int(0.5 * (w + h)) + 1,
                   'dissipation': 3.0,
                   'fps': 25,
                   'w': w,
                   'h': h,
                   'y': None,
                   'is_online': False,
                   'alpha_fm': 0.0,
                   'static_image': False,
                   'fixation_threshold_speed': 0.1 * (0.5 * (w + h)),
                   "ior_ray": 0.02 * min(h, w),
                   "ior_blur": 0.15 * min(h, w)}

    # configuration (plan)
    all_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    all_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
    even = ['0', '2', '4', '6', '8']
    odd = ['1', '3', '5', '7', '9']
    mix1 = ['0', '1', '2', 'A', 'B', 'C']
    mix2 = ['7', '8', '9', 'H', 'J', 'K']

    train_stimuli = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    train_supervised_classes = [all_digits, all_letters]
    test_stimuli = [[15, 16, 17, 15, 16, 17, 15, 16, 17],
                    [15, 16, 17, 15, 16, 17, 15, 16, 17]]
    static_test_stimuli = [[True, True, True, True, True, True, True, True, True],
                           [True, True, True, True, True, True, True, True, True]]
    interesting_classes_per_test_stimulus = [[all_digits, all_digits, all_digits,
                                              even, even, even,
                                              odd, odd, odd],
                                             [all_letters, all_letters, all_letters,
                                              mix1, mix1, mix1,
                                              mix2, mix2, mix2]]

    interesting_classes_per_test_stimulus_metric = [all_digits, all_digits, all_digits,
                                                    even, even, even,
                                                    odd, odd, odd,
                                                    all_letters, all_letters, all_letters,
                                                    mix1, mix1, mix1,
                                                    mix2, mix2, mix2]
    # ======================================================================================================================

    # if an argument is provided, then it is interpreted as a log-file, and the corresponding results are plotted
    eval_only = False

    if args.eval != "false":
        # log_file = os.path.join(log_dir, args.eval)
        eval_only = True
        files = []

    # initializing everything
    if not eval_only:

        # networks
        net, classifier = create_networks(architecture_type, net_output_classes, device)

        # optimizer (Adam, usually: 0.001)
        optimizer = torch.optim.Adam(list(net.parameters()) + list(classifier.parameters()), lr)

        # attention
        geymol = GEymol(geymol_opts, device=device)

        # checking if the dataset should be created
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            create_dataset(data_folder, h, w)

        # loading data configuration files
        path = data_folder + os.sep
        inv_class_map_classifier = load_dict_from_json(path + "inv_class_map_classifier.json", force_int_values=True)
        class_map_classifier = load_dict_from_json(path + "class_map_classifier.json", force_int_keys=True)
        stimulus_id_to_files = load_dict_from_json(path + "stimulus_id_to_files.json", force_int_keys=True)

        # fixing options
        device = torch.device(device)
        buffer_device = torch.device(buffer_device)
        alpha_c = geymol.parameters['alpha_c']
        alpha_of = geymol.parameters['alpha_of']
        alpha_virtual = geymol.parameters['alpha_virtual']

        # misc initializations
        i = 1
        j = 0
        data = None
        ax_data = None
        legend = None
        logs = []
        prev_saccade = False
        buffer = {'image': [], 'foa_xy': [], 'sup_labels_1_hot': [], 'sup_pixel_weights': []}
        buffer_size = 0
        last_buffered_idx = -1
        custom_mass = torch.zeros((h, w), device=device)
        data_idx = 0  # init
        order = None  # init
        first_of_stimulus = [True] * 200  # 200 is just a "big number" of stimuli
        conf_mat_labels = list(class_map_classifier.keys())  # class indices
        conf_mat_labels.append(len(class_map_classifier.keys()))  # background class index

        # initial drawing - legend
        legend, legend_labels = create_legend(class_map_classifier, (w, int(h * 0.075)))
        legend = legend / 255.

        # initial drawing - empty panel with legend
        data = build_basic_drawing_panel(h, w, legend)

        # initial drawing - interactive mode
        plt.ion()

        # initial drawing - drawing and showing
        ax_data = plt.imshow(data)
        plt.show()
        plt.pause(0.000001)

        # loading list of files
        files = [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[-4:] == '.npz']
        files.sort()

    # checking
    assert len(train_stimuli) == len(test_stimuli)
    assert len(test_stimuli) == len(interesting_classes_per_test_stimulus)
    assert len(static_test_stimuli) == len(test_stimuli)

    # running sessions (each session is train + test)
    processed_stimulus_id = -1
    for s in range(0, len(train_stimuli)):
        if eval_only:
            break

        _train_stimuli = train_stimuli[s]
        _train_sup_classes = [inv_class_map_classifier[k] for k in train_supervised_classes[s]]
        _test_stimuli = test_stimuli[s]
        _static_test_stimuli = static_test_stimuli[s]
        _interesting_classes_per_test_stimulus = interesting_classes_per_test_stimulus[s]
        _k = 0

        # running a single session
        for st in _train_stimuli + _test_stimuli:
            processed_stimulus_id += 1
            files = stimulus_id_to_files[st]

            # resetting attention at the beginning of each stimulus
            custom_mass *= 0.
            geymol.IOR_matrix *= 0.
            geymol.reset(None)

            # setting up supervisions or interesting classes
            if st in _train_stimuli:
                geymol.static_image = False
                geymol.parameters['alpha_c'] = alpha_c
                geymol.parameters['alpha_of'] = alpha_of
                interesting_classes_idx = None
                supervised_classes = _train_sup_classes
                is_test = False

            if st in _test_stimuli:
                if _k == 0:
                    torch.save({'net': net.state_dict(), 'classifier': classifier.state_dict()},
                               os.path.join(log_dir, "net_and_classifier_after_train_session_" + str(s) + ".pt"))
                if _static_test_stimuli[_k]:
                    files = [files[len(files) // 2]] * len(files)
                    geymol.static_image = True
                geymol.parameters['alpha_c'] = 0.
                geymol.parameters['alpha_of'] = 0.
                interesting_classes = _interesting_classes_per_test_stimulus[_k]
                interesting_classes_idx = torch.tensor([inv_class_map_classifier[k] for k in interesting_classes],
                                                       dtype=torch.long, device=device)
                supervised_classes = None
                is_test = True
                _k += 1

            # files = [files[len(files) // 2]]  # debug

            # exploration loop
            for file in files:

                # load current frame data
                loaded_data = np.load(path + file + ".npz", allow_pickle=True)
                info = loaded_data['info'].item()
                np_image = loaded_data['frame'] / 255.
                sup_labels = loaded_data['sup_labels']
                sup_labels = sup_labels.reshape(h * w)  # flat labels
                of = loaded_data['of']

                # initially selecting all supervisions, for all classes (weight = 1 for all of them)
                sup_pixel_weights = torch.ones((1, net_output_classes, h * w),
                                               dtype=torch.float32)  # flat weights over pxs

                # if needed, filtering labels attached to background pixels (weight = 0)
                bg_mask = torch.from_numpy(sup_labels == net_output_classes)
                if remove_bg_supervisions:
                    sup_pixel_weights[:, :, bg_mask] = 0.

                # determining the available supervisions in function of the stimulus
                if supervised_classes is not None:
                    prune_pixels = np.logical_not(np.isin(sup_labels, supervised_classes))
                    for jj in range(0, net_output_classes):
                        if jj not in supervised_classes:
                            sup_pixel_weights[:, jj, torch.from_numpy(prune_pixels)] = 0.
                else:
                    sup_pixel_weights *= 0.

                # torch data - image
                image = torch.tensor(np_image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
                if process_only_the_first_channel_of_the_images:
                    image = image[:, 0, None, :,
                            :]  # picking up a single channel (useful if all channels are equivalent)

                # torch data - optical flow
                of = torch.tensor(of).unsqueeze(0)

                # torch data - supervisions (binary supervisions, frequently called 1-hot here,
                # even if sometimes they are 0-hot)
                sup_labels_1_hot = torch.nn.functional.one_hot(
                    torch.tensor(sup_labels, dtype=torch.long),
                    num_classes=net_output_classes + 1).t().unsqueeze(0).to(torch.float32)  # added a fake class (bg)

                # torch data - supervisions, removing fake background class (always remove it!)
                sup_labels_1_hot = sup_labels_1_hot[:, 0:net_output_classes, :]  # flat 1-hot labels (over pixels)

                # torch data - creating the supervision mass
                fix = 1. if info['object_fully_visible'] else \
                    (torch.sum(of.abs(), dim=1).view(h * w) == 0.).to(torch.float32)
                sup_mass = torch.sum(sup_labels_1_hot * sup_pixel_weights * fix, dim=1)
                sup_mass = sup_mass.view(h, w)
                positive_supervision_available = (torch.sum(sup_mass) > 0.).item()

                # FOA - activate extra (virtual) mass only if needed
                if buffer_size == 0 and not positive_supervision_available:
                    geymol.parameters['alpha_virtual'] = 0.
                else:
                    geymol.parameters['alpha_virtual'] = alpha_virtual

                # FOA - let's move!
                foa_geymol, saccade = geymol.next_location(image.to(device),
                                                           of_t=of.to(device) if info['object_fully_visible']
                                                           else of.to(device) * 0.,
                                                           virtualmass=custom_mass
                                                           if not positive_supervision_available else sup_mass.to(
                                                               device))
                foa_xy = torch.tensor(foa_geymol[0:2], dtype=torch.long).unsqueeze(0)

                predict = True
                # predict = False  # debug
                if predict:

                    # fixing weight of background pixels (do not do it before having created the supervision mass!)
                    # moreover: do it before preparing the mini-batch
                    bg_sup = (torch.sum(sup_labels_1_hot, dim=1,
                                        keepdim=True) == 0).to(torch.bool).tile(1, net_output_classes, 1)
                    sup_pixel_weights[bg_sup] *= weight_of_bg_pixels  # background examples are weighed less

                    # building mini-batch (first element is current image/of/etc.)
                    b_image = []
                    b_foa_xy = []
                    b_sup_labels_1_hot = []
                    b_sup_pixel_weights = []
                    b_image.append(image.to(buffer_device))
                    b_foa_xy.append(foa_xy.to(buffer_device))
                    b_sup_labels_1_hot.append(sup_labels_1_hot.to(buffer_device))
                    b_sup_pixel_weights.append(sup_pixel_weights.to(buffer_device))

                    if buffer_size > 0:
                        while len(b_image) < mini_batch_size:
                            if order is not None and data_idx < order.shape[0]:
                                j = order[data_idx]
                            elif data_idx < buffer_size:
                                j = data_idx
                            else:
                                order = np.arange(buffer_size)
                                np.random.shuffle(order)
                                data_idx = 0
                                j = order[data_idx]
                            b_image.append(buffer['image'][j])
                            b_foa_xy.append(buffer['foa_xy'][j])
                            b_sup_labels_1_hot.append(buffer['sup_labels_1_hot'][j])
                            b_sup_pixel_weights.append(buffer['sup_pixel_weights'][j])
                            data_idx += 1

                    # packing and sending to the target device
                    b_image = torch.cat(b_image, dim=0).to(device)
                    b_foa_xy = torch.cat(b_foa_xy, dim=0).to(device)
                    b_sup_labels_1_hot = torch.cat(b_sup_labels_1_hot, dim=0).to(device)
                    b_sup_pixel_weights = torch.cat(b_sup_pixel_weights, dim=0).to(device)

                    # extracting features
                    t = time.time()
                    b_feats, b_region_idx = net(b_image, b_foa_xy)

                    # classify pixels
                    b_predictions = torch.sigmoid(classifier(b_feats).view(b_image.shape[0], net_output_classes, -1))
                    predictions_detached = b_predictions[0, None, :, :].detach().squeeze(0)
                    t = time.time() - t

                    # decisions and related stuff
                    decision, conf_matrix, conf_matrix_focussed = \
                        compute_confusion_matrices(predictions_detached.cpu().numpy(),
                                                   sup_labels, h, w, foa_xy[0, 0].item(), foa_xy[0, 1].item(),
                                                   conf_mat_labels,
                                                   bg_thres=0.2)
                    acc_per_class = compute_accuracies_per_class(conf_matrix)

                    # shapes and extra info
                    decision = decision.reshape(h, w)
                    sup_labels = sup_labels.reshape(h, w)
                    focussed_class = sup_labels[foa_xy[0, 0], foa_xy[0, 1]]

                    # checking if a supervised frame should be buffered
                    frame_was_buffered = False
                    if i > 1 and \
                            (not saccade and prev_saccade) \
                            and positive_supervision_available and sup_mass[foa_xy[0, 0], foa_xy[0, 1]] > 0. \
                            and decision[foa_xy[0, 0], foa_xy[0, 1]] != focussed_class:
                        if buffer_size < max_buffer_size:
                            last_buffered_idx += 1
                            buffer['image'].append(image.to(buffer_device))
                            buffer['foa_xy'].append(foa_xy.to(buffer_device))
                            buffer['sup_labels_1_hot'].append(sup_labels_1_hot.to(buffer_device))
                            buffer['sup_pixel_weights'].append(sup_pixel_weights.to(buffer_device))
                        else:
                            last_buffered_idx += 1
                            last_buffered_idx = last_buffered_idx % max_buffer_size
                            buffer['image'][last_buffered_idx] = image.to(buffer_device)
                            buffer['foa_xy'][last_buffered_idx] = foa_xy.to(buffer_device)
                            buffer['sup_labels_1_hot'][last_buffered_idx] = sup_labels_1_hot.to(buffer_device)
                            buffer['sup_pixel_weights'][last_buffered_idx] = sup_pixel_weights.to(buffer_device)
                        buffer_size = len(buffer['image'])
                        frame_was_buffered = True
                    prev_saccade = saccade

                    # computing loss function (after having fixed pixel weights)
                    if learn_from_buffered_data_only:
                        b_sup_pixel_weights[-1, :, :] = 0.
                    sum_w = torch.sum(b_sup_pixel_weights)
                    if sum_w == 0.:
                        sum_w = 1.0
                    b_sup_pixel_weights = b_sup_pixel_weights / sum_w
                    obj_value = torch.nn.functional.binary_cross_entropy(b_predictions, b_sup_labels_1_hot,
                                                                         reduction="sum", weight=b_sup_pixel_weights)
                else:
                    obj_value = None
                    acc_per_class = torch.zeros(net_output_classes)
                    frame_was_buffered = False
                    t = 0.
                    sup_labels = sup_labels.reshape(h, w)
                    focussed_class = sup_labels[foa_xy[0, 0], foa_xy[0, 1]]
                    conf_matrix = torch.zeros((net_output_classes, net_output_classes))
                    conf_matrix_focussed = torch.zeros((net_output_classes, net_output_classes))
                    decision = torch.zeros(h, w)

                # printing
                if i % args.print_every == 0:
                    print("frame: " + str(i) + ", stimulus_id: " + str(info['stimulus_id']) +
                          ", proc_stimulus_id: " + str(processed_stimulus_id) +
                          ", buffered: " + str(buffer_size) + ", obj_value: "
                          + ("{:.6f}".format(obj_value.item()) if obj_value is not None else "n/a") +
                          ", focussed_class: " + (class_map_classifier[focussed_class]
                                                  if focussed_class in class_map_classifier.keys() else "bg") +
                          ", acc: [" + accuracies_per_class_to_string(acc_per_class) + "]" +
                          ((" * buffer_" + "{:03d}".format(last_buffered_idx)) if frame_was_buffered else ""))

                # logging
                logs.append(OrderedDict({'frame': i, 'stimulus_id': info['stimulus_id'],
                                         'processed_stimulus_id': processed_stimulus_id, 'is_test': is_test, 'time': t,
                                         'focussed_class': focussed_class, 'buffer_size': buffer_size,
                                         'foa_x': foa_xy[0, 0].item(),
                                         'foa_y': foa_xy[0, 1].item(),
                                         'focussed_moving_obj':
                                             int(torch.sum(torch.abs(of[0, :, foa_xy[0, 0], foa_xy[0, 1]])) > 1e-3),
                                         'classes': conf_matrix.shape[0],
                                         'confusion_matrix': str(list(conf_matrix.flatten())),
                                         'confusion_matrix_focussed': str(list(conf_matrix_focussed.flatten()))}))

                # loading hot options and reacting
                if os.path.isfile("hot_options.json"):
                    hot_options = json.load(open("hot_options.json", 'r'))
                    if hot_options['stop'] == 1:
                        break
                else:
                    hot_options = None

                # drawing
                j += 1
                if not hot_options or 1 <= hot_options['plot_every'] <= j:
                    update_drawing_panel(data, np_image, decision, legend, legend_labels,
                                         net_output_classes, foa_xy[0, 0].item(), foa_xy[0, 1].item(),
                                         [geymol.IOR_matrix.cpu(), None,
                                          geymol.gradient_norm_t.cpu(), geymol.of_norm_t.cpu(),
                                          geymol.virtual_mass_t.cpu()],
                                         focussed_region_size=net.layer_3.region_sizes[0][0]
                                         if type(net.layer_3) is FovConv2dReg else None,
                                         highlight_frame=frame_was_buffered)
                    ax_data.set_data(data)
                    plt.pause(0.000001)
                    # plt.pause(1)  # debug

                    if frame_was_buffered:
                        plt.savefig(os.path.join(log_dir, "buffer_" + "{:03d}".format(last_buffered_idx) + ".jpg"))
                    j = 0

                # optimizing
                if predict:
                    net.zero_grad()
                    classifier.zero_grad()
                    obj_value.backward()
                    if not is_test:
                        optimizer.step()

                    # augment the custom mass
                    if interesting_classes_idx is not None:
                        mask = torch.zeros((h, w), dtype=torch.float32, device=device)
                        for jj in interesting_classes_idx:
                            mask[decision == jj.item()] = 1.
                            # mask[sup_labels == jj.item()] = 1.  # debug
                        custom_mass = mask
                        # custom_mass = torch.max(predictions_detached[interesting_classes_idx, :], dim=0)[0].view(h, w)

                # saving log (running)
                if not eval_only and i % 500 == 0 or i == 1:
                    save_result_log(os.path.join(log_dir, log_file), logs, append=(i > 1))
                    logs = []

                # incrementing frame counter
                i += 1

                # clearing garbage from time to time... (paranoid)
                if i % 100 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    if args.wandb:
        total_frames = copy.deepcopy(i)
    # saving log (after stop)
    if not eval_only:
        if i > 1 and len(logs) > 0:
            save_result_log(os.path.join(log_dir, log_file), logs, append=True)
        plt.ioff()
        plt.close('all')
        torch.save({'net': net.state_dict(), 'classifier': classifier.state_dict()},
                   os.path.join(log_dir, "net_and_classifier.pt"))

    # reading log and collecting (a lot of) data
    conf_matrix_total, conf_matrices_per_stimulus, \
    conf_matrix_focussed_total, conf_matrices_focussed_per_stimulus, \
    conf_matrix_peri_total, conf_matrices_peri_per_stimulus, \
    focussed_classes_total, focussed_classes_per_stimulus, \
    focussed_moving_classes_total, focussed_moving_classes_per_stimulus, \
    average_time, tot_frames, foa_xs_per_stimulus, foa_ys_per_stimulus, stimulus_id_per_stimulus = \
        read_result_log(os.path.join(log_dir, log_file), test_only=True)

    # accuracy: over the whole frame
    acc_total_unbalanced, acc_total_balanced, acc_total_per_class, \
    acc_per_stimulus_unbalanced, acc_per_stimulus_balanced, acc_per_stimulus_per_class = \
        compute_accuracies(conf_matrix_total, conf_matrices_per_stimulus)

    # accuracy: over the focussed part of the frame
    acc_focussed_total_unbalanced, acc_focussed_total_balanced, acc_focussed_total_per_class, \
    acc_focussed_per_stimulus_unbalanced, acc_focussed_per_stimulus_balanced, acc_focussed_per_stimulus_per_class = \
        compute_accuracies(conf_matrix_focussed_total, conf_matrices_focussed_per_stimulus)

    # accuracy: over the peripheral (not focussed) part of the frame
    acc_peri_total_unbalanced, acc_peri_total_balanced, acc_peri_total_per_class, \
    acc_peri_per_stimulus_unbalanced, acc_peri_per_stimulus_balanced, acc_peri_per_stimulus_per_class = \
        compute_accuracies(conf_matrix_peri_total, conf_matrices_peri_per_stimulus)

    # results dict
    results = {}

    # printing (time)
    print('average_time: ' + str(average_time) + ' seconds')

    # printing (accuracies)
    print('acc_total_per_class:')
    path = data_folder + os.sep
    inv_class_map_classifier = load_dict_from_json(path + "inv_class_map_classifier.json", force_int_values=True)
    class_map_classifier = load_dict_from_json(path + "class_map_classifier.json", force_int_keys=True)
    for i in range(0, len(acc_total_per_class)):
        print('   ' + ((class_map_classifier[i] + ' : ') if i in class_map_classifier.keys() else 'bg: ')
              + str(acc_total_per_class[i]))
        results[f"acc_class_{class_map_classifier[i]}" if i in class_map_classifier.keys() else "acc_class_bg"] = \
            acc_total_per_class[i]
    print('acc_total_unbalanced: ' + str(acc_total_unbalanced))
    results["acc_total_unbalanced"] = acc_total_unbalanced
    print('acc_total_balanced:   ' + str(acc_total_balanced))
    results["acc_total_balanced"] = acc_total_balanced

    # compute average accuracy per class groups
    results["digit_mean_acc"] = np.mean(acc_total_per_class[:10])
    results["letters_mean_acc"] = np.mean(acc_total_per_class[10:-1])
    results["acc_background"] = acc_total_per_class[-1]

    # compute metrics on FOA location

    results["interest_all_digits"] = 0
    results["interest_all_letters"] = 0
    results["interest_even"] = 0
    results["interest_odd"] = 0
    results["interest_mix1"] = 0
    results["interest_mix2"] = 0

    # loop over all the stimuli
    for stim in range(len(interesting_classes_per_test_stimulus_metric)):
        # get the interesting classes of the current stimulus
        ic = interesting_classes_per_test_stimulus_metric[stim]
        # get the number of frames spent on each class for the current stimulus
        current_stimulus_focussed = focussed_classes_per_stimulus[stim]
        # count the aggregated number of frames spent in each of the interesting classes for this stimulus
        frame_counter = 0
        # loop over all the interesting classes of the current stimulus (it is a string)
        for z in ic:  # remember to consider bg
            # get index of the interesting class
            idx = inv_class_map_classifier[z]
            # get the number of frames and aggregate
            frame_counter += current_stimulus_focussed[idx]
        if ic == all_digits:
            results["interest_all_digits"] += int(frame_counter)
        elif ic == all_letters:
            results["interest_all_letters"] += int(frame_counter)
        elif ic == even:
            results["interest_even"] += int(frame_counter)
        elif ic == odd:
            results["interest_odd"] += int(frame_counter)
        elif ic == mix1:
            results["interest_mix1"] += int(frame_counter)
        elif ic == mix2:
            results["interest_mix2"] += int(frame_counter)
        else:
            raise NotImplementedError

    # every group of interest appears 3 times, and every one lasts 2260 frames
    results["interest_all_digits_ratio"] = results["interest_all_digits"] / (2260. * 3)
    results["interest_all_letters_ratio"] = results["interest_all_letters"] / (2260. * 3)
    results["interest_even_ratio"] = results["interest_even"] / (2260. * 3)
    results["interest_odd_ratio"] = results["interest_odd"] / (2260. * 3)
    results["interest_mix1_ratio"] = results["interest_mix1"] / (2260. * 3)
    results["interest_mix2_ratio"] = results["interest_mix2"] / (2260. * 3)

    stem = Path(log_dir).name
    json.dump(results, open(os.path.join(log_dir, f"metrics_{stem}.json"), 'w'))

    # plotting (accuracies)
    print("Generating accuracy plot...")
    plt = plot(acc_per_stimulus_per_class, 'Stimulus', 'Accuracy (Everywhere)',
               acc_focussed_per_stimulus_per_class, 'Stimulus', 'Accuracy (Focussed)',
               acc_peri_per_stimulus_per_class, 'Stimulus', 'Accuracy (Peripheral)',
               stimuli_ids=stimulus_id_per_stimulus, path=os.path.join(log_dir, "accuracy.pdf"))
    if args.wandb:
        wandb.log({"accuracy_plot": wandb.Image(plt)}, step=total_frames)

    # printing or plotting (focussed areas)
    print("Generating focussed classes plot...")
    plt = plot(focussed_classes_per_stimulus, 'Stimulus', 'Distr. of Time Over Classes',
               normalize=True, ignore_last_class=True,
               stimuli_ids=stimulus_id_per_stimulus, path=os.path.join(log_dir, "focussed_classes.pdf"))
    if args.wandb:
        wandb.log({"focussed_accuracy_plot": wandb.Image(plt)}, step=total_frames)

    # loading file names for each stimulus
    print("Generating saliency maps... (it might take some time!)")
    stimulus_id_to_files = load_dict_from_json(path + "stimulus_id_to_files.json", force_int_keys=True)

    # saliency maps
    saliency_map_per_stimulus = []
    for s in range(0, len(stimulus_id_per_stimulus)):
        stimulus_id = stimulus_id_per_stimulus[s]

        # getting the file in the middle of a test stimulus
        files = stimulus_id_to_files[stimulus_id]
        file = files[len(files) // 2]
        frame = np.load(path + file + ".npz", allow_pickle=True)['frame'] / 255.

        foa_xs = foa_xs_per_stimulus[s]
        foa_ys = foa_ys_per_stimulus[s]

        xy = np.stack([foa_xs, foa_ys], axis=1)
        saliency_map_per_stimulus.append([frame, xy])

    # plt = plot_saliency_maps(saliency_map_per_stimulus, stimuli_ids=stimulus_id_per_stimulus,
    #                          path=os.path.join(log_dir, "saliency"))
    plt = plot_saliency_maps_split(saliency_map_per_stimulus, stimuli_ids=stimulus_id_per_stimulus,
                                   path=os.path.join(log_dir, "saliency"), seed=args.seed)
    if args.wandb:
        wandb.log({"saliency_plot": wandb.Image(plt)}, step=total_frames)
