import gc
import time
import torch
import numpy as np
import copy
import csv
import collections
import sys
import matplotlib.pyplot as plt
from fov_conv2d_reg import FovConv2dReg
import os


# *************
# CONFIGURATION
# *************
device = "cuda:0" if torch.cuda.is_available() else "cpu"
wh = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)]
in_features = [64]
out_features = [64]
kernels = [3, 5, 7, 9, 11, 15, 21, 29]
region_counts = [2, 4, 6, 8]
region_types = ["circle"]  # ["box", "circle"]
methods = ["baseline", "vanilla", "downscaling", "stride", "dilation"]  # ["baseline","vanilla","downscaling","stride","dilation"]
banks = ["independent"]  # ["independent", "shared"]
batch_sizes = [1]  # [1, 16]
crop_types = ["loose"]  # ["none", "loose", "tight"]
focus_areas = 1  # middle, top left, top right, bottom left, bottom right
repetitions = 3
log_file = "fovlayer_time.csv"
plot = True
plot_only = False
plot_settings = [{'curves': ['method', 'crop_type'],
                  'fixed': {'kernel_size': 11},
                  'x': 'w', 'y': 'avg_time',
                  'ignore': ['focus_areas', 'h', 'region_sizes', 'reduction_factors', 'repetitions']},
                 {'curves': ['method', 'crop_type'],
                  'fixed': {'kernel_size': 29},
                  'x': 'w', 'y': 'avg_time',
                  'ignore': ['focus_areas', 'h', 'region_sizes', 'reduction_factors', 'repetitions']},
                 {'curves': ['method', 'crop_type'],
                  'fixed': {'w': 512},
                  'x': 'kernel_size', 'y': 'avg_time',
                  'ignore': ['focus_areas', 'h', 'region_sizes', 'reduction_factors', 'repetitions']},
                 {'curves': ['method', 'crop_type'],
                  'fixed': {'w': 512},
                  'x': 'kernel_size', 'y': 'avg_time',
                  'ignore': ['focus_areas', 'h', 'region_sizes', 'reduction_factors', 'repetitions']}]


# *********
# FUNCTIONS
# *********
def run(opt, input_data, first_torch_call=False):
    experiments = []
    _optc = copy.deepcopy(opt)

    # computing FOA locations
    h = int(_optc['h'])
    w = int(_optc['w'])
    foas = [[h // 2, w // 2],
            [h // 4, w // 4],
            [h // 4, (3 * w) // 4],
            [(3 * h) // 4, w // 4],
            [(3 * h) // 4, (3 * w) // 4]]

    b = _optc['batch_size']
    foas_xy = []
    for _i in range(0, _opt['focus_areas']):
        foa_xy = torch.cat([torch.tensor(foas[_i]).view(1, 2)] * b, dim=0)
        foa_xy = foa_xy.to(torch.long).to(torch.device(_optc['device']))
        foas_xy.append(foa_xy)

    # computing region sizes and reduction factors
    scales = np.arange(0.1, 0.71, (0.7 - 0.1) / (_optc['region_count'] - 2)) \
        if _optc['region_count'] > 2 else np.array([0.1])

    region_sizes = list(np.round(scales * min(_optc['h'], _optc['w'])))
    for i in range(0, len(region_sizes)):
        if region_sizes[i] % 2 == 0:
            region_sizes[i] += 1
    region_sizes.append(-1)

    reduction_factors = list(np.round(np.arange(1.0, 0.099, -(1.0 - 0.1) / (_optc['region_count'] - 1)) * 100) / 100.0)

    _optc['region_sizes'] = region_sizes
    _optc['reduction_factors'] = reduction_factors

    # running
    avg_time = 0
    first_call_done = False
    for _r in range(0, _optc['repetitions']):
        for _i in range(0, len(foas_xy)):
            print('Options: ' + str(_optc))
            print('Repetition ' + str(_r + 1) + '/' + str(_optc['repetitions']))
            print('Focus area ' + str(_i + 1) + '/' + str(_optc['focus_areas']))
            print('--- Input shape: ' + str(input_data.shape))

            fov_layer = None
            conv_layer = None

            if _optc['method'] != 'baseline':
                fov_layer = FovConv2dReg(region_type=_optc['region_type'], method=_optc['method'],
                                         in_channels=_optc['in_features'],
                                         out_channels=_opt['out_features'], kernel_size=_optc['kernel_size'],
                                         region_sizes=_optc['region_sizes'], reduction_factors=_optc['reduction_factors'],
                                         banks=_optc['banks'],
                                         crop_type=_optc['crop_type']).to(torch.device(_optc['device']))
            else:
                conv_layer = torch.nn.Conv2d(in_channels=_optc['in_features'],
                                             out_channels=_opt['out_features'], kernel_size=_optc['kernel_size'],
                                             stride=1, dilation=1,
                                             padding=int(_optc['kernel_size']) // 2).to(torch.device(_optc['device']))

            if first_torch_call and not first_call_done:
                if _optc['method'] != 'baseline':
                    output_data, region_indices = fov_layer(input_data, foas_xy[_i],
                                                            compute_region_indices=True)
                else:
                    output_data = conv_layer(input_data)
                c = output_data[0, 0, 0, 0].item() + 3  # dummy operation
                print('--- (Fist Call Startup) Output shape: ' + str(output_data.shape))
                print('--- (Fist Call Startup) Dummy element: ' + str(c))

                del fov_layer
                del conv_layer
                del output_data
                if device[0:4] == "cuda":
                    gc.collect()
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                fov_layer = None
                conv_layer = None

                if _optc['method'] != 'baseline':
                    fov_layer = FovConv2dReg(region_type=_optc['region_type'], method=_optc['method'],
                                             in_channels=_optc['in_features'],
                                             out_channels=_opt['out_features'], kernel_size=_optc['kernel_size'],
                                             region_sizes=_optc['region_sizes'],
                                             reduction_factors=_optc['reduction_factors'],
                                             banks=_optc['banks'],
                                             crop_type=_optc['crop_type']).to(torch.device(_optc['device']))
                else:
                    conv_layer = torch.nn.Conv2d(in_channels=_optc['in_features'],
                                                 out_channels=_opt['out_features'], kernel_size=_optc['kernel_size'],
                                                 stride=1, dilation=1,
                                                 padding=int(_optc['kernel_size'])//2).to(torch.device(_optc['device']))
                first_call_done = True

            t = time.time()
            if _optc['method'] != 'baseline':
                # x = torch.clone(input_data)
                # f = torch.clone(foas_xy[_i])
                # fov_layer = torch.cuda.make_graphed_callables(fov_layer, (x, f, ))
                output_data, region_indices = fov_layer(input_data, foas_xy[_i],
                                                        compute_region_indices=True)
            else:
                output_data = conv_layer(input_data)
            c = output_data[0, 0, 0, 0].item() + 3  # dummy operation
            elapsed = time.time() - t

            print('--- Output shape: ' + str(output_data.shape))
            print('--- Dummy element: ' + str(c))
            print('--- Elapsed: ' + str(elapsed) + 's')
            if output_data.shape[2] != input_data.shape[2] or output_data.shape[3] != input_data.shape[3]:
                print('*** Mismatching shapes! (quitting)')
                sys.exit(0)
            print('')

            avg_time += elapsed

            del fov_layer
            del conv_layer
            del output_data
            if device[0:4] == "cuda":
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

    _optc['avg_time'] = avg_time / float(_optc['focus_areas'] * _optc['repetitions'])
    return _optc


def log_experiments(_exps, append=False):
    """Log some experiments to file (each experiment is a dictionary, this function expects a list of experiments)."""
    with open(log_file, 'a' if append else 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_exps[0].keys())
        if not append:
            writer.writeheader()
        writer.writerows(_exps)


def load_results(file):
    loaded_exps = []

    if not os.path.exists(file):
        return loaded_exps

    with open(file) as f:
        reader = csv.reader(f, skipinitialspace=True)
        ints = ['focus_areas', 'w', 'h', 'batch_size', 'region_count', 'in_features', 'out_features', 'kernel_size',
                'repetitions']
        booleans = []
        floats = ['avg_time']
        strings = ['device', 'banks', 'region_type', 'method', 'region_sizes', 'reduction_factors', 'crop_type']
        header = next(reader)
        for row in reader:
            dd = collections.OrderedDict(zip(header, row))
            for k, v in dd.items():
                if k in ints:
                    dd[k] = int(v)
                elif k in floats:
                    dd[k] = float(v)
                elif k in strings:
                    dd[k] = str(v)
                elif k in booleans:
                    dd[k] = True if v.lower() == 'true' else False
                else:
                    print('*** Missing map for key: ' + str(k))
                    sys.exit(0)
            loaded_exps.append(dd)
    return loaded_exps


def hash_keys_values(d, ignore_keys=[]):
    _hash = ''
    i = 0
    for _tk, _tv in d.items():
        if _tk in ignore_keys:
            continue
        if i > 0:
            _hash += '-'
        _hash += _tk + '_' + str(_tv)
        i += 1
    return _hash


def hash_results(exps, ignore_keys=[]):
    hash_to_done = {}
    for exp in exps:
        hash_to_done[hash_keys_values(exp, ignore_keys)] = True
    return hash_to_done


def collect_and_plot(file, setts):
    """Load results from file, organize them, plot them accordingly to the specified settings (setts)."""

    def build_tuple_dict(tuple_keys, d):
        _tuple = {}
        for tk in tuple_keys:
            _tuple[tk] = d[tk]
        return _tuple

    # loading all the experimental results
    loaded_exps = load_results(file)

    # for each plot setting...
    for s in setts:

        # keeping only experiments with the selected fixed values for the considered setting
        exps = []
        for e in loaded_exps:
            discard = False
            for fk, fv in s['fixed'].items():
                if e[fk] != fv:
                    discard = True
                    break
            if not discard:
                exps.append(e)

        # finding tuples of keys that will feature different figures
        figure_keys = []
        for ek in e.keys():
            if ek != s['x'] and ek != s['y'] and ek not in s['curves'] and ek not in s['ignore']:
                figure_keys.append(ek)

        # finding hashes (titles) of the different figures
        figure_hashes = []
        for e in exps:
            tuple_dict = build_tuple_dict(figure_keys, e)
            figure_hash = hash_keys_values(tuple_dict)
            if figure_hash not in figure_hashes:
                figure_hashes.append(figure_hash)

        # collecting data for each figure
        data_to_plot = {}
        for figure_hash in figure_hashes:
            data_to_plot[figure_hash] = collections.OrderedDict({})

            for e in exps:
                tuple_dict = build_tuple_dict(figure_keys, e)
                e_figure_hash = hash_keys_values(tuple_dict)

                if e_figure_hash == figure_hash:
                    curve = e[s['curves'][0]]
                    for ii in range(1, len(s['curves'])):
                        if type(e[s['curves'][ii]]) is not bool:
                            curve += "_" + e[s['curves'][ii]]
                        else:
                            if e[s['curves'][ii]]:
                                curve += "_" + s['curves'][ii]

                    if curve not in data_to_plot[figure_hash]:
                        data_to_plot[figure_hash][curve] = {'x': [], 'y': []}
                    data_to_plot[figure_hash][curve]['x'].append(e[s['x']])
                    data_to_plot[figure_hash][curve]['y'].append(e[s['y']])

        # plotting
        colors = ['red', 'magenta', 'blue', 'green', 'cyan', 'black', 'yellow']
        line_styles = ['solid', 'dashed', 'dotted']
        markers = ['*', 'o', 's']
        for f, data in data_to_plot.items():
            plt.figure(figsize=(10, 8))
            plt.xlabel(s['x'])
            plt.ylabel(s['y'])
            plt.title(f, fontdict={'fontsize': 8})
            k = 0
            kk = 0
            for leg, xy in data.items():
                x = xy['x']
                y = xy['y']
                #if leg != 'baseline_loose':  # warning: hack!
                plt.plot(x, y, label=leg, color=colors[k], linestyle=line_styles[kk], marker=markers[kk])
                kk += 1
                if kk >= len(line_styles):
                    kk = 0
                    k += 1
                    if k >= len(colors):
                        k = 0
            plt.legend()
            plt.show()


# ***********
# ENTRY POINT
# ***********
if not plot_only:

    # checking already done experiments
    ignore_when_comparing = ["avg_time", "reduction_factors", "region_sizes"]
    already_done = hash_results(load_results(log_file), ignore_keys=ignore_when_comparing)

    # running experiments
    first_exp_torch = True
    first_exp_log = True if len(already_done) == 0 else False
    for _banks in banks:
        for _region_type in region_types:
            for _w, _h in wh:
                for _batch_size in batch_sizes:

                    if _h > 512 and _batch_size > 1:  # warning: hack!
                        continue

                    for _in_features in in_features:
                        _input = torch.randn((_batch_size, _in_features, _h, _w), dtype=torch.float32,
                                             device=torch.device(device))
                        for _out_features in out_features:
                            for _region_count in region_counts:
                                for _kernel in kernels:
                                    for _method in methods:
                                        for _crop_type in crop_types:
                                            _opt = collections.OrderedDict({'device': device,
                                                                            'focus_areas': focus_areas,
                                                                            'repetitions': repetitions,
                                                                            'w': _w,
                                                                            'h': _h,
                                                                            'banks': _banks,
                                                                            'region_type': _region_type,
                                                                            'batch_size': _batch_size,
                                                                            'in_features': _in_features,
                                                                            'out_features': _out_features,
                                                                            'region_count': _region_count,
                                                                            'kernel_size': _kernel,
                                                                            'method': _method,
                                                                            'crop_type': _crop_type})
                                            if hash_keys_values(_opt, ignore_keys=ignore_when_comparing) \
                                                    in already_done.keys():
                                                continue
                                            else:
                                                _opt_with_results = run(_opt, _input,
                                                                        first_torch_call=first_exp_torch)
                                                log_experiments([_opt_with_results], append=not first_exp_log)
                                                first_exp_torch = False
                                                first_exp_log = False

if plot:

    # plotting
    collect_and_plot(log_file, plot_settings)
