import torch
from torchvision import transforms
from fov_conv2d_cont import FovConv2dCont, LinearMiddleBiasOne
from fov_conv2d_reg import FovConv2dReg
from torch.nn import Conv2d
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import requests
import math
from fvcore.nn.flop_count import flop_count
import pandas as pd
from stream_attention_nets import create_networks
from fvcore.nn import FlopCountAnalysis
from torch.profiler import (
    kineto_available, profile, record_function, supported_activities,
    DeviceType, ProfilerAction, ProfilerActivity
)
from torch.autograd.profiler import profile as _profile
import time
import gc
import numpy as np

# dog picture
image_url = 'https://i.guim.co.uk/img/media/c63dddb413272fb6e8c308f0298c6333b3e2084f/' + \
           '0_139_4256_2554/master/4256.jpg?width=620&quality=45&auto=format&' + \
           'fit=max&dpr=2&s=0a5a0ebeb96ddfeeee0baf78ea0489b6'

# other customizable options
# h = 1000
# w = 1000



def load_img_foa(in_channels=1):
    # get an RGB image
    if in_channels == 1:
        image = Image.open(requests.get(image_url, stream=True).raw).resize((w, h), Image.ANTIALIAS).convert('L')
    elif in_channels == 3:
        image = Image.open(requests.get(image_url, stream=True).raw).resize((w, h), Image.ANTIALIAS).convert('RGB')
    image = transforms.ToTensor()(image).unsqueeze_(0)

    # set the FOA coordinates
    foa_xy = torch.tensor([[h // 2, w // 2]], dtype=torch.long)

    return image, foa_xy


# crating a Foveated Layer (multiple cases)


arch_type = ["cnn", 'fov_reg_last', "fov_gaussian_last",
             'fov_net_last', "fov_reg_all",
             'fov_gaussian_all', 'fov_net_all', 'fov_reg_all_2']
# arch_type = ['cnn']

#
# arch_type = ['fov_gaussian_all',
#              ]

df_dict = {"model": [],
           "avg_time": [],
           "std_time": [],
           "GFLOPS": []}

net_output_classes = 20
device = "cuda"
h = 200
w = 200
in_channels = 1
repetitions = 100

print(f"Number of iterations : {repetitions}; Device: {device}; Frame width: {w}")

for arch in arch_type:
    print(f"-- Compute timings for architecture: {arch}...")

    timings = []
    first_torch_call = True
    for rep in range(repetitions):
        image, foa_xy = load_img_foa()
        # first creation
        feature_extractor, head_classifier = create_networks(arch, net_output_classes, device)

        # moving to the right device
        image = image.to(device)
        foa_xy = foa_xy.to(device)

        if first_torch_call:

            # warmup run
            output_data, _ = feature_extractor(image, foa_xy)
            c = output_data[0, 0, 0, 0].item() + 3  # dummy operation

            del feature_extractor
            del head_classifier
            del output_data
            del image
            del foa_xy
            if device[0:4] == "cuda":
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            feature_extractor = None
            head_classifier = None

            # recreate the model and inputs

            feature_extractor, head_classifier = create_networks(arch, net_output_classes, device)
            image, foa_xy = load_img_foa()
            # moving to the right device
            image = image.to(device)
            foa_xy = foa_xy.to(device)

            first_torch_call = False

        ############# real time computation  #######################
        t = time.time()
        output_data, _ = feature_extractor(image, foa_xy)
        c = output_data[0, 0, 0, 0].item() + 3  # dummy operation
        elapsed = time.time() - t
        ############################################################
        timings.append(elapsed)

        # clean all
        del feature_extractor
        del head_classifier
        del output_data
        del image
        del foa_xy
        if device[0:4] == "cuda":
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        feature_extractor = None
        head_classifier = None

    df_dict["avg_time"].append(np.mean(timings))
    df_dict["std_time"].append(np.std(timings))

    image, foa_xy = load_img_foa()
    image = image.to(device)
    foa_xy = foa_xy.to(device)
    feature_extractor, head_classifier = create_networks(arch, net_output_classes, device)
    with _profile(record_shapes=True, with_flops=True, use_kineto=kineto_available()) as prof:
        feature_extractor(image, foa_xy)
    profiler_output = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=1000)
    FLOPScounter = 0
    for i in prof.function_events:
        FLOPScounter += i.flops
    # print(f"Model - {arch}; \t GFLOPS: ", FLOPScounter / 1000000000.)
    # print(prof.total_average())
    # print(profiler_output)
    # exit()
    df_dict["model"].append(arch)
    # append reduction factor
    # append GFLOPS
    df_dict["GFLOPS"].append(FLOPScounter / 1000000000.)
    # exit()

df = pd.DataFrame(df_dict).drop_duplicates()
print(df)
df.to_csv('pane_timings_standard.csv')
