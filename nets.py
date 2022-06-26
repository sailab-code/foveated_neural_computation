import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torchinfo import summary
from utils import get_act_function
import torchvision
from typing import Optional, List, Tuple
import torch.nn as nn
import time
import math
from fov_conv2d_cont import FovConv2dCont, LinearMiddleBiasOne
from fov_conv2d_reg import FovConv2dReg


class Foveate2d(nn.Module):
    def __init__(self,
                 regions: int,
                 regions_radius: List[int],
                 region_valid_idx: List,
                 region_type: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size_list: List[int],
                 stride_list: List[int],
                 padding_list: List[int],
                 w: int,
                 h: int,
                 outmost_global: bool,
                 device,
                 # transposed: bool,
                 # output_padding: List[int, ...],
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 dilation_list: List[int] = None,
                 debug=False,
                 compute_region_indexes=False
                 ):
        super(Foveate2d, self).__init__()

        self.module_list = nn.ModuleList()
        self.conv_regions = regions
        self.index_regions = len(regions_radius) - 1  # 0 radius was added at the beginning
        self.regions_radius = regions_radius
        self.region_type = region_type
        self.region_valid_idx = region_valid_idx
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = w
        self.h = h
        self.upsample = nn.Upsample(size=[h, w], mode='nearest')

        self.device = device
        self.zero_tensor = torch.tensor([0], device=self.device, requires_grad=False)
        self.compute_region_indexes = compute_region_indexes

        # if last region has radius -1, select the whole frame as attentended region
        self.outmost_global = outmost_global

        # a different  convolutional layer for each region
        for idx in range(self.conv_regions):
            if dilation_list is not None:
                self.module_list.append(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride_list[idx],
                              kernel_size=kernel_size_list[idx], padding=padding_list[idx], dilation=dilation_list[idx],
                              bias=True, padding_mode='replicate'), )
            else:
                self.module_list.append(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride_list[idx],
                              kernel_size=kernel_size_list[idx], padding=padding_list[idx],
                              bias=True, padding_mode='replicate'), )

        if debug:  # Notice: does not work anymoer in the case of 1 conv region and multiple index regions

            with torch.no_grad():
                for i in range(len(self.module_list)):
                    if self.module_list[i].weight is not None:

                        self.module_list[i].weight *= 0.
                        mid = int(self.module_list[i].kernel_size[0]) // 2
                        self.module_list[i].weight[:, :, mid, mid] = 1.
                        if self.conv_regions > 1:
                            self.module_list[i].weight[:, :, mid, mid] -= float(i) / float(self.conv_regions)
                        self.module_list[i].weight /= self.module_list[i].weight.shape[1]

                        self.module_list[i].bias *= 0

    def compose_output(self, list_activations, foa_x, foa_y):
        self.b = foa_x.shape[0]
        return self.compose_regions(list_activations, foa_y, foa_x)  # TODO inverted x and y

    def compose_regions(self, list_activations, foa_x, foa_y):
        unique_cnn_flag = False
        # if last region is -1, hence we want to consider all the remaining of the frame
        if self.outmost_global:
            layer_output = list_activations[-1]  # the outmost region is taken as all the latest activation
            considered_list = list_activations[:-1]
            # region to consider for the output creation (the outmost one has been already considered)
            regions_to_consider = self.index_regions - 1

        else:
            layer_output = torch.zeros_like(list_activations[-1],
                                            device=list_activations[
                                                -1].device)  # the outmost region activation as all 0s
            considered_list = list_activations
            regions_to_consider = self.index_regions

        # Handling specific case of 1 CNN regions and multiple radii
        if len(list_activations) == 1:
            # the activation list to be considered is always composed by one CNN output
            considered_list = layer_output
            unique_cnn_flag = True

        f = self.out_channels

        if self.compute_region_indexes:
            if self.regions_radius[-1] == -1:
                # indexes for the outmost region when -1
                region_idx_global = torch.ones((self.b, f, self.h, self.w), device=list_activations[-1].device) * (
                        self.index_regions - 1)
            else:
                # indexes for the outmost region when we do not consider the whole frame
                region_idx_global = -torch.ones((self.b, f, self.h, self.w), device=list_activations[-1].device)

        # starting = time.time()

        z_index = torch.arange(start=0, end=self.b, device=list_activations[-1].device, dtype=torch.long).view(self.b,
                                                                                                               1, 1)
        f_index = torch.arange(start=0, end=f, device=list_activations[-1].device, dtype=torch.long).view(1, f, 1)

        # for idx, el in enumerate(considered_list):
        for idx in range(regions_to_consider):  #
            # handling case of 1 CNN layer
            if unique_cnn_flag:
                el = considered_list
            else:
                el = considered_list[idx]
            region_x, region_y = self.region_valid_idx[idx]
            new_region_x = (region_x + foa_x)
            new_region_y = (region_y + foa_y)
            inside = (new_region_x >= 0) * (new_region_x < self.h) * (new_region_y >= 0) * (new_region_y < self.w)

            ################ NEW IMPLEMENTATION #############
            first_dim = (new_region_x * self.w + new_region_y).unsqueeze(dim=1)

            second_dim = f_index * self.h * self.w + first_dim

            third_dim = z_index * f * self.h * self.w + second_dim

            region_idx_whole = torch.masked_select(third_dim, inside.view(inside.shape[0], 1, inside.shape[1]))

            layer_output.view(-1)[region_idx_whole] = el.view(-1)[region_idx_whole]

            if self.compute_region_indexes:

                region_idx_global.view(-1)[region_idx_whole] = idx

            #####################################################

        # end = time.time() - starting
        # print(f"For all the regions: {end} sec.")

        if self.compute_region_indexes:
            # => then select only first channel of features
            return layer_output, region_idx_global[:, 0]
            # return layer_output, region_idx_global
        else:
            return layer_output

    def forward(self, frame, foa_coordinates):

        outlist = []
        # forward for every type of kernel/region

        # import time
        # torch.cuda.synchronize()
        # starting = time.time()

        for conv_i in self.module_list:  # for every conv2d of the various regions,
            # process the whole frame with the call method
            out = conv_i(frame)
            # upsample in case of lower dim obtained - caused by bigger stride
            if out.shape[2:] != frame.shape[2:]:
                out = self.upsample(out)
            outlist.append(out)

        # torch.cuda.synchronize()
        # end = time.time() - starting
        # print(f"Convolutions for loop: {end} sec.")

        # call a method on the output of the foveated layer depending on the type of region wanted

        foa_x = foa_coordinates[:, 0, None]
        foa_y = foa_coordinates[:, 1, None]  # TODO check!!! can be removed and also the one on dataloader!

        return self.compose_output(outlist, foa_x, foa_y)


class NetFactory:
    @staticmethod
    def createNet(options, region_valid_idx=None, outmost_global=None):
        if options.wrapped_arch == "test":
            return BaseEncoder(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "fnn_reg":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return FNN_Reg(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "last_FL":
            return FNN_Back1(options)
        elif options.wrapped_arch == "first_FL":
            return FNN_Back2(options)
        elif options.wrapped_arch == "all_FL":
            return FNN_Back3(options)
        elif options.wrapped_arch == "first_FL_gaussian":
            return Back2_GaussianFirst(options)
        elif options.wrapped_arch == "last_FL_gaussian":
            return Back2_GaussianLast(options)
        elif options.wrapped_arch == "all_FL_gaussian":
            return Back3_NetGaussianAll(options)
        elif options.wrapped_arch == "first_FL_netmodulated":
            return Back2_NetModulatedFirst(options)
        elif options.wrapped_arch == "last_FL_netmodulated":
            return Back2_NetModulatedLast(options)
        elif options.wrapped_arch == "all_FL_netmodulated":
            return Back3_NetModulatedAll(options)
        elif options.wrapped_arch == "first_FL_netgenerated":
            return Back2_NetGeneratedFirst(options)
        elif options.wrapped_arch == "last_FL_netgenerated":
            return Back2_NetGeneratedLast(options)
        elif options.wrapped_arch == "all_FL_netgenerated":
            return Back3_NetGeneratedAll(options)

        elif options.wrapped_arch == "FNN_one_layerk_5_d_1":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNN(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_7_d_1":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_7_d_1(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_7_d_3":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_7_d_3(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_10_d_1":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_10_d_1(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_10_d_3":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_10_d_3(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_15_d_3":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_15_d_3(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_15_d_1":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_15_d_1(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_29_d_1":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_29_d_1(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layerk_30_d_1":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNNk_30_d_1(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "FNN_one_layer_custom":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return OneLayerFNN_custom(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "two_layer":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return SimpleEncoder(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "five":
            # az = SimpleEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return FiveLayerEncoder(options, region_valid_idx, outmost_global)
        elif options.wrapped_arch == "low_param":
            # az = LowerParamEncoder(options, region_valid_idx, outmost_global)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return LowerParamEncoder(options, region_valid_idx, outmost_global)

        elif options.wrapped_arch == "CNN_A":
            # az = CNN_A(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return CNN_A(options)
        elif options.wrapped_arch == "CNN_A112":
            # az = CNN_A(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return CNN_A112(options)
        elif options.wrapped_arch == "CNN_A100":
            # az = CNN_A(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1,  options.w, options.h))
            # exit()
            return CNN_A100(options)
        elif options.wrapped_arch == "CNN_RAM":
            # az = CNN_RAM(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, 1, options.w, options.h))
            # exit()
            return CNN_RAM(options)
        elif options.wrapped_arch == "FC64":
            # az = FC64(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return FC64(options)
        elif options.wrapped_arch == "FC256":
            # az = FC256(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return FC256(options)

        elif options.wrapped_arch == "vanilla":
            # az = VanillaCNN(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return VanillaCNN(options)
        elif options.wrapped_arch == "CNN_one_layer_64_global":
            # az = VanillaCNN(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return OneLayerCNN(options)
        elif options.wrapped_arch == "CNN_one_layer_custom":
            # az = VanillaCNN(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return OneLayerCNNCustom(options)
        elif options.wrapped_arch == "CNN_one_layer_maxpool":
            # az = VanillaCNN(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return OneLayerCNN_maxpool(options)

        elif options.wrapped_arch == "CNN_one_layer_regionwise":
            # az = VanillaCNN(options)
            # print(az)
            # summary(az.to("cuda:0"), input_size=(1, options.w, options.h))
            # exit()
            return OneLayerCNNRegionwise(options)


        else:
            raise AttributeError(f"Architecture {options['architecture']} unknown.")


class BaseEncoder(nn.Module):

    def __init__(self, options, region_valid_idx, outmost_global):
        super(BaseEncoder, self).__init__()
        self.config = options
        self.region_valid_idx = region_valid_idx
        self.outmost_global = outmost_global
        self.device = options.device
        self.in_channels = 1 if self.config.grayscale else 3
        self.activation = get_act_function(options.act)
        self.net_modulelist = nn.ModuleList()

        self._architecture()

    def _architecture(self):

        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             region_valid_idx=self.region_valid_idx,
                                             region_type=self.config.region_type,
                                             outmost_global=self.outmost_global,
                                             device=self.device,
                                             in_channels=self.in_channels, out_channels=1,
                                             w=self.config.w, h=self.config.h,
                                             kernel_size_list=[1 for i in range(self.config.regions)],
                                             stride_list=[1 for i in range(self.config.regions)],
                                             padding_list=[0 for i in range(self.config.regions)],
                                             debug=True,
                                             compute_region_indexes=True),
                                   )

    def forward(self, frame, foa):
        temp = frame
        for i, module in enumerate(self.net_modulelist):
            temp = module(temp, foa)
            if i != len(self.net_modulelist) - 1:  # do not put activation in last layer
                temp = self.activation(temp)  # TODO customize activation function

        return temp


class FNN_Reg(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(FovConv2dReg(self.in_channels, self.config.output_channels, self.config.kernel,
                                                region_type=self.config.region_type,
                                                method=self.config.reduction_method,
                                                region_sizes=self.config.region_sizes,
                                                reduction_factors=[1.0, self.config.reduction_factor],
                                                banks=self.config.banks,
                                                padding_mode="replicate")
                                   )


class VanillaCNN(nn.Module):
    def __init__(self, config, input_dim=1):
        super(VanillaCNN, self).__init__()
        self.in_channels = 1 if config.grayscale else 3

        self.conv1 = nn.Conv2d(self.in_channels, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = config.num_classes

        self.fc = nn.Linear(128, self.num_classes)

    def forward(self, x):
        # x 224 x 224
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 112 x 112
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 56 x 56
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  # 28 x 28
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)  # 1 x 1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FNN_Back1(nn.Module):
    """
    Last layer is a FL, all others are conv2d
    """

    def __init__(self, options):
        super(FNN_Back1, self).__init__()
        self.config = options
        self.device = options.device
        self.in_channels = 1 if self.config.grayscale else 3

        self.relative_scaling_factor = 28. / 224.

        self.conv1 = nn.Conv2d(self.in_channels, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fl = FovConv2dReg(64, 128, 3,
                               region_type=self.config.region_type,
                               method=self.config.reduction_method,
                               region_sizes=(11, -1),
                               reduction_factors=(1.0, 0.25),
                               banks=self.config.banks,
                               padding_mode="zeros")

    def forward(self, x, foa):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 112 x 112
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 56 x 56
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  # 28 x 28
        # rescale foa
        new_foa = foa * self.relative_scaling_factor
        new_foa = torch.floor(new_foa)
        x, ind = self.fl(x, new_foa, compute_region_indices=True)
        x = F.relu(x)
        return x, ind


class FNN_Back2(VanillaCNN):
    """
    First layer is a FL, all subsequent are conv2d
    """

    def __init__(self, config):
        super(FNN_Back2, self).__init__(config)
        self.fl = FovConv2dReg(self.in_channels, 16, 5,
                               region_type=config.region_type,
                               method=config.reduction_method,
                               region_sizes=(51, 101, -1),
                               reduction_factors=(1.0, 0.5, 0.25),
                               banks=config.banks,
                               padding_mode="zeros")

    def forward(self, x, foa):
        x = F.relu(self.fl(x, foa, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 112 x 112
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 56 x 56
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  # 28 x 28
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)  # 1 x 1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Back2_GaussianFirst(FNN_Back2):
    """
    First layer is FL Gaussian Modulated, all subsequent are conv2d
    """

    def __init__(self, options):
        super(Back2_GaussianFirst, self).__init__(options)
        self.fl = FovConv2dCont(self.in_channels, 16, 5, kernel_type='gaussian_modulated',
                                gaussian_kernel_size=7, sigma_min=0.01, sigma_max=10.0, sigma_function='exponential')


class Back2_NetModulatedFirst(Back2_GaussianFirst):
    """
    First layer is FL Net-Modulated, all subsequent are conv2d
    """

    def __init__(self, options):
        super(Back2_NetModulatedFirst, self).__init__(options)
        self.fl = FovConv2dCont(self.in_channels, 16, 5, kernel_type='net_modulated',
                                kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.Tanh(),
                                                               LinearMiddleBiasOne(10, 7 * 7)))


class Back2_NetGeneratedFirst(Back2_GaussianFirst):
    """
        First layer is FL Net-Generated, all subsequent are conv2d
        """

    def __init__(self, options):
        super(Back2_NetGeneratedFirst, self).__init__(options)
        out_channels = 16
        in_channels = self.in_channels
        kernel_size = 5
        self.fl = FovConv2dCont(in_channels, out_channels, kernel_size, kernel_type='net_generated',
                                padding_mode='zeros',
                                kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=True), torch.nn.Tanh(),
                                                               torch.nn.Linear(10, out_channels * in_channels * (
                                                                       kernel_size ** 2))))


class Back2_GaussianLast(VanillaCNN):
    """
    Only last layer is FL gaussian-modulated; gaussian kernel reduced to 5
    """

    def __init__(self, options):
        super(Back2_GaussianLast, self).__init__(options)
        self.conv4 = FovConv2dCont(64, 128, 3, kernel_type='gaussian_modulated',
                                   gaussian_kernel_size=5, sigma_min=0.01, sigma_max=10.0, sigma_function='exponential')
        self.relative_scaling_factor = 28. / 224.

    def forward(self, x, foa):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 112 x 112
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 56 x 56
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  # 28 x 28
        # rescale foa
        new_foa = foa * self.relative_scaling_factor
        new_foa = torch.floor(new_foa)
        x = F.relu(self.conv4(x, new_foa))
        x = self.avgpool(x)  # 1 x 1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Back2_NetModulatedLast(Back2_GaussianLast):
    """
        Only last layer is FL net-modulated;
        """

    def __init__(self, options):
        super(Back2_NetModulatedLast, self).__init__(options)
        out_channels = 128
        in_channels = 64
        kernel_size = 3
        self.conv4 = FovConv2dCont(in_channels, out_channels, kernel_size, kernel_type='net_modulated',
                                   kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.Tanh(),
                                                                  LinearMiddleBiasOne(10, 5 * 5)))


class Back2_NetGeneratedLast(Back2_GaussianLast):
    """
        Only last layer is FL net-generated;
        """

    def __init__(self, options):
        super(Back2_NetGeneratedLast, self).__init__(options)
        out_channels = 128
        in_channels = 64
        kernel_size = 3
        self.conv4 = FovConv2dCont(in_channels, out_channels, kernel_size, kernel_type='net_generated',
                                   padding_mode='zeros',
                                   kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=True), torch.nn.Tanh(),
                                                                  torch.nn.Linear(10, out_channels * in_channels * (
                                                                          kernel_size ** 2))))


class FNN_Back3(nn.Module):
    """
    All FL layers
    """

    def __init__(self, options):
        super(FNN_Back3, self).__init__()
        self.config = options
        self.device = options.device
        self.in_channels = 1 if self.config.grayscale else 3

        self.relative_scaling_factor2 = 112. / 224.
        self.relative_scaling_factor3 = 56. / 224.
        self.relative_scaling_factor4 = 28. / 224.

        # input 224
        self.fl1 = FovConv2dReg(self.in_channels, 16, 5,
                                region_type=self.config.region_type,
                                method=self.config.reduction_method,
                                region_sizes=(51, 101, -1),
                                reduction_factors=(1.0, 0.5, 0.25),
                                banks=self.config.banks,
                                padding_mode="zeros")

        # input 112
        self.fl2 = FovConv2dReg(16, 32, 5,
                                region_type=self.config.region_type,
                                method=self.config.reduction_method,
                                region_sizes=(25, 51, -1),
                                reduction_factors=(1.0, 0.5, 0.25),
                                banks=self.config.banks,
                                padding_mode="zeros")

        # input 56
        self.fl3 = FovConv2dReg(32, 64, 3,
                                region_type=self.config.region_type,
                                method=self.config.reduction_method,
                                region_sizes=(25, -1),
                                reduction_factors=(1.0, 0.25),
                                banks=self.config.banks,
                                padding_mode="zeros")
        # input 28
        self.fl = FovConv2dReg(64, 128, 3,
                               region_type=self.config.region_type,
                               method=self.config.reduction_method,
                               region_sizes=(11, -1),
                               reduction_factors=(1.0, 0.25),
                               banks=self.config.banks,
                               padding_mode="zeros")

    def forward(self, x, foa):
        x = F.relu(self.fl1(x, foa, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 112 x 112
        # rescale foa
        new_foa2 = foa * self.relative_scaling_factor2
        new_foa2 = torch.floor(new_foa2)
        x = F.relu(self.fl2(x, new_foa2, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 56 x 56
        # rescale foa
        new_foa3 = foa * self.relative_scaling_factor3
        new_foa3 = torch.floor(new_foa3)
        x = F.relu(self.fl3(x, new_foa3, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 28 x 28
        # rescale foa
        new_foa4 = foa * self.relative_scaling_factor4
        new_foa4 = torch.floor(new_foa4)
        x, ind = self.fl(x, new_foa4, compute_region_indices=True)
        x = F.relu(x)
        return x, ind


class Back3_NetGaussianAll(FNN_Back3):
    """
    All FL layers gaussian-modulated
    """

    def __init__(self, options):
        super(Back3_NetGaussianAll, self).__init__(options)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = options.num_classes

        self.fc = nn.Linear(128, self.num_classes)
        # input 224
        self.fl1 = FovConv2dCont(self.in_channels, 16, 5, kernel_type='gaussian_modulated',
                                 gaussian_kernel_size=7, sigma_min=0.01, sigma_max=10.0, sigma_function='exponential')

        # input 112
        self.fl2 = FovConv2dCont(16, 32, 5, kernel_type='gaussian_modulated',
                                 gaussian_kernel_size=7, sigma_min=0.01, sigma_max=10.0, sigma_function='exponential')

        # input 56
        self.fl3 = FovConv2dCont(32, 64, 3, kernel_type='gaussian_modulated',
                                 gaussian_kernel_size=5, sigma_min=0.01, sigma_max=10.0, sigma_function='exponential')
        # input 28
        self.fl = FovConv2dCont(64, 128, 3, kernel_type='gaussian_modulated',
                                gaussian_kernel_size=3, sigma_min=0.01, sigma_max=10.0, sigma_function='exponential')

    def forward(self, x, foa):
        x = F.relu(self.fl1(x, foa, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 112 x 112
        # rescale foa
        new_foa2 = foa * self.relative_scaling_factor2
        new_foa2 = torch.floor(new_foa2)
        x = F.relu(self.fl2(x, new_foa2, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 56 x 56
        # rescale foa
        new_foa3 = foa * self.relative_scaling_factor3
        new_foa3 = torch.floor(new_foa3)
        x = F.relu(self.fl3(x, new_foa3, compute_region_indices=False))
        x = F.max_pool2d(x, 2, 2)  # 28 x 28
        # rescale foa
        new_foa4 = foa * self.relative_scaling_factor4
        new_foa4 = torch.floor(new_foa4)
        x = F.relu(self.fl(x, new_foa4, compute_region_indices=False))
        x = self.avgpool(x)  # 1 x 1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Back3_NetModulatedAll(Back3_NetGaussianAll):
    """
    All FL layers net-modulated
    """

    def __init__(self, options):
        super(Back3_NetModulatedAll, self).__init__(options)

        # input 224
        self.fl1 = FovConv2dCont(self.in_channels, 16, 5, kernel_type='net_modulated',
                                 kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.Tanh(),
                                                                LinearMiddleBiasOne(10, 7 * 7)))

        # input 112
        self.fl2 = FovConv2dCont(16, 32, 5, kernel_type='net_modulated',
                                 kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.Tanh(),
                                                                LinearMiddleBiasOne(10, 7 * 7)))

        # input 56
        self.fl3 = FovConv2dCont(32, 64, 3, kernel_type='net_modulated',
                                 kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.Tanh(),
                                                                LinearMiddleBiasOne(10, 5 * 5)))
        # input 28
        self.fl = FovConv2dCont(64, 128, 3, kernel_type='net_modulated',
                                kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False), torch.nn.Tanh(),
                                                               LinearMiddleBiasOne(10, 3 * 3)))


class Back3_NetGeneratedAll(Back3_NetGaussianAll):
    """
    All FL layers net-generated
    """

    def __init__(self, options):
        super(Back3_NetGeneratedAll, self).__init__(options)

        # input 224
        self.fl1 = FovConv2dCont(self.in_channels, 16, 5, kernel_type='net_generated',
                                 padding_mode='zeros',
                                 kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=True), torch.nn.Tanh(),
                                                                torch.nn.Linear(10, 16 * self.in_channels * (5 ** 2))))

        # input 112
        self.fl2 = FovConv2dCont(16, 32, 5, kernel_type='net_generated',
                                 padding_mode='zeros',
                                 kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=True), torch.nn.Tanh(),
                                                                torch.nn.Linear(10, 32 * 16 * (5 ** 2))))

        # input 56
        self.fl3 = FovConv2dCont(32, 64, 3, kernel_type='net_generated',
                                 padding_mode='zeros',
                                 kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=True), torch.nn.Tanh(),
                                                                torch.nn.Linear(10, 64 * 32 * (3 ** 2))))
        # input 28
        self.fl = FovConv2dCont(64, 128, 3, kernel_type='net_generated',
                                padding_mode='zeros',
                                kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=True), torch.nn.Tanh(),
                                                               torch.nn.Linear(10, 128 * 64 * (3 ** 2))))


class OneLayerFNN(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=1, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[5, 5],
                                             stride_list=[1, 1],
                                             padding_list=[5 // 2, 5 // 2],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_7_d_1(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[7, 7],
                                             stride_list=[1, 1],
                                             padding_list=[7 // 2, 7 // 2],
                                             dilation_list=[1, 1],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_7_d_3(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[7, 7],
                                             stride_list=[1, 1],
                                             padding_list=[7 // 2, 7 // 2],
                                             dilation_list=[1, 3],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_10_d_1(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[10, 10],
                                             stride_list=[1, 1],
                                             padding_list=[10 // 2, 10 // 2],
                                             dilation_list=[1, 1],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_10_d_3(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[10, 10],
                                             stride_list=[1, 1],
                                             padding_list=[10 // 2, 10 // 2],
                                             dilation_list=[1, 3],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_15_d_3(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[15, 15],
                                             stride_list=[1, 1],
                                             padding_list=[15 // 2, 15 // 2],
                                             dilation_list=[1, 3],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_15_d_1(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[15, 15],
                                             stride_list=[1, 1],
                                             padding_list=[15 // 2, 15 // 2],
                                             dilation_list=[1, 1],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_29_d_1(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[29, 29],
                                             stride_list=[1, 1],
                                             padding_list=[29 // 2, 29 // 2],
                                             dilation_list=[1, 1, ],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNNk_30_d_1(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[30, 30, 30],
                                             stride_list=[1, 1, 1],
                                             padding_list=[30 // 2, 30 // 2, 30 // 2],
                                             dilation_list=[1, 1, 1],
                                             compute_region_indexes=True),
                                   )


class OneLayerFNN_custom(BaseEncoder):
    def _architecture(self):
        kernel = self.config.kernel
        dilation = self.config.dilation

        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=self.in_channels, out_channels=self.config.output_channels,
                                             region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[kernel, kernel],
                                             stride_list=[1, 1],
                                             padding_list=[1, 1],
                                             dilation_list=[1, dilation],
                                             compute_region_indexes=True),
                                   )


class SimpleEncoder(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             region_valid_idx=self.region_valid_idx,
                                             region_type=self.config.region_type,
                                             outmost_global=self.outmost_global,
                                             device=self.device,
                                             in_channels=self.in_channels, out_channels=32,
                                             w=self.config.w, h=self.config.h,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 1, 1, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             ))
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=32, out_channels=64, region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 1, 1, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             compute_region_indexes=True),
                                   )


class FiveLayerEncoder(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             region_valid_idx=self.region_valid_idx,
                                             region_type=self.config.region_type,
                                             outmost_global=self.outmost_global,
                                             device=self.device,
                                             in_channels=self.in_channels, out_channels=16,
                                             w=self.config.w, h=self.config.h,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 1, 1, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             ))
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=16, out_channels=32, region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 2, 3, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             ),
                                   )
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=32, out_channels=64, region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 2, 3, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             ),
                                   )
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=64, out_channels=64, region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 3, 5, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             compute_region_indexes=True),
                                   )


class LowerParamEncoder(BaseEncoder):
    def _architecture(self):
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             region_valid_idx=self.region_valid_idx,
                                             region_type=self.config.region_type,
                                             outmost_global=self.outmost_global,
                                             device=self.device,
                                             in_channels=self.in_channels, out_channels=32,
                                             w=self.config.w, h=self.config.h,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 3, 5, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             ))
        self.net_modulelist.append(Foveate2d(regions=self.config.regions, regions_radius=self.config.regions_radius,
                                             in_channels=32, out_channels=64, region_type=self.config.region_type,
                                             region_valid_idx=self.region_valid_idx,
                                             outmost_global=self.outmost_global,
                                             w=self.config.w, h=self.config.h,
                                             device=self.device,
                                             kernel_size_list=[3, 3, 5, 5, 5],
                                             stride_list=[1, 3, 5, 1, 1],
                                             padding_list=[1, 1, 2, 2, 2],
                                             compute_region_indexes=True),
                                   )


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes: typing.Iterable[int], out_dim, activation_function=nn.Sigmoid(),
                 activation_out=None):
        super(MLP, self).__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        i_h_sizes = [input_dim] + hidden_sizes  # add input dim to the iterable
        self.mlp = nn.Sequential()
        for idx in range(len(i_h_sizes) - 1):
            self.mlp.add_module(f"layer_{idx}",
                                nn.Linear(in_features=i_h_sizes[idx], out_features=i_h_sizes[idx + 1]))
            self.mlp.add_module(f"act_{idx}", activation_function)
        self.mlp.add_module("out_layer", nn.Linear(i_h_sizes[-1], out_dim))
        if activation_out is not None:
            self.mlp.add_module("out_layer_activation", activation_out)

    def init(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


class CNN_A(nn.Module):
    def __init__(self, config, input_dim=1):
        super(CNN_A, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(input_dim, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        #### 1 -> 10
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        return x


class CNN_A112(CNN_A):
    def __init__(self, config, input_dim=1):
        super(CNN_A112, self).__init__(config)

        self.fc1 = nn.Linear(12 * 12 * 128, 1024)


class CNN_A100(CNN_A):
    def __init__(self, config, input_dim=1):
        super(CNN_A100, self).__init__(config)

        self.fc1 = nn.Linear(10 * 10 * 128, 1024)


class CNN_RAM(nn.Module):
    def __init__(self, config, input_dim=1):
        super(CNN_RAM, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=10, stride=5)
        self.fc1 = nn.Linear(968, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class OneLayerCNN(nn.Module):
    def __init__(self, config, input_dim=1):
        super(OneLayerCNN, self).__init__()
        self.in_channels = 1 if config.grayscale else 3
        self.conv1 = nn.Conv2d(self.in_channels, 64, 15, stride=1, padding_mode='replicate', padding=15 // 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = config.num_classes

        self.fc = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class OneLayerCNNCustom(nn.Module):
    def __init__(self, config, input_dim=1):
        super(OneLayerCNNCustom, self).__init__()
        out_channels = config.output_channels
        self.in_channels = 1 if config.grayscale else 3
        kernel = config.kernel
        self.num_classes = config.num_classes

        # self.conv1 = nn.Conv2d(1, out_channels, kernel, stride=1, padding_mode='replicate',)
        self.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel, stride=1, padding_mode='replicate',
                               padding=int(kernel) // 2)
        # self.conv1 = nn.Conv2d(1, out_channels, kernel, stride=1, padding_mode='replicate', padding="same")
        if config.aggregation_type == "mean":
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif config.aggregation_type == "max":
            self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        # self.fc = nn.Linear(out_channels, 10)
        self.fc = MLP(input_dim=out_channels, hidden_sizes=[128, ],
                      out_dim=self.num_classes,
                      activation_function=nn.ReLU(),
                      activation_out=None)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class OneLayerCNNRegionwise(nn.Module):
    def __init__(self, config, input_dim=1):
        super(OneLayerCNNRegionwise, self).__init__()
        self.in_channels = 1 if config.grayscale else 3
        out_channels = config.output_channels
        kernel = config.kernel
        self.num_classes = config.num_classes

        # self.conv1 = nn.Conv2d(1, out_channels, kernel, stride=1, padding_mode='replicate',)
        self.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel, stride=1, padding_mode='replicate',
                               padding=int(kernel) // 2)
        # self.conv1 = nn.Conv2d(1, out_channels, kernel, stride=1, padding_mode='replicate', padding="same")

        inner_radius = 16

        self.region_valid_idx = self.define_regions(inner_radius, config)

        inside_region = self.obtain_region(config)

        # self.fc = nn.Linear(out_channels, 10)
        self.fc = MLP(input_dim=out_channels, hidden_sizes=[128, ],
                      out_dim=self.num_classes,
                      activation_function=nn.ReLU(),
                      activation_out=None)

    def obtain_region(self, config):
        region_x, region_y = self.region_valid_idx[0]
        inside = (region_x >= 0) * (region_x < config.h) * (region_y >= 0) * (region_y < config.w)
        return inside

    def define_regions(self, inner_radius, config):
        # dictionary containing region valid indexes
        region_valid_idx = []
        regions_radius = [inner_radius, -1]
        outmost_global = True
        regions_radius.insert(0, 0)  # insert new radius -1 in first position

        if outmost_global:
            explored_regions = regions_radius[:-2]
        else:
            explored_regions = regions_radius[:-1]

        for idx, el in enumerate(explored_regions):  # do not explore the last element
            radius1 = el
            radius2 = regions_radius[idx + 1]
            region_x, region_y = torch.meshgrid(
                torch.arange(start=-radius2 + 1, end=radius2 + 1, dtype=torch.long, device=config.device),
                torch.arange(start=-radius2 + 1, end=radius2 + 1, dtype=torch.long, device=config.device),
                # indexing="ij"
            )
            valid = self.condition_crown(region_x, region_y, radius1, radius2)
            region_x = region_x[valid].contiguous()
            region_y = region_y[valid].contiguous()
            region_valid_idx.append([region_x, region_y])

        return region_valid_idx

    def condition_crown(self, region_x, region_y, radius1, radius2):
        """
        Select the indexes respecting the circular crown region
        :param region_x:
        :param region_y:
        :param radius1: internal radius
        :param radius2: external radius
        :return:
        """

        valid_x = torch.logical_and(torch.abs(region_x) < radius2, torch.abs(region_x) >= radius1)
        valid_y = torch.logical_and(torch.abs(region_y) < radius2, torch.abs(region_y) >= radius1)

        valid = torch.logical_or(valid_x, valid_y)

        return valid

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class OneLayerCNN_maxpool(nn.Module):
    def __init__(self, config):
        super(OneLayerCNN_maxpool, self).__init__()
        self.config = config
        self.in_channels = 1 if config.grayscale else 3
        out_channels = config.output_channels
        kernel = config.kernel
        self.num_classes = config.num_classes

        pad = int(kernel) // 2
        self.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel_size=kernel, padding_mode='replicate',
                               padding=pad)
        # o = 256 - kernel + 1
        res_input = config.w
        o = res_input + pad * 2 - kernel + 1
        fc_inp_dim = math.floor((o - 10) / 10 + 1)
        self.fc = MLP(input_dim=fc_inp_dim * fc_inp_dim * out_channels, hidden_sizes=[128, ],
                      out_dim=self.num_classes,
                      activation_function=nn.ReLU(),
                      activation_out=None)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 10, 10)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FC64(nn.Module):
    def __init__(self, config, input_dim=1):
        super(FC64, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.h * config.w, 64)  # if input is 60x60
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class FC256(nn.Module):
    def __init__(self, config, input_dim=1):
        super(FC256, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.h * config.w, 256)  # if input is 60x60
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
