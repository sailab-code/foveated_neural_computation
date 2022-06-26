import torch
import torch.nn as nn
from nets import MLP
from nets import NetFactory
from utils import get_act_function_module


def FoveaFactory(options):
    if options.aggregation_arch == "simple":
        return OfflineFoveateNet(options)
    elif options.aggregation_arch == "plain":
        return OfflinePlainFoveated(options)
    elif options.aggregation_arch == "task3":
        return Task3Region(options)
    elif options.aggregation_arch == "none":
        return NetFactory.createNet(options)
    else:
        raise AttributeError(f"Architecture {options.aggregation_arch} unknown.")


class FoveateVanilla(nn.Module):
    def __init__(self, config):
        super(FoveateVanilla, self).__init__()
        self.config = config
        self.index_regions = len(config.region_sizes)
        self.net = NetFactory.createNet(config, region_valid_idx=None,
                                        outmost_global=None)  # pass the regions centered on center frame

    def forward(self, frame, foa):
        """
        if self.config.FLOPS_count:
            from flopth import flopth
            from fvcore.nn.flop_count import flop_count

            gflop_dict, _ = flop_count(self.net, (frame, foa,))
            gflops = sum(gflop_dict.values())
            print(_)
            print(gflop_dict)
            print(f'{gflops=}')
            print("DIRETTPO")
            # print("Other method:###########\n")
            #
            # sum_flops = flopth(self.wrapped_model, in_size=([list(frames[0].size()), list(foa[0].size())]))
            # print(sum_flops)
            exit()

        """
        return self.net(frame, foa)


class FoveateNet(nn.Module):

    def __init__(self, config):
        super(FoveateNet, self).__init__()

        # some checks on input dims
        self.conv_regions = config.regions
        self.index_regions = len(config.regions_radius)
        self.regions_radius = config.regions_radius
        self.region_type = config.region_type
        h = config.h
        w = config.w

        for idx, rad in enumerate(self.regions_radius):
            assert rad < min(h, w) // 2, "A region radius exceed frame size"
            if idx != len(self.regions_radius) - 1:
                assert rad != -1, "Radius '-1' only in last position"

                # if last region has radius -1, select the whole frame as attentended region

        self.outmost_global = self.regions_radius[-1] == -1

        # set into config

        # Initialization

        region_valid_idx = self.define_regions(config)
        self.net = NetFactory.createNet(config, region_valid_idx,
                                        self.outmost_global)  # pass the regions centered on center frame

    def forward(self, frame, foa):
        return self.net(frame, foa)

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

    def define_regions(self, config, condition=None):
        # dictionary containing region valid indexes
        region_valid_idx = []
        regions_radius = config.regions_radius
        regions_radius.insert(0, 0)  # insert new radius -1 in first position

        if self.outmost_global:
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

    def save(self, folder=None):
        pass


class FoveateNetCircular(FoveateNet):
    def __init__(self, config):
        super(FoveateNetCircular, self).__init__(config)

    def condition_crown(self, region_x, region_y, radius1, radius2):
        """
        Select the indexes respecting the circular crown region
        :param region_x:
        :param region_y:
        :param radius1: internal radius
        :param radius2: external radius
        :return:
        """
        valid = (torch.sqrt(region_x ** 2 + region_y ** 2) >= radius1) * (
                torch.sqrt(region_x ** 2 + region_y ** 2) < radius2)
        return valid



class OfflineFoveateNet(nn.Module):
    def __init__(self, config):
        super(OfflineFoveateNet, self).__init__()
        self.config = config
        self.fixations = config.fixations
        self.output_channels = config.output_channels
        self.num_classes = config.num_classes
        self.head_act = get_act_function_module(config.head_act)

        if config.new_implementation_fovea:
            self.net = FoveateVanilla(config)
        else:
            if self.config.region_type == "square" or self.config.region_type == "box":
                self.net = FoveateNet(config)
            elif self.config.region_type == "circular":
                self.net = FoveateNetCircular(config)
            else:
                raise NotImplementedError

        self.aggregators()

    def aggregators(self):

        self.pooling_layer = nn.Sequential(nn.AvgPool2d((self.config.h, self.config.w)),
                                           nn.Flatten())

        self.MLPHead = MLP(input_dim=self.output_channels, hidden_sizes=[32, 32], out_dim=self.num_classes,
                           activation_function=nn.Sigmoid(),
                           activation_out=None).to(self.config.device)

    def forward(self, frame, foa):

        # external loop on fixations - in this way, internally the foveatelayer is processing a single fixation,
        # across all the foveatelayers, but for the whole batch!!!
        output_fixation_seq = []  # list of output of foveatenet for every fixation

        for fix in range(self.fixations):
            # keep parallelizing over the batches, but do one fixation at the time

            foveate_out, region_idx = self.net(frame, foa[:, fix])
            spatial_average = self.pooling_layer(foveate_out)
            output_fixation_seq.append(spatial_average)

        stacked_seq_output = torch.stack(output_fixation_seq, dim=0)
        temporal_average = torch.mean(stacked_seq_output, dim=0)

        return self.MLPHead(temporal_average)


class OfflinePlainFoveated(OfflineFoveateNet):
    def __init__(self, config):
        super(OfflinePlainFoveated, self).__init__(config)

    def aggregators(self):

        self.MLPHead = MLP(input_dim=self.output_channels * self.net.index_regions, hidden_sizes=self.config.head_dim,
                           out_dim=self.num_classes,
                           activation_function=self.head_act,
                           activation_out=None).to(self.config.device)

    def forward(self, frame, foa):
        # external loop on fixations - in this way, internally the foveatelayer is processing a single fixation,
        # across all the foveatelayers, but for the whole batch!!!
        output_fixation_seq = []  # list of output of foveatenet for every fixation

        batch_size = frame.shape[0]

        for fix in range(self.fixations):
            # keep parallelizing over the batches, but do one fixation at the time

            foveate_out, region_idx = self.net(frame, foa[:, fix])

            region_out_seq = []
            for idx in range(self.net.index_regions):

                region = region_idx == idx
                batch_seq = []

                for b_idx in range(batch_size):  # batch_dim but dinamic (possible batches with different shapes)
                    region_out = torch.masked_select(foveate_out[b_idx], (region[b_idx]).unsqueeze(dim=0))
                    region_out = region_out.view(foveate_out.shape[1], -1)  # restore channel dims
                    if self.config.aggregation_type == "mean":
                        spatial_average = torch.mean(region_out, dim=-1)  # output is [1, c]
                    elif self.config.aggregation_type == "max":
                        spatial_average, _ = torch.max(region_out, dim=-1)  # output is [1, c]
                    else:
                        raise NotImplementedError

                    batch_seq.append(spatial_average)
                stacked_batch_seq = torch.stack(batch_seq, dim=0)  # [b, c]
                region_out_seq.append(stacked_batch_seq)

            stacked_regions_out = torch.stack(region_out_seq, dim=2)  # [b, c, #regions]
            output_fixation_seq.append(stacked_regions_out)

        stacked_seq_output = torch.stack(output_fixation_seq, dim=0)  # [#fixations, b, c, #regions, ]
        temporal_average = torch.mean(stacked_seq_output, dim=0)  # [b, c, #regions]

        # flatten channel and region dims
        temporal_average = temporal_average.view(batch_size, -1)

        return self.MLPHead(temporal_average)


class Task3Region(OfflineFoveateNet):
    def __init__(self, config):
        super(Task3Region, self).__init__(config)

    def aggregators(self):
        self.MLPHead = nn.Linear(self.output_channels * self.net.index_regions, self.num_classes).to(self.config.device)

    def forward(self, frame, foa):

        batch_size = frame.shape[0]

        foveate_out, region_idx = self.net(frame, foa)

        region_out_seq = []
        for idx in range(self.net.index_regions):

            region = region_idx == idx
            batch_seq = []

            for b_idx in range(batch_size):  # batch_dim but dinamic (possible batches with different shapes)
                region_out = torch.masked_select(foveate_out[b_idx], (region[b_idx]).unsqueeze(dim=0))
                region_out = region_out.view(foveate_out.shape[1], -1)  # restore channel dims
                if self.config.aggregation_type == "mean":
                    spatial_average = torch.mean(region_out, dim=-1)  # output is [1, c]
                elif self.config.aggregation_type == "max":
                    spatial_average, _ = torch.max(region_out, dim=-1)  # output is [1, c]
                else:
                    raise NotImplementedError

                batch_seq.append(spatial_average)
            stacked_batch_seq = torch.stack(batch_seq, dim=0)  # [b, c]
            region_out_seq.append(stacked_batch_seq)

        stacked_regions_out = torch.stack(region_out_seq, dim=2)  # [b, c, #regions]

        # flatten channel and region dims
        flatt = stacked_regions_out.view(batch_size, -1)

        return self.MLPHead(flatt)

