import torch
import math
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair


class FovConv2dReg(torch.nn.Module):
    """Foveated Convolutional Layer with a region-based dependency from the FOA coordinates, uniform spatial coverage,
    with a strong emphasis on speed/reduced-computations and control.

    This type of layer allows the user to specify a 'reduction factor' that tells how strongly the cost of convolution
    should be reduced when moving far away from the FOA. The image is divided into a fixed number of regions/areas
    around the FOA, and each of them has its own 'reduction factor'. The region that contains the FOA usually has the
    lowest reduction factor.

    Convolution is made cheaper by reducing the number of kernel components (spatially). In order to ensure that the
    reduced kernels still cover receptive inputs of similar sizes, the input must be downscaled, or stride must be used,
    or dilation should be exploited. These are the three methods implemented here.

    A nice side effect of this class is that it also allows the user to specify a custom model to be used to process
    the data in each region. Regions are downscaled accordingly to the 'reduction factor' and the custom model processes
    the downscaled crops.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 kernel_size=11,
                 dilation=1,
                 padding_mode="zeros",
                 region_type="circle",
                 region_sizes=(51, 101, 201, -1),
                 reduction_factors=(1.0, 0.5, 0.25, 0.1),
                 method="downscaling",
                 banks="independent",
                 crop_type="loose",
                 custom_model=None,
                 custom_model_padding=0,
                 bias=True):
        super(FovConv2dReg, self).__init__()

        # saving the main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self.region_type = region_type
        self.region_count = len(region_sizes) if not isinstance(region_sizes, int) else 1  # usually a list
        self.method = method
        self.reduction_factors = reduction_factors \
            if not (isinstance(reduction_factors, float) or isinstance(reduction_factors, int)) else [reduction_factors]
        self.max_region_size = None
        self.banks = banks
        self.crop_type = crop_type
        self.cropped_convs = crop_type is None or crop_type != "none"

        # regions: virtually, each region is a rectangular area (the user can only define a squared box, but then this
        # constraint is relaxed when introducing crops) used to mark a portion of the input image.
        # its coordinates are expressed in the reference frame defined by the FOA (initially assumed to be in (0,0)).
        # in practice, only some of the elements of the region are actually considered (consider the case of circles),
        # and their (x,y)-coordinates are generated and saved.
        self.region_sizes = []
        self.region_corners = []  # top-left corner (x,y) in the FOA ref. frame assumed to be in (0,0)
        self.region_xxs_yys = []  # actual (x,y)'s of the significant portion of the region

        # crops: a crop is similar to a region, even if it is actually a "padded" region.
        # crops are only rectangular, and their (x,y)-coordinates are the coordinates of all the elements inside
        # each rectangular crop (they are needed for indexing batched data).
        self.crop_sizes = []
        self.crop_corners = []  # top-left corner (x,y) in the FOA ref. frame assumed to be in (0,0)
        self.crop_xxs_yys = []

        # basic computational models of each region (crop)
        self.conv2ds = []

        # down-scaling factors of each image (crop)
        self.downscale_factors = []

        # device on which registered buffers are stored
        self.__buffers_device = None

        # preparing placeholders for offsets that will be created/used when indexing data for batch sizes > 1
        self.__bc_offsets = None
        self.__bc_offsets_crop = None
        self.__bf_offsets = None
        self.__bf_offsets_crop = None
        self.__br_offsets = None
        self.__offsets_status = None

        # preparing (squared) region sizes
        _region_sizes = list(region_sizes) \
            if not (isinstance(region_sizes, float) or isinstance(region_sizes, int)) else [region_sizes]

        # ensuring region sizes are of integer type and computing max region size
        _region_sizes[0] = int(_region_sizes[0])
        for i in range(0, self.region_count):
            _region_sizes[i] = int(_region_sizes[i])
        self.max_region_size = max(_region_sizes)

        # checking arguments
        assert region_type in ["box", "circle"], \
            "Unknown region type " + str(region_type) + " (it must be either box or circle)."
        assert crop_type in [None, "none", "loose", "tight"], \
            "Unknown crop type " + str(crop_type) + " (it must be one of: none, loose, tight)."
        assert method in ["downscaling", "dilation", "stride", "vanilla"], \
            "Unknown method " + str(method) + " (it must be one of: downscaling, dilation, stride, vanilla)."
        assert banks in ["independent", "shared"], \
            "Unknown type of filter banks " + str(banks) + " (it must be one of: independent, shared)."
        assert len(self.reduction_factors) == self.region_count, \
            "The number of regions and the number of reduction factors must be the same " \
            "(" + str(len(_region_sizes)) + "<>" + str(len(reduction_factors)) + ")."
        assert dilation >= 1, "Dilation factor must be greater than 0."
        assert type(padding_mode) is str, "Padding mode argument must be a string (usually: 'zeros')."
        assert _region_sizes[0] % 2 == 1, \
            "Each region size must be odd (found: " + str(_region_sizes) + ")."
        for i in range(0, self.region_count):
            assert _region_sizes[i] % 2 == 1, \
                "Each region size must be odd (found: " + str(_region_sizes) + ")."
            assert _region_sizes[i] == int(_region_sizes[i]), \
                "Each region size must be integer (found: " + str(_region_sizes) + ")."
            if i > 0:
                assert i == self.region_count - 1 and _region_sizes[i] == -1 or \
                       _region_sizes[i] > _region_sizes[i - 1], \
                    "Region sizes must be ordered in ascending order, " + \
                    " the last one can be -1 (found: " + str(_region_sizes) + ")."
                assert self.reduction_factors[i] <= self.reduction_factors[i - 1], \
                    "Reduction factors must in descending order (found: " + str(self.reduction_factors) + ")."
        assert custom_model is None or custom_model_padding is not None, \
            "You provided a custom computational model, so you must also provide its associated padding."
        assert custom_model is None or method == "downscaling", \
            "The only method that supports custom models is 'downscaling'."

        # let's define the class of convolutional layers that will be used (only) in the case of shared banks
        class InterpolatedKernelConv2d(torch.nn.Conv2d):

            def __init__(self, base_conv2d, kernel_size,
                         dilation=1, stride=1, padding=0, padding_mode='zeros', groups=1, bias=True):
                super(InterpolatedKernelConv2d, self).__init__(base_conv2d.in_channels, base_conv2d.out_channels,
                                                               kernel_size, stride, padding,
                                                               dilation, groups, bias, padding_mode)

                # saving base kernel and bias
                self.base_weight = base_conv2d.weight
                self.base_bias = base_conv2d.bias
                self.base_kernel_size = base_conv2d.kernel_size

                # purging the parameters that were originally allocated by the father constructor
                self.weight.requires_grad = False
                self.register_parameter('weight', None)
                self.weight = None

                if self.bias is not None:
                    self.bias.requires_grad = False
                    self.bias = None
                    self.register_parameter('bias', None)

            def forward(self, x):
                if self.kernel_size != self.base_kernel_size:
                    weight = torch.nn.functional.interpolate(self.base_weight,
                                                             size=self.kernel_size,
                                                             mode='bilinear',
                                                             align_corners=False,
                                                             recompute_scale_factor=False)

                    # keep the kernel integral coherent
                    weight = weight * torch.sum(torch.abs(self.base_weight)) / torch.sum(torch.abs(weight) + 1e-8)
                else:
                    weight = self.base_weight

                if self.padding_mode != 'zeros':
                    if isinstance(self.padding, str):
                        self._reversed_padding_repeated_twice = [0, 0] * len(weight)
                        if self.padding == 'same':
                            for d, k, i in zip(self.dilation, weight,
                                               range(len(weight) - 1, -1, -1)):
                                total_padding = d * (k - 1)
                                left_pad = total_padding // 2
                                self._reversed_padding_repeated_twice[2 * i] = left_pad
                                self._reversed_padding_repeated_twice[2 * i + 1] = (
                                        total_padding - left_pad)
                    else:
                        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
                    return torch.nn.functional.conv2d(torch.nn.functional.pad(x,
                                                                              self._reversed_padding_repeated_twice,
                                                                              mode=self.padding_mode),
                                                      weight, self.bias, self.stride,
                                                      _pair(0), self.dilation, self.groups)
                else:
                    return torch.nn.functional.conv2d(x, weight, bias=self.base_bias,
                                                      stride=self.stride,
                                                      padding=self.padding,
                                                      dilation=self.dilation,
                                                      groups=self.groups)

        # let's define a container for custom computational units (warning the uniform space coverage is violated!)
        class CustomModel(torch.nn.Module):
            def __init__(self, model, padding):
                super(CustomModel, self).__init__()
                assert model is not None and padding is not None, "Missing information on custom model."

                self.model = model  # any neural net
                self.padding = (padding, padding)  # this depends on the model that is used
                self.kernel_size = (1, 1)  # placeholder
                self.stride = (1, 1)  # placeholder
                self.dilation = (1, 1)  # placeholder

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        # building convolutional blocks (one per region) and related operations
        _crop_pads = []
        for i in range(0, self.region_count):

            # initial default choices (some of the will be altered in what follows)
            dst_kernel = self.kernel_size
            dst_dilation = self.dilation
            dst_stride = 1
            dst_downscale_fact = 1.0
            dst_extra_crop_pad = 0

            if method == "dilation" or method == "downscaling":
                dst_params = (self.kernel_size ** 2) * float(self.reduction_factors[i])
                dst_kernel = round(math.sqrt(dst_params))

                # promoting a reduction of parameters everytime that the reduction factor is < 1.0
                if float(self.reduction_factors[i]) < 1.0 and dst_kernel == self.kernel_size and dst_kernel > 2:
                    dst_kernel -= 1
                if dst_kernel == 1:
                    dst_kernel = 2

                if method == "dilation":

                    # preferring to avoid dilation unless the kernel reduction is significant
                    # (warning: this will end up in reducing the coverage of the filters for some configurations)
                    dst_dilation = self.dilation + math.floor(float(self.kernel_size - 1.) / float(dst_kernel - 1.))

                elif method == "downscaling":

                    # preferring to avoid downscaling unless the kernel reduction is significant
                    # (warning: this will end up in reducing the coverage of the filters for some configurations)
                    dst_downscale_fact = float(dst_kernel) / float(self.kernel_size)
                    if dst_downscale_fact >= 0.8:
                        dst_downscale_fact = 1.
                    dst_extra_crop_pad = 1 if dst_downscale_fact < 1.0 else 0

            elif method == "stride":

                # preferring to avoid stride unless the reduction factor is significant
                # (warning: this will end up in *INCREASING* the coverage of the filters for some configurations)
                dst_stride = round(1. / math.sqrt(self.reduction_factors[i]))
                dst_extra_crop_pad = dst_stride if dst_stride > 1 else 0

            elif method == "vanilla":
                pass

            # saving data
            dst_conv_padding = (int(dst_kernel) // 2) * dst_dilation
            _crop_pads.append(dst_conv_padding + dst_extra_crop_pad)
            self.downscale_factors.append(dst_downscale_fact)

            if custom_model is None:

                # banks
                if self.banks == "independent" or i == 0:
                    self.conv2ds.append(torch.nn.Conv2d(in_channels=self.in_channels,
                                                        out_channels=self.out_channels,
                                                        kernel_size=(dst_kernel, dst_kernel),
                                                        dilation=(dst_dilation, dst_dilation),
                                                        stride=(dst_stride, dst_stride),
                                                        padding=dst_conv_padding,
                                                        padding_mode=self.padding_mode,
                                                        groups=1,
                                                        bias=bias))

                elif self.banks == "shared":
                    self.conv2ds.append(InterpolatedKernelConv2d(self.conv2ds[0],
                                                                 kernel_size=(dst_kernel, dst_kernel),
                                                                 dilation=(dst_dilation, dst_dilation),
                                                                 stride=(dst_stride, dst_stride),
                                                                 padding=dst_conv_padding,
                                                                 padding_mode=self.padding_mode,
                                                                 groups=1,
                                                                 bias=bias))

            elif custom_model is not None:
                self.conv2ds.append(CustomModel(custom_model, custom_model_padding))
                _crop_pads[-1] = custom_model_padding + dst_extra_crop_pad

        # [debug only]
        # self.__set_debug_weights(0)

        # computing coordinates of each region (w.r.t. the frame centered in FOA (initially assumed to be in (0,0)).)
        # (warning: coordinates are [None, None] for the complementary-last region)
        # notice that a region might be divided into multiple regions due to cropped convolutions
        _region_count = 0
        _conv2ds = torch.nn.ModuleList()
        _downscale_factors = []
        _cropped_convs_extra_pads = []

        for i in range(0, self.region_count):
            region_size, region_corner, region_xx_yy = \
                self.__compute_region_data(_region_sizes[i], _region_sizes[i - 1] if i > 0 else None)

            new_region_sizes, new_region_corners, new_region_xxs_yys, crop_sizes, crop_corners, crop_xxs_yys = \
                self.__compute_region_and_crop_data(region_size, region_corner, region_xx_yy, _crop_pads[i])

            for r in range(0, len(new_region_sizes)):
                self.region_sizes.append(new_region_sizes[r])
                self.region_corners.append(new_region_corners[r])
                self.region_xxs_yys.append(new_region_xxs_yys[r])

                self.crop_sizes.append(crop_sizes[r])
                self.crop_corners.append(crop_corners[r])
                self.crop_xxs_yys.append(crop_xxs_yys[r])

                self.register_buffer('region_' + str(_region_count) + '_xx', new_region_xxs_yys[r][0], persistent=False)
                self.register_buffer('region_' + str(_region_count) + '_yy', new_region_xxs_yys[r][1], persistent=False)
                self.register_buffer('crop_' + str(_region_count) + '_xx', crop_xxs_yys[r][0], persistent=False)
                self.register_buffer('crop_' + str(_region_count) + '_yy', crop_xxs_yys[r][1], persistent=False)

                _conv2ds.append(self.conv2ds[i])
                _downscale_factors.append(self.downscale_factors[i])
                _region_count += 1

            if i == 0:
                self.__buffers_device = region_xx_yy[0].device

        # updating data with artificially generated regions (needed in case of tight crops)
        self.region_count = _region_count
        self.conv2ds = _conv2ds
        self.downscale_factors = _downscale_factors

    def forward(self, input_data, foa_xy, compute_region_indices=True):
        b = input_data.shape[0]
        h = input_data.shape[2]
        w = input_data.shape[3]
        d = input_data.device
        output_features = None

        # checking (image sizes are only known at "forward" time)
        assert h > self.max_region_size and w > self.max_region_size, \
            "One or more regions are bigger than (or equal to) the input size " + \
            str(h) + "x" + str(w) + " (" + str(self.region_sizes) + ")."

        # getting FOA coordinates
        foa_xy = foa_xy.to(torch.long)
        foa_x, foa_y = foa_xy[:, 0, None], foa_xy[:, 1, None]  # each of them is b x 1

        # better be sure that FOA is inside the current image
        assert torch.min(foa_xy) >= 0 and torch.max(foa_x) < h and torch.max(foa_y) < w, \
            "Focus coordinates are out of the image frame (\nfoa_x=" + str(foa_x) + "\nfoa_y=" + str(foa_y) + "\n)"

        # checking if device has changed and fixing references to buffers (if needed)
        self.__recollect_references_to_buffers(d)

        # preparing (and buffering) offsets that are needed when indexing data in the case of batches of size > 1
        self.__prepare_offset_tensors(b, h, w, d)

        # loop over the regions (from the largest to the smallest)
        for i in range(self.region_count - 1, -1, -1):

            # setting flags
            last_and_complementary_region = i == self.region_count - 1 and self.region_sizes[i][0] == -1
            last_and_not_complementary_region = i == self.region_count - 1 and self.region_sizes[i][0] != -1

            # crop a portion of the input (if needed)
            o, x_out_gap, y_out_gap = self.__crop_input(input_data, foa_x, foa_y, i) \
                if self.cropped_convs and not last_and_complementary_region else (input_data, 0, 0)

            # saving the size of the crop (or of the full input - if cropping was not performed)
            # cast to int to measure the FLOPS
            h_crop = int(o.shape[2])
            w_crop = int(o.shape[3])

            # skipping regions (crops) that are out of the image plane
            if h_crop == 0 or w_crop == 0:
                continue

            # downscale (if needed)
            o = torch.nn.functional.interpolate(o,
                                                size=(max(round(h_crop * self.downscale_factors[i]), 1),
                                                      max(round(w_crop * self.downscale_factors[i]), 1)),
                                                mode='bilinear',
                                                align_corners=False,
                                                recompute_scale_factor=False) \
                if self.downscale_factors[i] < 1. else o

            # convolution (over the full image or over the cropped data)
            o = self.conv2ds[i](o)

            # fixing even kernels
            kernel_coverage = int(self.conv2ds[i].kernel_size[0])
            o = o if kernel_coverage % 2 != 0 else o[:, :, :-self.conv2ds[i].dilation[0], :-self.conv2ds[i].dilation[0]]

            # upscale (due to stride or do to previously performed down-sampling)
            o = torch.nn.functional.interpolate(o,
                                                size=(h_crop, w_crop),
                                                mode='bilinear',
                                                align_corners=False,
                                                recompute_scale_factor=False) \
                if self.conv2ds[i].stride[0] > 1 or self.downscale_factors[i] < 1. else o

            # preparing the output tensors
            if last_and_complementary_region:
                output_features = o.contiguous()

                if compute_region_indices:
                    region_indices = torch.ones((b, h, w), dtype=torch.float32, device=o.device) * i
                continue
            elif last_and_not_complementary_region:
                output_features = torch.zeros((b, self.out_channels, h, w), dtype=torch.float32, device=d,
                                              requires_grad=o.requires_grad)

                if compute_region_indices:
                    region_indices = torch.ones(b, h, w, dtype=torch.float32, device=o.device) * (-1)

            # filling (a portion of) the output tensor with the output features of the current region ("i"-th region)
            self.__fill_tensor(output_features, o, foa_x, foa_y, i, x_out_gap, y_out_gap,
                               region_indices_to_fill=region_indices if compute_region_indices else None)

        # returning data
        if compute_region_indices:
            return output_features, region_indices
        else:
            return output_features

    def __str__(self):
        _s = "[" + self.__class__.__name__ + "]"
        _s += "\n- in_channels: " + str(self.in_channels)
        _s += "\n- out_channels: " + str(self.out_channels)
        _s += "\n- kernel_size: " + str(self.kernel_size)
        _s += "\n- dilation: " + str(self.dilation)
        _s += "\n- padding_mode: " + str(self.padding_mode)
        _s += "\n- region_type: " + str(self.region_type)
        _s += "\n- region_sizes: " + str(self.region_sizes)
        _s += "\n- region_count: " + str(self.region_count)
        _s += "\n- method: " + str(self.method)
        _s += "\n- banks: " + str(self.banks)
        _s += "\n- crop_type: " + str(self.crop_type)
        _s += "\n- reduction_factors: " + str(self.reduction_factors)
        _s += "\n- downscale_factors: " + str(self.downscale_factors)
        for i in range(0, self.region_count):
            _s += "\n--- conv2d: " + str(self.conv2ds[i].kernel_size[0]) + "x" + str(self.conv2ds[i].kernel_size[0])
            _s += ", dilation=" + str(self.conv2ds[i].dilation[0])
            _s += ", stride=" + str(self.conv2ds[i].stride[0])
            _s += ", padding=" + str(self.conv2ds[i].padding[0])
        return _s

    def __compute_region_data(self, region_size, prev_region_size):

        # the last-complementary region has no coordinates to store
        if region_size == -1:
            xx = None
            yy = None
            region_size_x = region_size  # placeholder
            region_size_y = region_size  # placeholder
            region_corner_x = - (region_size_x // 2)  # placeholder
            region_corner_y = - (region_size_y // 2)  # placeholder

        # box region
        elif self.region_type == "box":
            he = region_size // 2
            xs = torch.arange(start=-he, end=he + 1, dtype=torch.long)
            ys = torch.arange(start=-he, end=he + 1, dtype=torch.long)
            yy, xx = torch.meshgrid(ys, xs, indexing='xy')

            if prev_region_size is not None:
                he_in = prev_region_size // 2
                valid = torch.logical_or(torch.logical_or(xx < -he_in, xx > he_in),
                                         torch.logical_or(yy < -he_in, yy > he_in))
            else:
                valid = torch.ones(xx.shape, dtype=torch.bool, device=xx.device)

            xx = xx[valid].contiguous()
            yy = yy[valid].contiguous()

            xx.unsqueeze_(0)
            yy.unsqueeze_(0)

            region_size_x = region_size
            region_size_y = region_size
            region_corner_x = - (region_size_x // 2)
            region_corner_y = - (region_size_y // 2)

        # circular region
        elif self.region_type == "circle":
            radius = region_size // 2
            radius_in = prev_region_size // 2 if prev_region_size is not None else -1
            xs = torch.arange(start=-radius, end=radius + 1, dtype=torch.long)
            ys = torch.arange(start=-radius, end=radius + 1, dtype=torch.long)
            yy, xx = torch.meshgrid(ys, xs, indexing='xy')

            dist = torch.sqrt(xx ** 2 + yy ** 2)
            valid = torch.logical_and(dist > radius_in, dist <= radius)
            xx = xx[valid].contiguous()
            yy = yy[valid].contiguous()

            xx.unsqueeze_(0)
            yy.unsqueeze_(0)

            region_size_x = region_size
            region_size_y = region_size
            region_corner_x = - (region_size_x // 2)
            region_corner_y = - (region_size_y // 2)

        else:
            raise RuntimeError("Unexpected region type! (" + str(self.region_type) + ")")

        return (region_size_x, region_size_y), (region_corner_x, region_corner_y), (xx, yy)

    def __compute_region_and_crop_data(self, region_size, region_corner, region_xx_yy, crop_padding):
        region_sizes = []
        region_corners = []
        region_xxs_yys = []

        crop_sizes = []
        crop_corners = []
        crop_xxs_yys = []

        last_and_complementary_region = region_size[0] == -1

        # indices of the area to crop, needed only in case of cropped convolutions
        if self.cropped_convs and not last_and_complementary_region:

            def __compute_crop_data(_region_size, _region_corner, _crop_padding):
                _region_size_x = _region_size[0]
                _region_size_y = _region_size[1]
                _region_corner_x = _region_corner[0]
                _region_corner_y = _region_corner[1]

                crop_size_x = _region_size_x + 2 * _crop_padding
                crop_size_y = _region_size_y + 2 * _crop_padding
                crop_corner_x = _region_corner_x - _crop_padding
                crop_corner_y = _region_corner_y - _crop_padding

                xs = torch.arange(start=crop_corner_x, end=crop_corner_x + crop_size_x, dtype=torch.long)
                ys = torch.arange(start=crop_corner_y, end=crop_corner_y + crop_size_y, dtype=torch.long)

                _crop_yy, _crop_xx = torch.meshgrid(ys, xs, indexing='xy')
                _crop_xx = torch.flatten(_crop_xx).contiguous().unsqueeze(0)
                _crop_yy = torch.flatten(_crop_yy).contiguous().unsqueeze(0)

                return (crop_size_x, crop_size_y), (crop_corner_x, crop_corner_y), (_crop_xx, _crop_yy)

            if self.crop_type == "loose":

                # we only need to compute the coordinates for cropped convolutions
                crop_size, crop_corner, crop_xx_yy = __compute_crop_data(region_size, region_corner, crop_padding)

                region_sizes.append(region_size)
                region_corners.append(region_corner)
                region_xxs_yys.append(region_xx_yy)

                crop_sizes.append(crop_size)
                crop_corners.append(crop_corner)
                crop_xxs_yys.append(crop_xx_yy)

            elif self.crop_type == "tight":

                # in this case we needed to recompute the region coordinates, splitting the original coordinates into
                # the ones of multiple regions, and then we have to compute the coordinates for cropped convolutions
                # (warning: this is only optimal for symmetric regions (on all the axes, jointly))
                xx, yy = region_xx_yy

                region_size_x = region_size[0]
                region_size_y = region_size[1]
                region_corner_x = region_corner[0]
                region_corner_y = region_corner[1]

                # increasing the size of a squared area centered on the FOA, until it "touches" the current region
                e = int(0)
                while True:
                    if torch.count_nonzero(yy[xx == e] == e) > 0 or \
                            torch.count_nonzero(yy[xx == e] == -e) > 0 or \
                            torch.count_nonzero(yy[xx == -e] == e) > 0 or \
                            torch.count_nonzero(yy[xx == -e] == -e) > 0:
                        break
                    else:
                        e += 1
                inner_edge_x = 2 * e + 1
                inner_edge_y = 2 * e + 1

                # defining the sizes and center coordinates of the sub-regions (new regions)
                new_region_sizes = []
                new_region_corners = []
                if e > 0:
                    h = region_size_x // 2 - (inner_edge_x // 2) + 1
                    w = region_size_y

                    new_region_sizes.append((h, w))
                    new_region_corners.append((region_corner_x,
                                               region_corner_y))

                    new_region_sizes.append((h, w))
                    new_region_corners.append((inner_edge_x // 2,
                                               region_corner_y))

                    h = inner_edge_x - 2
                    w = region_size_y // 2 - (inner_edge_y // 2) + 1

                    new_region_sizes.append((h, w))
                    new_region_corners.append((region_corner_x + region_size_x // 2 - (inner_edge_x // 2) + 1,
                                               region_corner_y))

                    new_region_sizes.append((h, w))
                    new_region_corners.append((region_corner_x + region_size_x // 2 - inner_edge_x // 2 + 1,
                                               inner_edge_y // 2))
                else:
                    new_region_sizes.append(region_size)
                    new_region_corners.append(region_corner)

                # for each mini-crop, we have sub-select the region coordinates and
                # compute the coordinates for cropped convolutions
                for new_region_size, new_region_corner in zip(new_region_sizes, new_region_corners):
                    valid = torch.logical_and(torch.logical_and(xx >= new_region_corner[0],
                                                                xx < (new_region_corner[0] + new_region_size[0])),
                                              torch.logical_and(yy >= new_region_corner[1],
                                                                yy < (new_region_corner[1] + new_region_size[1])))
                    new_region_xx_yy = (xx[valid].contiguous(), yy[valid].contiguous())

                    region_sizes.append(new_region_size)
                    region_corners.append(new_region_corner)
                    region_xxs_yys.append(new_region_xx_yy)

                    crop_size, crop_corner, crop_xx_yy = \
                        __compute_crop_data(new_region_size, new_region_corner, crop_padding)

                    crop_sizes.append(crop_size)
                    crop_corners.append(crop_corner)
                    crop_xxs_yys.append(crop_xx_yy)

        else:
            region_sizes.append(region_size)
            region_corners.append(region_corner)
            region_xxs_yys.append(region_xx_yy)

            crop_sizes.append((None, None))
            crop_corners.append((None, None))
            crop_xxs_yys.append((None, None))

        return region_sizes, region_corners, region_xxs_yys, crop_sizes, crop_corners, crop_xxs_yys

    @staticmethod
    def __create_offset_tensor(b, c, h, w, device):
        b_idx = torch.arange(start=0, end=b, device=device, dtype=torch.long)
        b_offsets = b_idx.view(b, 1, 1) * (c * h * w)  # b x 1 x 1
        c_idx = torch.arange(start=0, end=c, device=device, dtype=torch.long)
        c_offsets = c_idx.view(1, c, 1) * (h * w)  # 1 x c x 1
        return b_offsets + c_offsets  # (b x 1 x 1) + (1 x c x 1) = b x c x 1

    def __recollect_references_to_buffers(self, device):
        if self.__buffers_device != device:
            region_xxs = [None] * self.region_count
            region_yys = [None] * self.region_count
            cropped_conv_xxs = [None] * self.region_count
            cropped_conv_yys = [None] * self.region_count

            for buf in self.named_buffers():
                buf_name, buf_data = buf

                if buf_name[0:6] == 'region':
                    i = int(buf_name[7:-3])
                    if buf_name[-2:] == 'xx':
                        region_xxs[i] = buf_data
                    elif buf_name[-2:] == 'yy':
                        region_yys[i] = buf_data

                elif buf_name[0:5] == 'crop_':
                    i = int(buf_name[5:-3])
                    if buf_name[-2:] == 'xx':
                        cropped_conv_xxs[i] = buf_data
                    elif buf_name[-2:] == 'yy':
                        cropped_conv_yys[i] = buf_data

            for i in range(0, self.region_count):
                self.region_xxs_yys[i] = (region_xxs[i], region_yys[i])
                self.crop_xxs_yys[i] = (cropped_conv_xxs[i], cropped_conv_yys[i])

            self.__buffers_device = device

    def __prepare_offset_tensors(self, b, h, w, device):

        # if batch size is greater than one...
        if b > 1:

            # if something changed in the way "forward" is called (input size, device, other options)...
            if self.__offsets_status is None or self.__offsets_status != [b, h, w, device, self.cropped_convs]:

                # store updated "forward" options
                self.__offsets_status = (b, h, w, device, self.cropped_convs)

                # create the offset tensor used to index the tensor that will store the output features
                self.__bf_offsets = self.__create_offset_tensor(b, self.out_channels, h, w, device=device)

                # create the offset tensor used to index the region indices output tensor
                self.__br_offsets = self.__create_offset_tensor(b, 1, h, w, device=device).view(b, 1)

                # if cropped convolutions are used, then we need additional offset tensors...
                if self.cropped_convs:

                    # create the offset tensor used to index the input data
                    # (full-res sizes or downscaled sizes)
                    self.__bc_offsets = [None] * self.region_count

                    # create the offset tensors used to index the input and output data
                    # (cropped sizes or downscaled-and-cropped sizes)
                    self.__bc_offsets_crop = [None] * self.region_count
                    self.__bf_offsets_crop = [None] * self.region_count

                    for i in range(0, self.region_count):
                        crop_size_x, crop_size_y = self.crop_sizes[i]

                        if crop_size_x is None or crop_size_y is None:
                            continue

                        self.__bc_offsets[i] = self.__create_offset_tensor(b, self.in_channels, h, w, device=device)
                        self.__bc_offsets_crop[i] = \
                            self.__create_offset_tensor(b, self.in_channels, crop_size_x, crop_size_y, device=device)
                        self.__bf_offsets_crop[i] = \
                            self.__create_offset_tensor(b, self.out_channels, crop_size_x, crop_size_y, device=device)

                else:

                    # clearing
                    self.__bc_offsets = self.__bc_offsets_crop = self.__bf_offsets_crop = None
        else:

            # clearing
            self.__offsets_status = self.__bf_offsets = self.__br_offsets = None
            self.__bc_offsets = self.__bc_offsets_crop = self.__bf_offsets_crop = None

    def __set_debug_weights(self, w_type=0):
        with torch.no_grad():
            for i in range(0, self.region_count):
                if self.conv2ds[i].weight is not None:

                    if w_type == 0:
                        self.conv2ds[i].weight *= 0.
                        mid = int(self.conv2ds[i].kernel_size[0]) // 2
                        self.conv2ds[i].weight[:, :, mid, mid] = 1.
                        if self.region_count > 1:
                            self.conv2ds[i].weight[:, :, mid, mid] -= float(i) / float(self.region_count)
                        self.conv2ds[i].weight /= self.conv2ds[i].weight.shape[1]

                    elif w_type == 1:
                        self.conv2ds[i].weight[:, :, :, :] = 1.
                        self.conv2ds[i].weight /= float(self.conv2ds[i].weight.shape[1] *
                                                        self.conv2ds[i].weight.shape[2] *
                                                        self.conv2ds[i].weight.shape[3])

                    self.conv2ds[i].bias *= 0

    def __crop_input(self, tensor_to_crop, foa_x, foa_y, region_id):
        b = tensor_to_crop.shape[0]
        c = tensor_to_crop.shape[1]
        h = tensor_to_crop.shape[2]
        w = tensor_to_crop.shape[3]

        crop_corner_x, crop_corner_y = self.crop_corners[region_id]
        crop_size_x, crop_size_y = self.crop_sizes[region_id]

        # (ref: input_data)
        xf = foa_x + crop_corner_x  # b x 1
        xt = foa_x + crop_corner_x + crop_size_x - 1  # b x 1
        yf = foa_y + crop_corner_y  # b x 1
        yt = foa_y + crop_corner_y + crop_size_y - 1  # b x 1

        # (ref: input_data)
        xfc = torch.clamp(xf, 0, None)  # b x 1
        xtc = torch.clamp(xt, None, h - 1)  # b x 1
        yfc = torch.clamp(yf, 0, None)  # b x 1
        ytc = torch.clamp(yt, None, w - 1)  # b x 1

        # only one of the terms of each summation below is (eventually) non-zero (ref: input_data)
        x_out_gap = torch.clamp(xf - xfc, None, 0) + torch.clamp(xt - xtc, None, 0)  # b x 1
        y_out_gap = torch.clamp(yf - yfc, None, 0) + torch.clamp(yt - ytc, None, 0)  # b x 1

        if b == 1:

            # if we ask for a crop that is fully out of the original tensor, we get xtc < xfc and/or ytc < yfc,
            # so we force conditions that will make the code generate a crop of size 0x0, that is detected and discarded
            if xtc[0] < xfc[0] or ytc[0] < yfc[0]:
                xtc[0] = xfc[0] - 1
                ytc[0] = yfc[0] - 1

            # bound-based selection
            cropped_tensor = tensor_to_crop[:, :, xfc[0]:(xtc[0] + 1), yfc[0]:(ytc[0] + 1)]

        else:

            # getting cropped-convolution area coordinates
            xx, yy = self.crop_xxs_yys[region_id]  # each of them is 1 x g ("g" is number of pix of the area)

            # shifting the area coordinates the by focus of attention
            xx = xx + foa_x  # (1 x r) + (b x 1) = b x r
            yy = yy + foa_y  # (1 x r) + (b x 1) = b x r

            # getting the bool mask (b x r) with 1s in positions that are not out-of-border
            valid = torch.logical_and(torch.logical_and(xx >= 0, xx < h), torch.logical_and(yy >= 0, yy < w))

            # creating a single flat vector of indices of the cropped-convolution area points for all the examples
            # in the batch and all the input channels: recall that "xx", "yy", "valid" are "b x r"
            idx = (xx * w + yy).unsqueeze(dim=1)  # b x 1 x r
            idx = self.__bc_offsets[region_id] + idx  # (b x c x 1) + (b x 1 x r) = b x c x r

            # keeping only the indices associated to points that are not out-of-borders
            # (notice that the selection is broad-casted over the in-channels dimension)
            source_idx = torch.masked_select(idx, valid.view(b, 1, valid.shape[1]))  # c*(\sum_{j=0}^{b-1} g_in_j)

            # creating the tensor that will store the cropped data
            cropped_tensor = torch.zeros(b, c, crop_size_x, crop_size_y, device=tensor_to_crop.device)

            # moving xx, yy to the reference frame of the cropped area
            tt_x = foa_x + crop_corner_x  # b x 1
            tt_y = foa_y + crop_corner_y  # b x 1
            idx = ((xx - tt_x) * crop_size_y + (yy - tt_y)).unsqueeze(dim=1)
            idx = self.__bc_offsets_crop[region_id] + idx

            # keeping only the indices associated to points that are not out-of-borders
            # (notice that the selection is broad-casted over the in-channels dimension)
            cropped_area_idx = torch.masked_select(idx, valid.view(b, 1, valid.shape[1]))

            # storing
            cropped_tensor.view(-1)[cropped_area_idx] = tensor_to_crop.view(-1)[source_idx]

        return cropped_tensor, x_out_gap, y_out_gap

    def __fill_tensor(self, tensor_to_fill, source_tensor, foa_x, foa_y, region_id,
                      x_out_gap, y_out_gap, region_indices_to_fill=None):
        b = tensor_to_fill.shape[0]
        h = tensor_to_fill.shape[2]
        w = tensor_to_fill.shape[3]

        # getting region coordinates
        xx, yy = self.region_xxs_yys[region_id]  # each of them is 1 x r ("r" is the number of points in the region)

        # shifting region coordinates by focus of attention (ref: input_data)
        xx = xx + foa_x  # (1 x r) + (b x 1) = b x r
        yy = yy + foa_y  # (1 x r) + (b x 1) = b x r

        # getting the bool mask (b x r) with 1s in positions associated to coordinates that are not out-of-border
        valid = torch.logical_and(torch.logical_and(xx >= 0, xx < h), torch.logical_and(yy >= 0, yy < w))  # b x r

        # storing into the output tensor
        if b == 1:

            # going from from (1 x r) to two flat vectors of size "r_in"
            # where "r_in" is the number of not-out-of-border points in the current region
            xx = xx[valid]  # r_in
            yy = yy[valid]  # r_in
            dest_idx = xx * w + yy  # r_in

            if self.cropped_convs:
                crop_corner_x = self.crop_corners[region_id][0]
                crop_corner_y = self.crop_corners[region_id][1]

                # in case of cropped convolutions, we need to compute the translation offset to map coordinates in the
                # reference frame of the cropped area (ref: cropped_area)
                t_x = foa_x + crop_corner_x - x_out_gap  # b x 1
                t_y = foa_y + crop_corner_y - y_out_gap  # b x 1

                source_idx = ((xx - t_x.squeeze(0)) * source_tensor.shape[3] + (yy - t_y.squeeze(0)))
            else:
                source_idx = dest_idx

            # filling the output tensor
            source_tensor = source_tensor.contiguous()
            tensor_to_fill.view(b, self.out_channels, -1)[:, :, dest_idx] = \
                source_tensor.view(b, self.out_channels, -1)[:, :, source_idx]

            # eventually filling the tensor with region indices
            if region_indices_to_fill is not None:
                region_indices_to_fill.view(b, -1)[:, dest_idx] = region_id

        else:

            # creating a single flat vector of indices of the region points for all the examples
            # in the batch and all the out features: recall that "xx", "yy", "valid" are "b x r"
            dest_coord_idx = (xx * w + yy).unsqueeze(dim=1)  # b x 1 x r
            idx = self.__bf_offsets + dest_coord_idx  # (b x f x 1) + (b x 1 x r) = b x f x r

            # keeping only the indices associated to points that are not out-of-borders
            # (notice that the selection is broad-casted over the feature dimension)
            dest_idx = torch.masked_select(idx, valid.view(b, 1, valid.shape[1]))  # f*(\sum_{j=0}^{b-1}r_in_j)

            if self.cropped_convs:
                crop_corner_x = self.crop_corners[region_id][0]
                crop_corner_y = self.crop_corners[region_id][1]

                # in case of cropped convolutions, we need to compute the translation offset to map coordinates in the
                # reference frame of the cropped area
                t_x = foa_x + crop_corner_x  # b x 1
                t_y = foa_y + crop_corner_y  # b x 1

                # creating a single flat vector of indices of the cropped conv region points for all the examples
                # in the batch and all the out features: recall that "xx", "yy", "valid" are "b x r"
                idx = ((xx - t_x) * source_tensor.shape[3] + (yy - t_y)).unsqueeze(dim=1)
                idx = self.__bf_offsets_crop[region_id] + idx

                # keeping only the indices associated to points that are not out-of-borders
                # (notice that the selection is broad-casted over the feature dimension)
                source_idx = torch.masked_select(idx, valid.view(b, 1, valid.shape[1]))
            else:
                source_idx = dest_idx

            # filling the output tensor
            source_tensor = source_tensor.contiguous()
            tensor_to_fill.view(-1)[dest_idx] = source_tensor.view(-1)[source_idx]

            if region_indices_to_fill is not None:
                # creating a single flat vector of indices of the region points for all the examples
                idx = self.__br_offsets + dest_coord_idx.view(b, -1)  # (b x 1) + (b x r) = b x r

                # keeping only the indices associated to points that are not out-of-borders
                region_idx = idx[valid]  # flat

                # filling the region indices tensor
                region_indices_to_fill.view(-1)[region_idx] = region_id
