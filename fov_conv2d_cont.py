import torch
import math


class FovConv2dCont(torch.nn.Module):
    """Foveated Convolutional Layer with a continuous dependency from the FOA coordinates.

    This type of convolutional layer is based on the idea of exploiting a different kernel in different pixel
    coordinates, in function on their 'distance' from the FOA.

    There are different ways to achieve such an effect. Here we explore three different cases.
    1) A learnable base kernel is modulated by a Gaussian function that depends on the distance from FOA. The actual
       implementation does NOT alter the base kernel. The input data is altered by the Gaussian function (what we refer
       to as 'input modulation') and, afterwards, a classic convolution is applied using the learnable base kernel.
       The Gaussian function is approximated on a squared area ('modulating kernel').
    2) Same as above, but the modulating function is not a Gaussian, but it is computed by a Neural Network that, given
       the difference between the FOA coordinates and the considered (x,y)-pixel, outputs a modulating function. The
       network has some architectural constraints to fulfil (see the code), and it outputs the function on a squared
       area ('modulating kernel').
    3) Finally, we considered the case in which a Neural Network that, given the difference between the FOA coordinates
       and the considered (x,y)-pixel, outputs the kernel to be used to compute the convolution (i.e., there is no
       base kernel, we immediately get the kernel to use for each coordinate).
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 kernel_size=5,
                 dilation=1,
                 padding_mode='zeros',
                 kernel_type="gaussian_modulated",
                 **kwargs):
        super(FovConv2dCont, self).__init__()

        # checking arguments
        assert kernel_type in ["gaussian_modulated", "net_modulated", "net_generated"], \
            "Invalid kernel type: " + str(kernel_type) + \
            " (it must be one of: gaussian_modulated, net_modulated, net_generated)"

        # main operations
        self.input_modulation = None
        self.convolution = None

        # wrapping PyTorch Conv2d into a dummy class with a 2-argument forward
        # (just to make the interface uniform with the other type of convolution implemented here)
        class Conv2d(torch.nn.Conv2d):
            def __init__(self, *_args, **_kwargs):
                super(Conv2d, self).__init__(*_args, **_kwargs)

                # [debug only]
                # self.__set_debug_weights(0)

            def forward(self, x, foa_xy):
                return super().forward(x)

            def __set_debug_weights(self, w_type=0):
                with torch.no_grad():
                    if w_type == 0:
                        self.weight *= 0.
                        mid = int(self.kernel_size[0]) // 2
                        self.weight[:, :, mid, mid] = 1.

                    elif w_type == 1:
                        self.weight[:, :, :, :] = 1.
                        self.weight /= float(self.weight.shape[1] *
                                             self.weight.shape[2] *
                                             self.weight.shape[3])

                    self.bias *= 0

        # another helper class introduced only to make the interface more uniform
        # (its a callable object that simply returns the first argument of the call)
        class Identity:
            def __call__(self, a, b):
                return a

            def __str__(self):
                return self.__class__.__name__

        if kernel_type == "gaussian_modulated":
            self.input_modulation = GaussianFOAConv2d(**kwargs)  # last is 'gaussian_kernel_size', 'sigma_min', ...
            self.convolution = Conv2d(in_channels, out_channels, int(kernel_size),
                                      padding=int((kernel_size-1)*dilation + 1) // 2, stride=1,
                                      dilation=dilation, groups=1, bias=True, padding_mode=padding_mode)

        elif kernel_type == "net_modulated":
            assert len(kwargs) == 1 and 'kernel_net' in kwargs, "Too many arguments or missing argument kernel_net."
            self.__kernel_net = kwargs['kernel_net']  # saving this ref is needed to register the net in this module

            # checking the kernel network, when it is used as a modulator.
            # when the input is (0,0) the network must output a kernel that is 1 only in the middle of the spatial area.
            # this can be achieved using a network with no-biases-at all (read below) and activations that are 0 in 0.
            # there is indeed a bias that must set to 1 and kept fixed, and it is the one of the neuron associated to
            # the middle of the spatial area.
            # we provide a Linear layer that already implements a bias like that, that is LinearMiddleBiasOne.
            with torch.no_grad():
                ker = self.__kernel_net(torch.tensor([[0., 0.]]))  # n x (kernel_size^2)
                ker_size = int(math.sqrt(ker.shape[1]))
                assert ker_size % 2 == 1, \
                    "Invalid kernel network (invalid size of the generated output - odd kernel needed)."
                assert ker[0, ker.shape[1] // 2] == 1., \
                    "Invalid kernel network (invalid output at zero coordinates)."
                assert torch.sum(torch.abs(ker)) == 1., \
                    "Invalid kernel network (invalid output at zero coordinates)."

            self.input_modulation = NeuralFOAConv2d(1, 1, dilation=1, **kwargs)  # last is 'kernel_net'
            self.convolution = Conv2d(in_channels, out_channels, int(kernel_size),
                                      padding=int((kernel_size-1)*dilation + 1) // 2, stride=1,
                                      dilation=dilation, groups=1, bias=True, padding_mode=padding_mode)

        elif kernel_type == "net_generated":
            assert len(kwargs) == 1 and 'kernel_net' in kwargs, "Too many arguments or missing argument kernel_net."
            self.__kernel_net = kwargs['kernel_net']  # saving this ref is needed to register the net in this module

            # checking consistency of the net output and the specified kernel parameters
            with torch.no_grad():
                ker = self.__kernel_net(torch.tensor([[0., 0.]]))  # n x (tot_kernel_size^2)
                ker_size = int(math.sqrt(ker.shape[1] / (in_channels * out_channels)))
                assert ker_size % 2 == 1, \
                    "Invalid kernel network (invalid size of the generated output - odd kernel needed)."
                assert ker_size == int(kernel_size), \
                    "Invalid kernel network (invalid size of the generated kernel - doesn't match the kernel_size arg)."

            self.input_modulation = Identity()
            self.convolution = NeuralFOAConv2d(in_channels, out_channels, dilation=dilation, padding_mode=padding_mode,
                                               **kwargs)  # last is 'kernel_net'

    def forward(self, input_data, foa_xy, compute_region_indices=False):
        x = self.input_modulation(input_data, foa_xy)
        x = self.convolution(x, foa_xy)
        # returning data
        if compute_region_indices:
            return x, None
        else:
            return x

    def __str__(self):
        _s = "[" + self.__class__.__name__ + "]"
        _s += "\n- input_modulation:\n{\n\t"
        _s += (str(self.input_modulation).replace('\n', '\n\t') if self.input_modulation else "\tnone") + "\n}"
        _s += "\n- convolution:\n{\n\t"
        _s += str(self.convolution).replace('\n', '\n\t')
        _s += "\n}"
        return _s


class BaseFOAConv2d:
    """Base class for convolutions in which the filter depends on the FOA coordinates."""

    def __init__(self, dilation=1, padding_mode='zeros'):
        assert dilation >= 1 and (dilation - round(dilation)) == 0, "Invalid dilation factor (" + str(dilation) + ")."
        assert type(padding_mode) is str, "Invalid padding mode (usually: 'zeros')."

        # these attributes will store references to cached computations
        self.h = -1
        self.w = -1
        self.xx_yy = None
        self.dilation = int(dilation)
        self.padding_mode = padding_mode

    def __call__(self, input_data, foa_xy):
        """Convolution with per-pixel kernels (it supports batched data)."""

        # Basics:
        # - The number of channels in the input image is 'c', while the number of input channel
        #   in the kernel is 'in_channels', and they play two different roles.
        # - Similarly, the number of kernel out channels is 'out_channels', that is not necessarily the number of
        #   per-pixel-features generated by the convolution (read what follows).
        # - A kernel that is spatial-only (i.e., that is independently applied to each of the 'c' channels), has
        #   in_channels = 1 and out_channels = 1, and the outcome of the convolution is such that there will be
        #   c-features per pixel.
        # - A kernel that operates in spatial-and-depth-wise manner, has in_channels = c and a custom number
        #   of out_channels, and the outcome of the convolution is such that there will be out_channels-features
        #   per pixel.
        b = input_data.shape[0]
        c = input_data.shape[1]

        # if needed, pre-compute some shared values (again, only if needed)
        self.build_shared_internal_cache(input_data)
        self.build_custom_internal_cache(input_data)

        # generating a kernel for each input coordinate, in function of FOA
        # per_pixel_kernels is expected to be: b x out_channels x in_channels x kk x wh
        per_pixel_kernels = self.generate_per_pixel_kernels(foa_xy)

        # guessing kernel properties from the generated data
        out_channels = per_pixel_kernels.shape[1]
        in_channels = per_pixel_kernels.shape[2]
        kernel_size = int(math.sqrt(per_pixel_kernels.shape[3]))
        spatial_only_kernel = in_channels == 1 and out_channels == 1

        # checking
        assert in_channels == 1 or c == in_channels, \
            "Invalid kernel 'in_channels' (" + str(in_channels) + ") for an input image with " + str(c) + " channels."

        # extract receptive inputs (patches, one per pixel): b x (c x kk) x wh
        if self.padding_mode == 'zeros':
            patches = torch.nn.functional.unfold(input_data, kernel_size=kernel_size,
                                                 dilation=self.dilation, padding=((kernel_size-1)*self.dilation+1) // 2,
                                                 stride=1)
        else:
            pad = ((kernel_size - 1) * self.dilation + 1) // 2
            padded_input_data = torch.nn.functional.pad(input_data, (pad, pad, pad, pad),
                                                        mode=self.padding_mode)
            patches = torch.nn.functional.unfold(padded_input_data, kernel_size=kernel_size,
                                                 dilation=self.dilation,
                                                 padding=0,
                                                 stride=1)

        # convolutions
        patches = patches.view(b, 1, c, kernel_size ** 2, -1)  # b x 1 x c x kk x wh

        if spatial_only_kernel:
            dims_to_sum = 3  # summing over the spatial area covered by the convolution (not on the input channels)
            conv_out_channels = c
        else:
            dims_to_sum = [2, 3]  # summing over the whole kernel volume (input channels and spatial kernel area)
            conv_out_channels = out_channels

        conv_data = torch.sum(patches * per_pixel_kernels, dim=dims_to_sum)
        return conv_data.view(b, conv_out_channels, self.h, self.w)

    def build_shared_internal_cache(self, input_data):
        """Check and save input properties, and compute cached data that are function of such properties."""

        if self.h != input_data.shape[2] or self.w != input_data.shape[3] \
                or self.xx_yy.device != input_data.device:
            self.h = input_data.shape[2]
            self.w = input_data.shape[3]
            xs = torch.arange(start=0, end=self.h, dtype=torch.long, device=input_data.device)
            ys = torch.arange(start=0, end=self.w, dtype=torch.long, device=input_data.device)
            yy, xx = torch.meshgrid(ys, xs, indexing='xy')
            self.xx_yy = torch.stack([xx, yy], dim=2).view(-1, 2)  # hw x 2 (row-wise scanning of coordinates)

    def generate_per_pixel_kernels(self, foa_xy):
        raise NotImplementedError("Implement this method in your son class!")

    def build_custom_internal_cache(self, input_data):
        raise NotImplementedError("Implement this method in your son class!")


class GaussianFOAConv2d(BaseFOAConv2d):
    """Convolution with a Gaussian with a sigma that changes in function of the distance from FOA."""

    def __init__(self, gaussian_kernel_size=5, sigma_min=0.01, sigma_max=1.0, sigma_function="linear", bias=False):
        super(GaussianFOAConv2d, self).__init__(dilation=1)

        # checking arguments
        assert gaussian_kernel_size > 0, "Invalid size of the Gaussian kernel."
        assert sigma_max > sigma_min > 0 and sigma_max > 0, \
            "Invalid sigmas: " + str(sigma_min) + ", " + str(sigma_max) + \
            " (they must be > 0 and in ascending order)."
        assert sigma_function in ["linear", "exponential"], \
            "Unknown sigma function: " + str(sigma_function) + " (it must be one of: linear, exponential)."

        # saving arguments
        self.gaussian_kernel_size = int(gaussian_kernel_size)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if sigma_function == "linear":
            self.sigma_function = self.__sigma_function_linear
        elif sigma_function == "exponential":
            self.sigma_function = self.__sigma_function_exponential

        # this attribute will store references to (custom) cached computations
        self.__kernel_sq_dists = None

    def generate_per_pixel_kernels(self, foa_xy):
        b = foa_xy.shape[0]
        kk = self.gaussian_kernel_size ** 2
        hw = self.h * self.w

        # generate sigma values (a sigma for each pixel, in function of distance to FOA)
        sigmas = self.sigma_function(foa_xy)  # a sigma for each pixel (b x hw)

        # generate kernels (a kernel for each pixel)
        # self.__kernel_sq_dists: kk -> 1 x kk x 1
        # sigmas: b x hw -> b x 1 x hw
        # per_pixel_kernels: b x kk x hw
        per_pixel_kernels = torch.exp((-self.__kernel_sq_dists.unsqueeze(0).unsqueeze(2) /
                                       (2 * (sigmas.unsqueeze(1) ** 2))))  # b x kk x wh

        # normalizing kernels
        per_pixel_kernels /= torch.sum(per_pixel_kernels, dim=1, keepdim=True)

        # per_pixel_kernels is expected to be b x out_channels x in_channels x kk x wh, that, in this case, is
        # basically b x 1 x 1 x kk x wh
        return per_pixel_kernels.view(b, 1, 1, kk, hw)

    def build_custom_internal_cache(self, input_data):
        if self.__kernel_sq_dists is None or self.__kernel_sq_dists.device != input_data.device:
            xs = torch.arange(start=-(self.gaussian_kernel_size // 2),
                              end=(self.gaussian_kernel_size // 2) + 1, dtype=torch.long, device=input_data.device)
            yyk, xxk = torch.meshgrid(xs, xs, indexing='xy')
            self.__kernel_sq_dists = (xxk ** 2 + yyk ** 2).view(-1)

    def __sigma_function_linear(self, foa_xy):

        # self.xx_yy: hw x 2 -> 1 x hw x 2
        # foa_xy: b x 2 -> b x 1 x 2
        # diff_xx_yy: b x hw x 2
        diff_xx_yy = self.xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # (1 x hw x 2) - (b x 1 x 2) = b x hw x 2

        # normalized by the diagonal of the image
        dists = torch.sqrt(torch.sum(diff_xx_yy ** 2., dim=2)) / math.sqrt(self.h ** 2 + self.w ** 2)  # b x hw

        sigmas = (1.0 - dists) * self.sigma_min + dists * self.sigma_max  # b x hw
        return sigmas  # b x hw

    def __sigma_function_exponential(self, foa_xy):

        # self.xx_yy: hw x 2
        # foa_xy: b x 2
        # diff_xx_yy: b x hw x 2
        diff_xx_yy = self.xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # (1 x hw x 2) - (b x 1 x 2) = b x hw x 2

        # normalized by the (squared) diagonal of the image, goes to 0 in 3 * standard-deviation
        weights = torch.exp(-torch.sum(diff_xx_yy ** 2., dim=2) / (2. * (0.333 ** 2) * (self.h ** 2 + self.w ** 2)))

        sigmas = weights * self.sigma_min + (1.0 - weights) * self.sigma_max   # b x hw
        return sigmas  # b x hw

    def __str__(self):
        _s = "[" + self.__class__.__name__ + "]"
        _s += "\n- gaussian_kernel_size: " + str(self.gaussian_kernel_size)
        _s += "\n- sigma_min: " + str(self.sigma_min)
        _s += "\n- sigma_max: " + str(self.sigma_max)
        _s += "\n- sigma_function: " + str(self.sigma_function)
        return _s


class NeuralFOAConv2d(BaseFOAConv2d):
    """Convolution at (x,y) with a kernel generated by a Neural Network in function (x,y) in the frame of FOA."""

    def __init__(self, in_channels, out_channels, dilation=1, padding_mode='zeros', kernel_net=None):
        super(NeuralFOAConv2d, self).__init__(dilation=dilation, padding_mode=padding_mode)

        # checking arguments
        assert kernel_net is not None, "The kernel network is missing (kernel_net argument)."
        assert in_channels > 0 and (in_channels - round(in_channels) == 0), "Invalid number of input channels."
        assert out_channels > 0 and (out_channels - round(out_channels) == 0), "Invalid number of output channels."

        # main attributes: the size of the input/output features and the expected size of the generated kernel (or -1)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # neural network that processes n x 2 inputs, that are "n" pairs of spatial coordinates,
        # and it outputs n kernels, i.e., n x (in_channels x out_channels x kernel_size^2)
        self.kernel_net = kernel_net

    def generate_per_pixel_kernels(self, foa_xy):
        b = foa_xy.shape[0]
        hw = self.h * self.w

        # self.xx_yy: hw x 2 -> 1 x hw x 2
        # foa_xy: b x 2 -> b x 1 x 2
        # diff_xx_yy: b x hw x 2
        diff_xx_yy = self.xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # (1 x hw x 2) - (b x 1 x 2) = b x hw x 2

        # network-based generation of the modulating kernels
        # input: bhw x 2
        # output: bhw x (out_channels x in_channels x kk)
        per_pixel_kernels = self.kernel_net(diff_xx_yy.view(-1, 2).to(torch.float))

        # guessing the kernel size from the network output
        kk = per_pixel_kernels.shape[1] // (self.out_channels * self.in_channels)

        # checking coherence
        assert kk > 0, "The number of out_channels and in_channels do not match with the network output."

        # per_pixel_kernels is expected to be b x out_channels x in_channels x kk x wh
        per_pixel_kernels = per_pixel_kernels.view(b, hw, self.out_channels, self.in_channels, kk)
        return torch.permute(per_pixel_kernels, (0, 2, 3, 4, 1))  # likely a view (slowing down the next operations)

    def build_custom_internal_cache(self, input_data):
        pass

    def __str__(self):
        _s = "[" + self.__class__.__name__ + "]"
        _s += "\n- in_channels: " + str(self.in_channels)
        _s += "\n- out_channels: " + str(self.out_channels)
        _s += "\n- dilation: " + str(self.dilation)
        _s += "\n- padding_mode: " + str(self.padding_mode)
        _s += "\n- kernel_net:\n\t"
        _s += str(self.kernel_net).replace('\n', '\n\t')
        return _s


class LinearMiddleBiasOne(torch.nn.Linear):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(LinearMiddleBiasOne, self).__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x = super().forward(x)
        x[:, self.out_features // 2] += 1.
        return x
