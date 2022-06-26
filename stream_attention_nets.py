import torch
from fov_conv2d_reg import FovConv2dReg
from fov_conv2d_cont import FovConv2dCont, LinearMiddleBiasOne


def create_networks(_arch_type, _num_outputs, _device):
    # class of the feature extractor
    class SequentialFeatureExtractor(torch.nn.Module):

        def __init__(self, arch_type):
            super(SequentialFeatureExtractor, self).__init__()

            if arch_type == 'cnn':
                self.layer_1 = torch.nn.Conv2d(1, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_2 = torch.nn.Conv2d(32, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_3 = torch.nn.Conv2d(32, 32, kernel_size=7, dilation=2, padding=6, bias=True)

            elif arch_type == 'fov_reg_all':
                self.layer_1 = FovConv2dReg(1, 32, kernel_size=5, dilation=2, region_type="circle",
                                            method="downscaling", region_sizes=(87, -1),
                                            reduction_factors=(1.0, 0.5), banks="shared", bias=True)
                self.layer_2 = FovConv2dReg(32, 32, kernel_size=5, dilation=2, region_type="circle",
                                            method="downscaling", region_sizes=(59, -1),
                                            reduction_factors=(1.0, 0.35), banks="shared", bias=True)
                self.layer_3 = FovConv2dReg(32, 32, kernel_size=7, dilation=2, region_type="circle",
                                            method="downscaling", region_sizes=(29, -1),
                                            reduction_factors=(1.0, 0.25), banks="shared", bias=True)

            elif arch_type == 'fov_reg_last':
                self.layer_1 = torch.nn.Conv2d(1, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_2 = torch.nn.Conv2d(32, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_3 = FovConv2dReg(32, 32, kernel_size=7, dilation=2, region_type="circle",
                                            method="downscaling", region_sizes=(29, -1),
                                            reduction_factors=(1.0, 0.25), banks="shared", bias=True)

            elif arch_type == 'fov_net_all':
                self.layer_1 = FovConv2dCont(1, 32, kernel_size=5, dilation=2, kernel_type='net_modulated',
                                             kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False),
                                                                            torch.nn.Tanh(),
                                                                            LinearMiddleBiasOne(10, 7 * 7)))
                self.layer_2 = FovConv2dCont(32, 32, kernel_size=5, dilation=2, kernel_type='net_modulated',
                                             kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False),
                                                                            torch.nn.Tanh(),
                                                                            LinearMiddleBiasOne(10, 7 * 7)))
                self.layer_3 = FovConv2dCont(32, 32, kernel_size=7, dilation=2, kernel_type='net_modulated',
                                             kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False),
                                                                            torch.nn.Tanh(),
                                                                            LinearMiddleBiasOne(10, 7 * 7)))

            elif arch_type == 'fov_net_last':
                self.layer_1 = torch.nn.Conv2d(1, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_2 = torch.nn.Conv2d(32, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_3 = FovConv2dCont(32, 32, 7, dilation=2, kernel_type='net_modulated',
                                             kernel_net=torch.nn.Sequential(torch.nn.Linear(2, 10, bias=False),
                                                                            torch.nn.Tanh(),
                                                                            LinearMiddleBiasOne(10, 7 * 7)))

            elif arch_type == 'fov_gaussian_all':
                self.layer_1 = FovConv2dCont(1, 32, kernel_size=5, dilation=2, kernel_type='gaussian_modulated',
                                             gaussian_kernel_size=7, sigma_min=0.01, sigma_max=1.0,
                                             sigma_function='exponential', bias=True)
                self.layer_2 = FovConv2dCont(32, 32, kernel_size=5, dilation=2, kernel_type='gaussian_modulated',
                                             gaussian_kernel_size=7, sigma_min=0.01, sigma_max=5.0,
                                             sigma_function='exponential', bias=True)
                self.layer_3 = FovConv2dCont(32, 32, kernel_size=7, dilation=2, kernel_type='gaussian_modulated',
                                             gaussian_kernel_size=7, sigma_min=0.01, sigma_max=10.0,
                                             sigma_function='exponential', bias=True)

            elif arch_type == 'fov_gaussian_last':
                self.layer_1 = torch.nn.Conv2d(1, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_2 = torch.nn.Conv2d(32, 32, kernel_size=5, dilation=2, padding=4, bias=True)
                self.layer_3 = FovConv2dCont(32, 32, kernel_size=7, dilation=2, kernel_type='gaussian_modulated',
                                             gaussian_kernel_size=7, sigma_min=0.01, sigma_max=10.0,
                                             sigma_function='exponential', bias=True)

        def forward(self, _input_data, _foa_xy):
            if type(self.layer_1) is FovConv2dReg:
                x = self.layer_1(_input_data, _foa_xy, compute_region_indices=False)
            elif type(self.layer_1) is FovConv2dCont:
                x = self.layer_1(_input_data, _foa_xy)
            else:
                x = self.layer_1(_input_data)

            x = torch.nn.functional.leaky_relu(x)

            if type(self.layer_2) is FovConv2dReg:
                x = self.layer_2(x, _foa_xy, compute_region_indices=False)
            elif type(self.layer_2) is FovConv2dCont:
                x = self.layer_2(x, _foa_xy)
            else:
                x = self.layer_2(x)

            x = torch.nn.functional.leaky_relu(x)

            if type(self.layer_3) is FovConv2dReg:
                x, reg_idx = self.layer_3(x, _foa_xy, compute_region_indices=True)
            elif type(self.layer_3) is FovConv2dCont:
                x = self.layer_3(x, _foa_xy)
                reg_idx = None
            else:
                x = self.layer_3(x)
                reg_idx = None

            x = torch.nn.functional.leaky_relu(x)
            return x, reg_idx

    # feature extractor
    feature_extractor = SequentialFeatureExtractor(_arch_type).to(torch.device(_device))

    # classifier (head)
    head_classifier = torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, kernel_size=1, bias=True),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, _num_outputs, kernel_size=1, bias=True)
    ).to(torch.device(_device))

    return feature_extractor, head_classifier
