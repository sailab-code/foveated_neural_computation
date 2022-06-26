import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import FoveateNet, FoveateNetCircular, OfflineFoveateNet
import torchvision
from torchvision import transforms as T
import utils
import time
import json
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
from time import sleep
from model import FoveaFactory
from nets import NetFactory
import wandb
import torch.utils.data as data_utils


class Model_Wrapper:
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.log_interval = None
            self.tensorboard = None

            self.hidden_sizes = None
            self.activation = None
            self.output_activation = None

            # other
            self.save_model_flag = None
            self.device = None
            self.logdir = None
            self.args_dict = None

            self.name_exp = None
            self.steps = None

            self.region_type = None
            self.regions_radius = None
            self.regions = None
            self.wrapped_arch = None
            self.aggregation_arch = None
            self.aggregation_type = None
            self.motion_needed = None
            self.foa_options = None
            self.foa_flag = None
            self.mode = None

            self.grayscale = None
            self.optimizer = None
            self.lr = None
            self.wandb = None
            self.batch_dim = None
            self.output_channels = 64
            self.kernel = None
            self.dilation = None
            self.num_classes = None
            self.head_dim = None
            self.head_act = None

            # new configs for Foveate layer
            self.reduction_factor = None
            self.reduction_method = None
            self.region_sizes = None
            self.banks = None
            self.new_implementation_fovea = None
            self.total_epochs = None

            # FLOPS
            self.FLOPS_count = None

        def setter(self, args_dict):
            # insert all the  arguments into the attributes of the instance
            # in this way, every command line argument becomes a config property
            for item in args_dict:
                setattr(self, item, args_dict[item])

            if self.hidden_sizes is None:
                self.hidden_sizes = []

            self.args_dict = args_dict

    def __init__(self, config: Config):
        self.config = config

        # to be populated
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None

        if self.config.tensorboard:
            self.config.tb_writer = self.tb_writer = SummaryWriter(
                os.path.join(self.config.logdir, self.config.name_exp))
            with open(os.path.join(self.config.logdir, self.config.name_exp, "command_line_string.txt"),
                      "w") as text_file:
                text_file.write(self.config.string_command_line + " \n")

            self.first_flag_writer = True

        # frame input dims
        self.w = None
        self.h = None

    def __call__(self, dset):

        # handle the dataset info
        self._data_loader(dset)
        if self.config.region_type == "square":
            self.wrapped_model = FoveateNet(self.config).to(self.config.device)
        elif self.config.region_type == "custom":
            self.wrapped_model = FoveateNetCircular(self.config).to(self.config.device)
        else:
            raise NotImplementedError

        self._optimizer(self.config.optimizer)
        # self.wrapped_model.reset_parameters()

        self._criterion()
        self._accuracy()

    def _data_loader(self, dset):  # handle dataset data and metadata

        self.data_tr, self.tar_tr, self.data_gray = dset["trainset"].__getwhole__(self.config.device)

        self.config.w = self.w = dset["trainset"].final_w
        self.config.h = self.h = dset["trainset"].final_h

        self.input_signal = None

    def _criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _accuracy(self):
        self.TrainAccuracy = utils.Accuracy(type="multiclass")
        self.ValidAccuracy = utils.Accuracy(type="multiclass")
        self.TestAccuracy = utils.Accuracy(type="multiclass")

    def _optimizer(self, optim):

        if optim == "sgd":
            self.net_optimizer = torch.optim.SGD(self.wrapped_model.net.parameters(),
                                                 self.config.lr)  # passo campionamento come lr
        elif optim == "adam":
            self.net_optimizer = torch.optim.Adam(self.wrapped_model.net.parameters(),
                                                  self.config.lr)
        else:
            raise NotImplementedError

    def train_step(self, step):
        # self.net_optimizer.zero_grad()
        frame, target, foa_coordinates = self.input_signal.get_next()

        output = self.wrapped_model(frame, foa_coordinates)

        plt.imshow(output[0].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
        plt.title(f"GT: {target.item()}")
        plt.show()

        exit()

        pass

    def test_step(self, step):
        ####  TEST
        self.wrapped_model.eval()
        self.TestAccuracy.reset()
        global_test_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_test, unit="batch") as tepoch:
                for j, (frames, foa, target) in enumerate(tepoch):
                    tepoch.set_description(f"Test Epoch {step}")

                    frames, target = frames.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames)

                    test_loss = self.criterion(output, target)
                    global_test_loss += test_loss.item()

                    self.TestAccuracy.update(output, target)

                tepoch.set_postfix(test_loss=test_loss.item(), accuracy=self.TestAccuracy.compute())
                sleep(0.1)

                print(
                    f"Epoch {step}: Test Average Loss \t {global_test_loss / len(tepoch)}, Test Accuracy: \t {self.TestAccuracy.compute()}")

    def train_loop(self, steps):

        exception_handler = None

        # import time
        # starting = time.time()
        # time_list = []
        #

        # time_list.append(time.time())
        for i in range(steps):  # max learning steps
            # if i % 2500 == 0:
            #     time_list.append(time.time()-starting)
            try:
                if exception_handler:
                    print("\n\n---Handling Keyboard Interrupt---\n")
                    if self.config.save_model_flag:
                        print("Saving model parameters before closing...")
                        self.wrapped_model.save()
                    if self.config.tensorboard:
                        print("Closing Tensorboard writer...")
                        self.tb_writer.close()
                    break
                if self.config.aggregation_arch == "test":
                    train_ret = self.train_step_debug(step=i)
                    continue
                train_ret = self.train_step(step=i)
                self.test_step(step=i)
                if train_ret == -1:
                    return train_ret

            except KeyboardInterrupt:
                print("\n\n---Keyboard Interrupt---\n")
                exception_handler = sys.exc_info()

    def train_valid_test_loop(self, steps):

        exception_handler = None

        # import time
        # starting = time.time()
        # time_list = []
        #

        # time_list.append(time.time())
        for i in range(steps):  # max learning steps
            # if i % 2500 == 0:
            #     time_list.append(time.time()-starting)
            try:
                if exception_handler:
                    print("\n\n---Handling Keyboard Interrupt---\n")
                    if self.config.save_model_flag:
                        print("Saving model parameters before closing...")
                        self.wrapped_model.save()
                    if self.config.tensorboard:
                        print("Closing Tensorboard writer...")
                        self.tb_writer.close()
                    break
                if self.config.wrapped_arch == "test":
                    train_ret = self.train_step_debug(step=i)
                    continue

                train_acc = self.train_step(step=i)
                val_acc = self.val_step(step=i)
                test_acc = self.test_step(step=i)

                # getting the best validation accuracy and others
                if val_acc > self.ValidAccuracy.external_best_acc:
                    self.ValidAccuracy.external_best_acc = val_acc
                    self.TestAccuracy.external_best_acc = test_acc
                    # log into wandb

                if self.config.wandb and (i % self.config.log_interval == 0 or i == self.config.total_epochs - 1):
                    wandb.log({"epoch": i, "best_val_accuracy": self.ValidAccuracy.external_best_acc,
                               "best_test_accuracy": self.TestAccuracy.external_best_acc,
                               }, step=i)

            except KeyboardInterrupt:
                print("\n\n---Keyboard Interrupt---\n")
                exception_handler = sys.exc_info()


class StandardMNISTWrapper(Model_Wrapper):
    def __init__(self, config: Model_Wrapper.Config):
        super(StandardMNISTWrapper, self).__init__(config)

    def _data_loader(self, dset):  # handle dataset data and metadata

        self.config.w = self.w = dset["trainset"].final_w
        self.config.h = self.h = dset["trainset"].final_h

        self.data = DataLoader(dset["trainset"], self.config.batch_dim,
                               shuffle=True, num_workers=self.config.num_workers)

        self.data_test = DataLoader(dset["testset"], self.config.batch_dim,
                                    shuffle=True, num_workers=self.config.num_workers)
        if "valset" in dset:
            self.data_val = DataLoader(dset["valset"], self.config.batch_dim,
                                       shuffle=True, num_workers=self.config.num_workers)

    def __call__(self, dset):

        # handle the dataset info
        self._data_loader(dset)
        self.wrapped_model = NetFactory.createNet(self.config).to(self.config.device)

        self._optimizer(self.config.optimizer)
        # self.wrapped_model.reset_parameters()

        self._criterion()
        self._accuracy()

        mean = torch.tensor([0.1307], dtype=torch.float32)
        std = torch.tensor([0.3081], dtype=torch.float32)
        self.unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def _criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _optimizer(self, optim):

        if optim == "sgd":
            self.net_optimizer = torch.optim.SGD(self.wrapped_model.parameters(),
                                                 self.config.lr)  # passo campionamento come lr
        elif optim == "adam":
            self.net_optimizer = torch.optim.Adam(self.wrapped_model.parameters(),
                                                  self.config.lr)
        else:
            raise NotImplementedError

    def train_step(self, step):
        # in every training epoch loop over all dataset
        self.wrapped_model.train()
        global_train_loss = 0.
        self.TrainAccuracy.reset()

        with tqdm(self.data, unit="batch") as tepoch:
            for j, (frames, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {step}")

                self.net_optimizer.zero_grad()

                frames, target = frames.to(self.config.device), target.to(
                    self.config.device)

                output = self.wrapped_model(frames)

                loss = self.criterion(output, target)

                loss.backward()
                self.net_optimizer.step()

                with torch.no_grad():  # Accuracy computation
                    global_train_loss += loss.item()

                    self.TrainAccuracy.update(output, target)

                    tepoch.set_postfix(loss=loss.item(), accuracy=self.TrainAccuracy.compute())
                    sleep(0.1)

            # print(f"Loss at step {step}: \t {loss.item()}, Accuracy: \t {self.TrainAccuracy.compute()}")

            avg_train_loss = global_train_loss / len(tepoch)
            train_acc = self.TrainAccuracy.compute()

            print(
                f"Epoch {step}: Train Average Loss \t {avg_train_loss}, Train Accuracy: \t {train_acc}")

            if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                frames = self.unnormalize(frames)[0]
                plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                plt.title(f"GT: {target[0].item()}")
                plt.show()
                wandb.log({"epoch": step, "train_loss": avg_train_loss, "train_acc": train_acc,
                           "train_image": wandb.Image(plt)}, step=step)
                plt.close()

            return train_acc

    def val_step(self, step):
        ####  VALIDATION
        self.wrapped_model.eval()
        self.ValidAccuracy.reset()
        global_val_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_val, unit="batch") as tepoch:
                for j, (frames, target) in enumerate(tepoch):
                    tepoch.set_description(f"Val Epoch {step}")

                    frames, target = frames.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames)

                    val_loss = self.criterion(output, target)
                    global_val_loss += val_loss.item()

                    self.ValidAccuracy.update(output, target)

                tepoch.set_postfix(val_loss=val_loss.item(), accuracy=self.ValidAccuracy.compute())
                sleep(0.1)

                avg_val_loss = global_val_loss / len(tepoch)
                val_acc = self.ValidAccuracy.compute()
                print(
                    f"Epoch {step}: Val Average Loss \t {avg_val_loss}, Val Accuracy: \t {val_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, "val_loss": avg_val_loss, "val_acc": val_acc,
                               "val_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return val_acc

    def test_step(self, step):
        ####  TEST
        self.wrapped_model.eval()
        self.TestAccuracy.reset()
        global_test_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_test, unit="batch") as tepoch:
                for j, (frames, target) in enumerate(tepoch):
                    tepoch.set_description(f"Test Epoch {step}")

                    frames, target = frames.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames)

                    test_loss = self.criterion(output, target)
                    global_test_loss += test_loss.item()

                    self.TestAccuracy.update(output, target)

                tepoch.set_postfix(test_loss=test_loss.item(), accuracy=self.TestAccuracy.compute())
                sleep(0.1)

                avg_test_loss = global_test_loss / len(tepoch)
                test_acc = self.TestAccuracy.compute()
                print(
                    f"Epoch {step}: Test Average Loss \t {avg_test_loss}, Test Accuracy: \t {test_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, "test_loss": avg_test_loss, "test_acc": test_acc,
                               "test_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return test_acc


class StandardWrapperBackground(StandardMNISTWrapper):
    def __init__(self, config: Model_Wrapper.Config):
        super(StandardWrapperBackground, self).__init__(config)

    def _data_loader(self, dset):  # handle dataset data and metadata

        self.config.w = self.w = 224
        self.config.h = self.h = 224

        self.data = DataLoader(dset["trainset"], self.config.batch_dim,
                               shuffle=True, num_workers=self.config.num_workers)
        self.data_val = DataLoader(dset["valset"], self.config.batch_dim,
                                   shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_original = DataLoader(dset["test_original"], self.config.batch_dim,
                                             shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_mixedsame = DataLoader(dset["test_mixedsame"], self.config.batch_dim,
                                              shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_mixednext = DataLoader(dset["test_mixednext"], self.config.batch_dim,
                                              shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_mixedrand = DataLoader(dset["test_mixedrand"], self.config.batch_dim,
                                              shuffle=True, num_workers=self.config.num_workers)

    def __call__(self, dset):
        # handle the dataset info
        self._data_loader(dset)
        self.wrapped_model = NetFactory.createNet(self.config).to(self.config.device)

        self._optimizer(self.config.optimizer)
        # self.wrapped_model.reset_parameters()

        self._criterion()
        self._accuracy()

        mean = torch.tensor([0.4717, 0.4499, 0.3837], dtype=torch.float32)
        std = torch.tensor([0.2600, 0.2516, 0.2575], dtype=torch.float32)

        self.unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def train_step_custom(self, step, deactivate_foa=False):
        # in every training epoch loop over all dataset
        self.TrainAccuracy.reset()
        global_train_loss = 0.

        with tqdm(self.data, unit="batch") as tepoch:
            for j, (frames, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {step}")

                self.net_optimizer.zero_grad()

                frames, target = frames.to(self.config.device), target.to(
                    self.config.device)

                output = self.wrapped_model(frames)

                loss = self.criterion(output, target)

                loss.backward()
                self.net_optimizer.step()

                # profiler.step()

                with torch.no_grad():  # Accuracy computation
                    global_train_loss += loss.item()

                    self.TrainAccuracy.update(output, target)

                    tepoch.set_postfix(loss=loss.item(), accuracy=self.TrainAccuracy.compute())
                    sleep(0.001)

            # profiler.export_chrome_trace("profiler/trace.json")

            avg_train_loss = global_train_loss / len(tepoch)
            train_acc = self.TrainAccuracy.compute()
            print(
                f"Epoch {step}: Train Average Loss \t {avg_train_loss}, Train Accuracy: \t {train_acc}")

            if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                frames = self.unnormalize(frames)[0]
                plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy())
                plt.title(f"GT: {target[0].item()}")
                plt.show()
                wandb.log({"epoch": step, "train_loss": avg_train_loss, "train_acc": train_acc,
                           "train_image": wandb.Image(plt)}, step=step)
                plt.close()
            return train_acc

    def val_step_custom(self, step, deactivate_foa=False):
        ####  VALID
        # self.wrapped_model.eval()
        self.ValidAccuracy.reset()
        global_val_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_val, unit="batch") as tepoch:
                for j, (frames, target) in enumerate(tepoch):
                    tepoch.set_description(f"Val Epoch {step}")

                    frames, target = frames.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames)
                    val_loss = self.criterion(output, target)
                    self.ValidAccuracy.update(output, target)

                    global_val_loss += val_loss.item()

                    tepoch.set_postfix(val_loss=val_loss.item(), accuracy=self.ValidAccuracy.compute())
                    sleep(0.001)

                avg_val_loss = global_val_loss / len(tepoch)
                val_acc = self.ValidAccuracy.compute()
                print(
                    f"Epoch {step}: Val Average Loss \t {avg_val_loss}, Val Accuracy: \t {val_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy())

                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, "val_loss": avg_val_loss, "val_acc": val_acc,
                               "val_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return val_acc

    def test_step_custom(self, step, dataset, data_name, deactivate_foa=False):
        ####  TEST
        # self.wrapped_model.eval()
        self.TestAccuracy.reset()
        global_test_loss = 0.
        with torch.no_grad():
            with tqdm(dataset, unit="batch") as tepoch:
                for j, (frames, target) in enumerate(tepoch):
                    tepoch.set_description(f"Set {data_name}; Test Epoch {step}")

                    frames, target = frames.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames)

                    test_loss = self.criterion(output, target)
                    self.TestAccuracy.update(output, target)

                    global_test_loss += test_loss.item()

                    tepoch.set_postfix(test_loss=test_loss.item(), accuracy=self.TestAccuracy.compute())
                    sleep(0.001)

                avg_test_loss = global_test_loss / len(tepoch)
                test_acc = self.TestAccuracy.compute()
                print(
                    f"Set {data_name}; Epoch {step}: Test Average Loss \t {avg_test_loss}, Test Accuracy: \t {test_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy())
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, f"{data_name}_loss": avg_test_loss, f"{data_name}_acc": test_acc,
                               f"{data_name}_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return test_acc

    def train_multiple_test_loop(self, steps, deactivate_foa=False):

        exception_handler = None

        external_best_val_acc = -1
        original_best_test = -1
        mixednext_best_test = -1
        mixedrand_best_test = -1
        mixedsame_best_test = -1

        for i in range(steps):  # max learning steps

            try:
                if exception_handler:
                    print("\n\n---Handling Keyboard Interrupt---\n")
                    if self.config.save_model_flag:
                        print("Saving model parameters before closing...")
                        self.wrapped_model.save()
                    if self.config.tensorboard:
                        print("Closing Tensorboard writer...")
                        self.tb_writer.close()
                    break
                if self.config.wrapped_arch == "test":
                    train_ret = self.train_step_debug(step=i)
                    continue

                train_acc = self.train_step_custom(step=i, deactivate_foa=deactivate_foa)
                val_acc = self.val_step_custom(step=i, deactivate_foa=deactivate_foa)

                test_acc = self.test_step_custom(step=i, dataset=self.dset_test_original, data_name="original",
                                                 deactivate_foa=deactivate_foa)
                test_next = self.test_step_custom(step=i, dataset=self.dset_test_mixednext, data_name="mixednext",
                                                  deactivate_foa=deactivate_foa)
                test_rand = self.test_step_custom(step=i, dataset=self.dset_test_mixedrand, data_name="mixedrand",
                                                  deactivate_foa=deactivate_foa)
                test_same = self.test_step_custom(step=i, dataset=self.dset_test_mixedsame, data_name="mixedsame",
                                                  deactivate_foa=deactivate_foa)

                # getting the best validation accuracy and others
                if val_acc > external_best_val_acc:
                    # log best valid acc
                    external_best_val_acc = val_acc
                    # log corresponding best tests
                    original_best_test = test_acc
                    mixednext_best_test = test_next
                    mixedrand_best_test = test_rand
                    mixedsame_best_test = test_same
                    # log into wandb

                if self.config.wandb and (i % self.config.log_interval == 0 or i == self.config.total_epochs - 1):
                    wandb.log({"epoch": i, "best_val_accuracy": external_best_val_acc,
                               "best_train_accuracy": train_acc,
                               "best_test_accuracy_original": original_best_test,
                               "best_test_accuracy_mixednext": mixednext_best_test,
                               "best_test_accuracy_mixedrand": mixedrand_best_test,
                               "best_test_accuracy_mixedsame": mixedsame_best_test,
                               }, step=i)



            except KeyboardInterrupt:
                print("\n\n---Keyboard Interrupt---\n")
                exception_handler = sys.exc_info()


class MNIST_ModelWrapper(Model_Wrapper):

    def __init__(self, config: Model_Wrapper.Config):
        super(MNIST_ModelWrapper, self).__init__(config)

        mean = torch.tensor([0.1307], dtype=torch.float32)
        std = torch.tensor([0.3081], dtype=torch.float32)

        self.unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def __call__(self, dset):

        # handle the dataset info
        self._data_loader(dset)
        self.wrapped_model = FoveaFactory(self.config).to(self.config.device)

        self._optimizer(self.config.optimizer)

        self._criterion()
        self._accuracy()

    def _optimizer(self, optim):

        params = [self.wrapped_model.parameters()]

        if optim == "sgd":
            self.net_optimizer = torch.optim.SGD(itertools.chain(*params),
                                                 self.config.lr)
        elif optim == "adam":
            self.net_optimizer = torch.optim.Adam(itertools.chain(*params),
                                                  self.config.lr)
        else:
            raise NotImplementedError

    def _data_loader(self, dset):  # handle dataset data and metadata

        self.config.w = self.w = dset["trainset"].final_w
        self.config.h = self.h = dset["trainset"].final_h

        self.data = DataLoader(dset["trainset"], self.config.batch_dim,
                               shuffle=True, num_workers=self.config.num_workers)

        self.data_test = DataLoader(dset["testset"], self.config.batch_dim,
                                    shuffle=True, num_workers=self.config.num_workers)

        if "valset" in dset:
            self.data_val = DataLoader(dset["valset"], self.config.batch_dim,
                                       shuffle=True, num_workers=self.config.num_workers)

    def train_step_debug(self, step):
        # in every training epoch loop over all dataset
        for i, (frames, foa, target) in enumerate(self.data):

            self.net_optimizer.zero_grad()

            # frames, foa, target = frames, foa, target
            frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                self.config.device)

            frames = self.unnormalize(frames)  # unnormalize for visualization
            # output = self.wrapped_model(frames, foa)

            # plot input and foa coordinates
            plt.imshow(frames[0].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
            for j in range(foa.shape[1]):
                plt.plot(foa[0, j, 0].cpu().numpy(), foa[0, j, 1].cpu().numpy(), 'bo', markersize=10, alpha=1)
            plt.show()

            # plot output of fixations for the first sample of the batch
            for fix in range(foa.shape[1]):
                output = self.wrapped_model(frames, foa[:, fix])

                plt.imshow(output[0].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                plt.plot(foa[0, fix, 0].cpu().numpy(), foa[0, fix, 1].cpu().numpy(), 'bo', markersize=10, alpha=1)
                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])
                plt.title(f"GT: {target[0].item()}")
                plt.show()

            exit()

        pass

    def train_step(self, step):
        # in every training epoch loop over all dataset
        self.TrainAccuracy.reset()
        global_train_loss = 0.

        with tqdm(self.data, unit="batch") as tepoch:
            for j, (frames, foa, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {step}")

                self.net_optimizer.zero_grad()

                frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                    self.config.device)

                foveate_output = self.wrapped_model(frames, foa)

                loss = self.criterion(foveate_output, target)

                loss.backward()
                self.net_optimizer.step()

                # profiler.step()

                with torch.no_grad():  # Accuracy computation
                    global_train_loss += loss.item()

                    if self.config.aggregation_arch == "gru_step":
                        self.TrainAccuracy.update(foveate_output[:, :, -1], target[:, -1])
                    else:
                        self.TrainAccuracy.update(foveate_output, target)

                    tepoch.set_postfix(loss=loss.item(), accuracy=self.TrainAccuracy.compute())
                    sleep(0.1)

            # profiler.export_chrome_trace("profiler/trace.json")

            avg_train_loss = global_train_loss / len(tepoch)
            train_acc = self.TrainAccuracy.compute()
            print(
                f"Epoch {step}: Train Average Loss \t {avg_train_loss}, Train Accuracy: \t {train_acc}")

            if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                frames = self.unnormalize(frames)[0]
                plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                for i in range(foa[0].shape[0]):
                    plt.plot(foa[0, i, 0].cpu().numpy(), foa[0, i, 1].cpu().numpy(), 'bo', markersize=10,
                             alpha=1)
                if self.config.aggregation_arch == "gru_step":
                    plt.title(f"GT: {target[0][-1].item()}")
                else:
                    plt.title(f"GT: {target[0].item()}")
                plt.show()
                wandb.log({"epoch": step, "train_loss": avg_train_loss, "train_acc": train_acc,
                           "train_image": wandb.Image(plt)}, step=step)
                plt.close()
            return train_acc

    def val_step(self, step):
        ####  VALID
        # self.wrapped_model.eval()
        self.ValidAccuracy.reset()
        global_val_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_val, unit="batch") as tepoch:
                for j, (frames, foa, target) in enumerate(tepoch):
                    tepoch.set_description(f"Val Epoch {step}")

                    frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames, foa)

                    if self.config.aggregation_arch == "gru_step":
                        val_loss = self.criterion(output[:, :, -1], target)
                        self.ValidAccuracy.update(output[:, :, -1], target)
                    else:
                        val_loss = self.criterion(output, target)
                        self.ValidAccuracy.update(output, target)

                    global_val_loss += val_loss.item()

                    tepoch.set_postfix(val_loss=val_loss.item(), accuracy=self.ValidAccuracy.compute())
                    sleep(0.1)

                avg_val_loss = global_val_loss / len(tepoch)
                val_acc = self.ValidAccuracy.compute()
                print(
                    f"Epoch {step}: Val Average Loss \t {avg_val_loss}, Val Accuracy: \t {val_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                    for i in range(foa[0].shape[0]):
                        plt.plot(foa[0, i, 0].cpu().numpy(), foa[0, i, 1].cpu().numpy(), 'bo', markersize=10,
                                 alpha=1)
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, "val_loss": avg_val_loss, "val_acc": val_acc,
                               "val_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return val_acc

    def test_step(self, step):
        ####  TEST
        # self.wrapped_model.eval()
        self.TestAccuracy.reset()
        global_test_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_test, unit="batch") as tepoch:
                for j, (frames, foa, target) in enumerate(tepoch):
                    tepoch.set_description(f"Test Epoch {step}")

                    frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                        self.config.device)


                    output = self.wrapped_model(frames, foa)

                    if self.config.aggregation_arch == "gru_step":
                        test_loss = self.criterion(output[:, :, -1], target)
                        self.TestAccuracy.update(output[:, :, -1], target)
                    else:
                        test_loss = self.criterion(output, target)
                        self.TestAccuracy.update(output, target)

                    global_test_loss += test_loss.item()

                    tepoch.set_postfix(test_loss=test_loss.item(), accuracy=self.TestAccuracy.compute())
                    sleep(0.1)

                avg_test_loss = global_test_loss / len(tepoch)
                test_acc = self.TestAccuracy.compute()
                print(
                    f"Epoch {step}: Test Average Loss \t {avg_test_loss}, Test Accuracy: \t {test_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                    for i in range(foa[0].shape[0]):
                        plt.plot(foa[0, i, 0].cpu().numpy(), foa[0, i, 1].cpu().numpy(), 'bo', markersize=10,
                                 alpha=1)
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, "test_loss": avg_test_loss, "test_acc": test_acc,
                               "test_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return test_acc


class Offline_ModelWrapper(Model_Wrapper):

    def __init__(self, config: Model_Wrapper.Config):
        super(Offline_ModelWrapper, self).__init__(config)

    def _data_loader(self, dset):  # handle dataset data and metadata

        self.config.w = self.w = dset["trainset"].final_w
        self.config.h = self.h = dset["trainset"].final_h

        self.data = DataLoader(dset["trainset"], batch_size=self.config.batch_dim,
                               shuffle=False, num_workers=self.config.num_workers)

    def train_step(self, step):
        # in every training epoch loop over all dataset
        for i, (frames, foa) in enumerate(self.data):
            frames, foa = frames.to(self.config.device), foa.to(self.config.device)
            output = self.wrapped_model(frames, foa)

            for i in range(output.shape[0]):
                plt.imshow(output[i].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                plt.plot(foa[i, 1].cpu().numpy(), foa[i, 0].cpu().numpy(), 'bo', markersize=10, alpha=1)
                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])
                plt.show()

        pass


########## Task 3 wrapper


class Background_ModelWrapper(MNIST_ModelWrapper):

    def __init__(self, config: Model_Wrapper.Config):
        super(Background_ModelWrapper, self).__init__(config)

        # transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])

        mean = torch.tensor([0.4717, 0.4499, 0.3837], dtype=torch.float32)
        std = torch.tensor([0.2600, 0.2516, 0.2575], dtype=torch.float32)

        self.unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def _data_loader(self, dset):  # handle dataset data and metadata

        self.config.w = self.w = 224
        self.config.h = self.h = 224

        self.data = DataLoader(dset["trainset"], self.config.batch_dim,
                               shuffle=True, num_workers=self.config.num_workers)
        self.data_val = DataLoader(dset["valset"], self.config.batch_dim,
                                   shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_original = DataLoader(dset["test_original"], self.config.batch_dim,
                                             shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_mixedsame = DataLoader(dset["test_mixedsame"], self.config.batch_dim,
                                              shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_mixednext = DataLoader(dset["test_mixednext"], self.config.batch_dim,
                                              shuffle=True, num_workers=self.config.num_workers)
        self.dset_test_mixedrand = DataLoader(dset["test_mixedrand"], self.config.batch_dim,
                                              shuffle=True, num_workers=self.config.num_workers)

    def train_step_debug(self, step):
        # in every training epoch loop over all dataset
        for i, (frames, foa, target) in enumerate(self.data):

            self.net_optimizer.zero_grad()

            # frames, foa, target = frames, foa, target
            frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                self.config.device)

            frames = self.unnormalize(frames)  # unnormalize for visualization
            # output = self.wrapped_model(frames, foa)

            # plot input and foa coordinates
            plt.imshow(frames[0].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
            plt.plot(foa[0, 1].cpu().numpy(), foa[0, 0].cpu().numpy(), 'bo', markersize=10,
                     alpha=1)

            # plot output of fixations for the first sample of the batch
            for fix in range(foa.shape[1]):
                output = self.wrapped_model(frames, foa[:, fix])

                plt.imshow(output[0].detach().cpu().permute(1, 2, 0).numpy(), cmap='gray', vmin=0., vmax=1.)
                plt.plot(foa[0, fix, 0].cpu().numpy(), foa[0, fix, 1].cpu().numpy(), 'bo', markersize=10, alpha=1)
                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])
                plt.title(f"GT: {target[0].item()}")
                plt.show()

            exit()

        pass

    def train_step_custom(self, step, deactivate_foa=False):
        # in every training epoch loop over all dataset
        self.TrainAccuracy.reset()
        global_train_loss = 0.

        with tqdm(self.data, unit="batch") as tepoch:
            for j, (frames, foa, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {step}")

                self.net_optimizer.zero_grad()

                frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                    self.config.device)

                foveate_output = self.wrapped_model(frames, foa)

                loss = self.criterion(foveate_output, target)

                loss.backward()
                self.net_optimizer.step()

                # profiler.step()

                with torch.no_grad():  # Accuracy computation
                    global_train_loss += loss.item()

                    self.TrainAccuracy.update(foveate_output, target)

                    tepoch.set_postfix(loss=loss.item(), accuracy=self.TrainAccuracy.compute())
                    sleep(0.001)

            # profiler.export_chrome_trace("profiler/trace.json")

            avg_train_loss = global_train_loss / len(tepoch)
            train_acc = self.TrainAccuracy.compute()
            print(
                f"Epoch {step}: Train Average Loss \t {avg_train_loss}, Train Accuracy: \t {train_acc}")

            if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                frames = self.unnormalize(frames)[0]
                plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy())
                plt.plot(foa[0, 1].cpu().numpy(), foa[0, 0].cpu().numpy(), 'bo', markersize=10,
                         alpha=1)
                plt.title(f"GT: {target[0].item()}")
                plt.show()
                wandb.log({"epoch": step, "train_loss": avg_train_loss, "train_acc": train_acc,
                           "train_image": wandb.Image(plt)}, step=step)
                plt.close()
            return train_acc

    def val_step_custom(self, step, deactivate_foa=False):
        ####  VALID
        # self.wrapped_model.eval()
        self.ValidAccuracy.reset()
        global_val_loss = 0.
        with torch.no_grad():
            with tqdm(self.data_val, unit="batch") as tepoch:
                for j, (frames, foa, target) in enumerate(tepoch):
                    tepoch.set_description(f"Val Epoch {step}")

                    frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames, foa)
                    val_loss = self.criterion(output, target)
                    self.ValidAccuracy.update(output, target)

                    global_val_loss += val_loss.item()

                    tepoch.set_postfix(val_loss=val_loss.item(), accuracy=self.ValidAccuracy.compute())
                    sleep(0.001)

                avg_val_loss = global_val_loss / len(tepoch)
                val_acc = self.ValidAccuracy.compute()
                print(
                    f"Epoch {step}: Val Average Loss \t {avg_val_loss}, Val Accuracy: \t {val_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy())
                    plt.plot(foa[0, 1].cpu().numpy(), foa[0, 0].cpu().numpy(), 'bo', markersize=10,
                             alpha=1)
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, "val_loss": avg_val_loss, "val_acc": val_acc,
                               "val_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return val_acc

    def test_step_custom(self, step, dataset, data_name, deactivate_foa=False):
        ####  TEST
        # self.wrapped_model.eval()
        self.TestAccuracy.reset()
        global_test_loss = 0.
        with torch.no_grad():
            with tqdm(dataset, unit="batch") as tepoch:
                for j, (frames, foa, target) in enumerate(tepoch):
                    tepoch.set_description(f"Set {data_name}; Test Epoch {step}")

                    frames, foa, target = frames.to(self.config.device), foa.to(self.config.device), target.to(
                        self.config.device)

                    output = self.wrapped_model(frames, foa)

                    test_loss = self.criterion(output, target)
                    self.TestAccuracy.update(output, target)

                    global_test_loss += test_loss.item()

                    tepoch.set_postfix(test_loss=test_loss.item(), accuracy=self.TestAccuracy.compute())
                    sleep(0.001)

                avg_test_loss = global_test_loss / len(tepoch)
                test_acc = self.TestAccuracy.compute()
                print(
                    f"Set {data_name}; Epoch {step}: Test Average Loss \t {avg_test_loss}, Test Accuracy: \t {test_acc}")

                if self.config.wandb and (step % self.config.log_interval == 0 or step == self.config.total_epochs - 1):
                    frames = self.unnormalize(frames)[0]
                    plt.imshow(frames.detach().cpu().permute(1, 2, 0).numpy())
                    plt.plot(foa[0, 1].cpu().numpy(), foa[0, 0].cpu().numpy(), 'bo', markersize=10,
                             alpha=1)
                    plt.title(f"GT: {target[0].item()}")
                    plt.show()
                    wandb.log({"epoch": step, f"{data_name}_loss": avg_test_loss, f"{data_name}_acc": test_acc,
                               f"{data_name}_image": wandb.Image(plt)}, step=step)
                    plt.close()
                return test_acc

    def train_multiple_test_loop(self, steps, deactivate_foa=False):

        exception_handler = None

        external_best_val_acc = -1
        original_best_test = -1
        mixednext_best_test = -1
        mixedrand_best_test = -1
        mixedsame_best_test = -1

        for i in range(steps):  # max learning steps

            try:
                if exception_handler:
                    print("\n\n---Handling Keyboard Interrupt---\n")
                    if self.config.save_model_flag:
                        print("Saving model parameters before closing...")
                        self.wrapped_model.save()
                    if self.config.tensorboard:
                        print("Closing Tensorboard writer...")
                        self.tb_writer.close()
                    break
                if self.config.wrapped_arch == "test":
                    train_ret = self.train_step_debug(step=i)
                    continue

                train_acc = self.train_step_custom(step=i, deactivate_foa=deactivate_foa)
                val_acc = self.val_step_custom(step=i, deactivate_foa=deactivate_foa)

                test_acc = self.test_step_custom(step=i, dataset=self.dset_test_original, data_name="original",
                                                 deactivate_foa=deactivate_foa)
                test_next = self.test_step_custom(step=i, dataset=self.dset_test_mixednext, data_name="mixednext",
                                                  deactivate_foa=deactivate_foa)
                test_rand = self.test_step_custom(step=i, dataset=self.dset_test_mixedrand, data_name="mixedrand",
                                                  deactivate_foa=deactivate_foa)
                test_same = self.test_step_custom(step=i, dataset=self.dset_test_mixedsame, data_name="mixedsame",
                                                  deactivate_foa=deactivate_foa)

                # getting the best validation accuracy and others
                if val_acc > external_best_val_acc:
                    # log best valid acc
                    external_best_val_acc = val_acc
                    # log corresponding best tests
                    original_best_test = test_acc
                    mixednext_best_test = test_next
                    mixedrand_best_test = test_rand
                    mixedsame_best_test = test_same
                    # log into wandb

                if self.config.wandb and (i % self.config.log_interval == 0 or i == self.config.total_epochs - 1):
                    wandb.log({"epoch": i, "best_val_accuracy": external_best_val_acc,
                               "best_train_accuracy": train_acc,
                               "best_test_accuracy_original": original_best_test,
                               "best_test_accuracy_mixednext": mixednext_best_test,
                               "best_test_accuracy_mixedrand": mixedrand_best_test,
                               "best_test_accuracy_mixedsame": mixedsame_best_test,
                               }, step=i)


            except KeyboardInterrupt:
                print("\n\n---Keyboard Interrupt---\n")
                exception_handler = sys.exc_info()
