# ECML-PKDD 2022- **Foveated Neural Computation**

This repository contains the code and data  for the paper [**Foveated Neural Computation**](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_620.pdf), accepted for publication at the ECML-PKDD 2022 conference. 

*Authors:*  [Matteo Tiezzi](https://mtiezzi.github.io/), Simone Marullo,  Alessandro Betti, Enrico Meloni, Lapo Faggi, Marco Gori and Stefano Melacci.

DOI: [**Foveated Neural Computation**](https://link.springer.com/chapter/10.1007/978-3-031-26409-2_2)

_Notice that reproducibility is not guaranteed by PyTorch across different releases, platforms, hardware. Moreover,
determinism cannot be enforced due to use of PyTorch operations for which deterministic implementations do not exist
(e.g. bilinear upsampling)._

Make sure to have Python dependencies by running:
```
pip install -r requirements.txt
```

We tested the code with PyTorch 1.10.
Follow the [instructions](https://pytorch.org/get-started/) on the official website for further details.


QUICK START: DEFINING Foveated Convolutional Layers (FCLs)
--------------------------
Have a look at the [Colab Notebook](https://github.com/sailab-code/foveated_neural_computation/blob/main/foveated_convolutional_layer.ipynb) for a complete example on how to define and use all the Foveated Convolutional Layers!

A very tiny example. If you would create a Conv2d layer in PyTorch as follows:



    import torch     
    h = 224
    w = 224
    in_channels = 3
    out_channels = 4
    kernel_size = 7
    device = "cpu"
    
    image = torch.rand((1, 3, h, w)) # some random input tensor
    net = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, padding_mode='zeros')
    out = net(image)


Then, you can easily define and use a Piecewise-FCL as follows:




    # set the FOA coordinates to the center of the image
    foa_xy = torch.tensor([[h // 2, w // 2]], dtype=torch.long)
    net = FovConv2dReg(in_channels, out_channels, kernel_size, 
                        region_type="circle", method="downscaling",
                        padding_mode='zeros', region_sizes=(151, -1), 
                        reduction_factors=(1.0, .5), banks="shared")
    out = net(image, foa_xy)


Have a look at the [Colab Notebook](https://github.com/sailab-code/foveated_neural_computation/blob/main/foveated_convolutional_layer.ipynb) for a complete example on how to define and use all the Foveated Convolutional Layers : Gaussian Modulated (GM-FCL), Neural Modulated/Generated (NM/NG-FCLs)!


REPOSITORY DESCRIPTION
----------------------

We provide the source code in a zip archive, once extracted the folder structure is the following:

    data :                      folder with dataset and utilities to generate datasets
    fov_conv2d_reg.py :         implementation of Pw-FCL 
    fov_conv2d_cont.py :        implementation of Nm/Ng/Gm-FCLs 
    fovlayer_time.py :          script for the inference time comparisons 
    model.py :                  wrapper classes for the underlying CNNs/FCLs
    model_wrapper.py :          wrapper classes for training/validation/test boilerplate
    utils.py :                  code utilities
    nets.py:                    NN architectures definitions
    requirements.txt :          python packages requirements
    geymol.py :                 extended library of the trajectory prediction model from Zanca et al. (2020) 
    mnist_fcl.py :              script for running the experiments on Dual-Intention and Stock-Fashion with FCLs
    mnist_cnn.py :              script for running the experiments on Dual-Intention and Stock-Fashion with CNNs
    background_cnn.py :         script for running the experiments on Background spurious correlation with CNNs
    background_fcl.py :         script for running the experiments on Background spurious correlation with FCLs
    task3_utilities :           code utilities for the experiments on Background spurious correlation
    stream_attention.py :       script for running the experiments on the stream of visual stimuli
    stream_attention_nets.py :  utilities for neural architectures
    stream_attention_utils.py : utilities   
    


HOW TO RUN AN EXPERIMENT
========================

We provide details on how to run the experiments from the main paper. 

Inference Times 
---------------

Launch the `fovlayer_time.py` script to measure and compare the inference times of the several implementations of Pw-FCL
as reported in the paper (Section 3 and supplemental material).  


Dual-Intention 
--------------
The dataset is available in the `data` folder. We provide both the `1K` (low data regime) and `10K` version of the data,
with the following folder names:

 - `MNIST_28__MNIST_28-biased-size_200_examples1000`
 - `MNIST_28__MNIST_28-biased-size_200_examples10000`


To run the experiments on the Dual-Intention dataset, launch  `mnist_cnn.py`  for the standard CNN models or use
`mnist_fcl.py` for the FCLs.
In the following, we report the description of the script arguments that allow to control the hyperpameters.
For categorical arguments, we report the available options with `choices={option1, option2, etc.}`.
When the argument has to be a list of element, we denote it with `list=[el1, el2, ..., eln]`.

### Standard CNNs
 
Use the `mnist_cnn.py` script.

Available arguments:

    usage: mnist_cnn.py [-h] [-dataset [dataset]] [-lr [dt]] [-opt OPTIMIZER] [-wrapped_arch WRAPPED_ARCH] 
      [-aggregation_type AGGREGATION_TYPE] [--grayscale] [-id [id]] [-save] [-logdir [logdir]] [--no_cuda] 
      [--cuda_dev CUDA_DEV] [--n_gpu_use N_GPU_USE] [--log_interval N] [--tensorboard] [--fixed_seed FIXED_SEED] 
      [--seed S] [--batch_dim batch] [--num_workers S] [--output_channels out_channels] [--kernel kernel_size]
      [--num_classes number of classes] [--total_epochs number of epochs] [--wandb WANDB] [--FLOPS_count]


#### Argument description:

Here we describe only the arguments that must be customized to run an experiment, along with the possible parameter choices
      
      --dataset             Dataset from the intention-MNIST or stock_MNIST family;
                              choices={MNIST_28__MNIST_28-biased-size_200_examples1000,
                              MNIST_28__MNIST_28-biased-size_200_examples10000}
      --lr                  model learning rate
      --wrapped_arch        Architectures;  choices={vanilla, CNN_one_layer_custom, CNN_one_layer_maxpool}
      --aggregation_type    Region pooling function aggregation; choices={mean, max}
      --fixed_seed          Flag to set the fixed seed; choices={"true", "false"}
      --seed                specify seed 
      --batch_dim           Batch dims
      --num_workers         specify number of workers
      --output_channels     Output channels of the layer                           
      --kernel              Weight Kernel size
      --total_epochs        Number of epochs

The computed metrics of each experiment are printed on screen in the console.

We report here a command line example to launch the script:

    python mnist_cnn.py --aggregation_type=max --batch_dim=32 --dataset=MNIST_28__MNIST_28-biased-size_200_examples1000 --fixed_seed=True --kernel=29 --lr=0.001 --num_classes=10 --output_channels=64 --seed=123 --total_epochs=100 --wrapped_arch=CNN_one_layer_custom

###  FCLs
 
Use the `mnist_fcl.py` script.

### Argument description:

Here we describe only the arguments that must be customized to run an experiment, along with the possible parameter choices
      
      --dataset             Dataset from the intention-MNIST or stock_MNIST family;  
                              choices={MNIST_28__MNIST_28-biased-size_200_examples1000,
                              MNIST_28__MNIST_28-biased-size_200_examples10000}
      --region_sizes        List of region diameters;  -1 in last position means that the outmost region considers the remaining pixels
                            list=[33, -1]
      --reduction_factor    Reduction factor for the outmost region (here we consider FCL with R=2)
      --reduction_method    Reduction method for foveated regions; choices={"downscaling", "dilation", "stride", "vanilla"}'
      --region_type         Shape of foveated regions; choices={box, circle}
      --banks               Type of filter banks; choices={"independent", "shared"}
      --lr                  model learning rate
      --wrapped_arch        Architectures; single choice here, the FCL choices={fnn_reg}
      --aggregation_type    Region pooling function aggregation;  choices={mean, max}
      --fixed_seed          Flag to set the fixed seed; choices={"true", "false"}
      --seed                specify seed 
      --batch_dim           Batch dims
      --num_workers         specify number of workers
      --output_channels     Output channels of the layer                           
      --kernel              Weight Kernel size
      --total_epochs        Number of epochs

The computed metrics of each experiment are printed on screen in the console.

We report here a command line example to launch the script:

    python mnist_fcl.py --banks=shared --batch_dim=32 --dataset=MNIST_28__MNIST_28-biased-size_200_examples10000 --fixed_seed=True --kernel=15 --lr=0.001 --output_channels=64 --reduction_factor=0.5 --reduction_method=downscaling --region_type=box --seed=1234567 --total_epochs=100  --wrapped_arch=fnn_reg


Stock-Fashion
--------------
The dataset is available in the `data` folder. We provide both the `1K` (low data regime) and `10K` version of the data,
with the following folder names:

 - `FashionMNIST_28__MNIST_28-biased-size_200_examples1000`
 - `FashionMNIST_28__MNIST_28-biased-size_200_examples10000`

To run the experiments on the Dual-Intention dataset, launch the `mnist_cnn.py`  for the standard CNN models or use
`mnist_fcl.py` for the FCLs.

The same details/arguments/parameters of the previous experiment hold, with the following exceptions:


    --dataset             Dataset from the intention-MNIST or stock_MNIST family;  
                                choices={FashionMNIST_28__MNIST_28-biased-size_200_examples1000,
                                FashionMNIST_28__MNIST_28-biased-size_200_examples10000}
    --num_classes         Number of total target classes of the task; Must be set to 20 choices={20}


The computed metrics of each experiment are printed on screen in the console.

We report here a command line to launch the script in the case of CNNs:

    python mnist_cnn.py --batch_dim=32 --dataset=FashionMNIST_28__MNIST_28-biased-size_200_examples10000 --fixed_seed=True --lr=0.001 --num_classes=20 --seed=123 --total_epochs=100  --wrapped_arch=vanilla


We report here a command line example to launch the script in the case of FCLs:

    python mnist_fcl.py  --banks=independent --batch_dim=32 --dataset=FashionMNIST_28__MNIST_28-biased-size_200_examples10000 --fixed_seed=True --head_dim=128 --kernel=29 --lr=0.001 --num_classes=20  --reduction_factor=0.25 --reduction_method=downscaling --region_type=box --seed=1234567 --total_epochs=100  --wrapped_arch=fnn_reg


Background Spurious correlations
--------------------------------

We provide the [preprocessed data for this task at this  link](https://drive.google.com/file/d/1ARvf0SkA9a0K_s0KPKWDOi-QL9kVGrRd/view?usp=sharing) - please unzip the data in folder `data/task3`. The subfolder `original_preprocessed` contains the 
training data preprocessed into a resolution of 224x224 and the train/validation index 
(ratio validation size= 20% training size).  The folder `original_foa_center` and `val_foa_center` contains the object 
center location for each  sample of the train and validation set (same filename as the ones in `original_preprocessed`). 
The `original_preprocessed`  is the test split from the [Background challenge](https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz)


To run the experiments on the Dual-Intention dataset, launch the `background_cnn.py`  for the standard CNN models or use
`background_fcl.py` for the FCLs.
In the following, we report the description of the script arguments that allow to control the hyperpameters.
For categorical arguments, we report the available options with `choices={option1, option2, etc.}`.
When the argument have to be a list of element, we denote it with `list=[el1, el2, ..., eln]`.

### Standard CNNs
 
Use the `background_cnn.py` script. We test the CNN* model (`vanilla` in the code).

Available arguments:

    usage: background_cnn.py [-h] [-dataset [dataset]] [-lr [dt]] [-opt OPTIMIZER] [-wrapped_arch WRAPPED_ARCH] 
      [-aggregation_type AGGREGATION_TYPE] [--grayscale] [-id [id]] [-save] [-logdir [logdir]] [--no_cuda] 
      [--cuda_dev CUDA_DEV] [--n_gpu_use N_GPU_USE] [--log_interval N] [--tensorboard] [--fixed_seed FIXED_SEED] 
      [--seed S] [--batch_dim batch] [--num_workers S] [--output_channels out_channels] [--kernel kernel_size]
      [--num_classes number of classes] [--total_epochs number of epochs] [--wandb WANDB] [--FLOPS_count]


#### Argument description:

Here we describe only the arguments that must be customized to run an experiment, along with the possible parameter choices

      --lr                  model learning rate
      --wrapped_arch        Architectures vanilla;  choices={vanilla}
      --aggregation_type    Region pooling function aggregation; choices={mean, max}
      --fixed_seed          Flag to set the fixed seed; choices={"true", "false"}
      --seed                specify seed 
      --batch_dim           Batch dims
      --num_workers         specify number of workers
      --output_channels     Output channels of the layer                           
      --kernel              Weight Kernel size
      --total_epochs        Number of epochs

The computed metrics of each experiment are printed on screen in the console.

We report here a command line example to launch the script:

    python background_cnn.py --aggregation_type=mean --batch_dim=16 --dataset=background --fixed_seed=True --lr=0.0001 --num_classes=3  --seed=1234567 --total_epochs=100  --wrapped_arch=vanilla


###  FCLs
 
Use the `background_fcl.py` script. In this case, we built some variants of the CNN* by injecting FCL in first, last or 
all the layers. In the case of Pw-FCLs, the reduction factor and region size are fixed (described in the paper 
and implemented in the `nets.py` file). Notice that in the case of injecting a Pw-FCL into the last layer, we aggregate
the extracted information via the piece-wise pooling. We specify this in the code through `--aggregation_arch task3`.
In all the other settings (Pw-FCL only in the first layer, or other FCL models like Gm/Nm/Ng-FCLs) the average pooling
operation acts frame-wise (`--aggregation_arch none`). 

You can specify the network variant via the  `--wrapped_arch  variant`  argument. Name structure: 

    {fcl_layer_position}_{fcl_variant} 

where `fcl_layer_position = {first, last, all}` and `fcl_variant = {FL, gaussian, netmodulated, netgenerated}`.
Notice that here we provide the object location as an additional info to the FCLs. In the next section we will describe 
the arguments to run the same experiment without providing this location.


#### Argument description:

Here we describe only the arguments that must be customized to run an experiment, along with the possible parameter choices

      --reduction_factor    Reduction factor for the outmost region (here we consider FCL with R=2)
      --reduction_method    Reduction method for foveated regions; choices={"downscaling", "dilation", "stride", "vanilla"}'
      --region_type         Shape of foveated regions; choices={box, circle}
      --banks               Type of filter banks; choices={"independent", "shared"}
      --lr                  model learning rate
      --wrapped_arch        Architectures;  choices={first_FL, all_FL, last_FL, first_FL_gaussian, last_FL_gaussian,
                            all_FL_gaussian, first_FL_netmodulated, last_FL_netmodulated, all_FL_netmodulated,
                            first_FL_netgenerated, last_FL_netgenerated}
      --aggregation_type    Region pooling function aggregation;  choices={mean, max}
      --aggregation_arch    Type of aggregation for the MLP head;  If we use an FCL in the last layer, then 
                            we must specify the "task3" option here. Otherwise if the FCL is only in the first layer
                            this argument has to be "none";  choices={task3, none}
      --fixed_seed          Flag to set the fixed seed; choices={"true", "false"}
      --seed                specify seed 
      --batch_dim           Batch dims
      --num_workers         specify number of workers
      --output_channels     Output channels of the layer                           
      --kernel              Weight Kernel size
      --total_epochs        Number of epochs

The computed metrics of each experiment are printed on screen in the console.

We report here some command line examples to launch the script:

    python background_fcl.py --aggregation_arch=none --banks=independent --batch_dim=16 --dataset=background --fixed_seed=True --lr=0.0001 --num_classes=3 --reduction_method=downscaling --region_type=circle --seed=123 --total_epochs=100  --wrapped_arch=first_FL
    python background_fcl.py --aggregation_arch=task3 --aggregation_type=mean --banks=independent --batch_dim=16 --dataset=background --fixed_seed=True --lr=0.0001 --num_classes=3 --reduction_method=downscaling --region_type=circle --seed=123 --total_epochs=100  --wrapped_arch=last_FL
    python background_fcl.py --aggregation_arch=none --batch_dim=16 --dataset=background --fixed_seed=True --lr=0.0001 --num_classes=3  --seed=1234567 --total_epochs=100 --wrapped_arch=first_FL_netmodulated


### Experiment with FOA in the center of the frame

We also run the same experiment without providing the object location to FCLs. 
This can be simply done via specifying the following argument:

    -- foa_centered "true"

Hence, a command line can be provided as follows:

    python background_fcl.py --aggregation_arch=task3 --aggregation_type=mean --banks=independent --batch_dim=16 --dataset=background --fixed_seed=True --foa_centered=true --lr=0.0001 --num_classes=3  --reduction_method=downscaling --region_type=circle --seed=1234567 --total_epochs=100  --wrapped_arch=all_FL


### Experiment with 9 classes

To run experiments considering the full dataset, simply specify `--num_classes=9`





###  Stream of visual stimuli

To run the experiments on the visual attention control in a stream of frames, launch the `stream_attention.py`.
The script will download, at the first execution, all the necessary data and will save it into an apposite folder.
At the end of the execution, the script will produce results and plots into the folder `results_exp_pane`.
See the file `stream_attention_nets.py` for more details on the neural architectures. 

Available arguments:


    usage: stream_attention.py [-h] [--lr LR] [--net NET] [--w W] [--fixed_seed FIXED_SEED] [--seed SEED] 
        [--print_every PRINT_EVERY] [--eval EVAL] [--wandb WANDB]

#### Argument description:


      --lr                  model learning rate
      --net                 the architecture used for the feature extraction; choices={cnn, fov_reg_all, fov_reg_last,
                            fov_net_all, fov_net_last, fov_gaussian_all, fov_gaussian_last}
      --w                   frame width  (in the paper we tested with w= 200 and w= 1000)
      --fixed_seed          Flag to set the fixed seed; choices={"true", "false"}
      --seed                specify seed 
      --eval                Flag to specify only evaluation mode


We report here some command line examples to launch the script (at the first run, the script will create the dataset):

    python stream_attention.py --fixed_seed=True --net=fov_gaussian_all --seed=1234567 --w 200
    python stream_attention.py --fixed_seed=True --net=cnn --seed=1234567 --w 200


Differences with respect to the paper text
------------------------------------------

- The `Dual-Intention` dataset from the paper is here referred to as `MNIST_28__MNIST_28-biased-size_200_examples{K}`
  where `{K}`  assumes a value in {1000, 10000}.
- The CNN* model from the paper is here `vanilla`  
- `CNN_one_layer_custom` is the standard CNN layer
- `CNN_one_layer_maxpool` is the standard CNN layer followed by maxpooling


Acknowledgement
---------------

This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).


