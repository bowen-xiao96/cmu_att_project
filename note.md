# Note on recurrent network project

Bowen Xiao, 09/17/2018

## Code structure

Code for dataset definition, model definition, experiment runner and auxiliary utilities are separated.

* Dataset definition

  contains dataset preprocessing and PyTorch `Dataloader` code.

  * dataset/cifar
  * dataset/cub
  * dataset/imagenet

  `get_<dataset_name>_dataset.get_dataloader` function will return both training and testing `Dataloader` objects

* Model definition

  * Main model: `multiple_recurrent_l.py`

    Defines a VGG16 network structure with multiple recurrent connections attached. For detailed information, please directly look into it, since it is heavily commented. It accepts a list as parameter specifying the scheme of the recurrent connections, and is thus very flexible.

    Feel free to modify the structure of this model. We need to do experiments on different gating structures on ImageNet.

    Input: ImageNet Images

    Output: prediction made at the end of the network at each unrolling step

  * `multiple_recurrent_s.py`

    Similar model, but works on CIFAR dataset

  * `multiple_recurrent_newloss.py`

    Similar model, but with loss function embedded in the model, which makes GPU memory usage more balanced. Moreover, it supports different kinds of loss functions, including loss at each time step, loss at final time step and refinement loss.

  * `resnet18_recurrent.py`

    The backbone of this network is ResNet-18 instead of VGG16. Stilling working on it. For ResNet models that have already worked on CIFAR dataset, please refer to Siming's code.

* Experiments

  * `train_multiple_recurrent_l.py`

    Train the `multiple_recurrent_l.py` model on ImageNet with preloaded weights. The logics in this model is very straightforward and feel free to modify the hyperparameters in it. Already trained models can be found at `/data2/bowen/attention/pay_attention/experiments/multiple_recurrent_l`, and most of our supporting experiments are done on that.

  * `train_multiple_recurrent_s.py`

    Train the CIFAR model

  * Supporting experiments

    * Please refer to `adversarial_noise`, `familiarity_effect`, `feature_analysis`, `fine_grained`, `noise`, `occlusion`, `surround_suppress` directories for the respective code of each supporting experiment. They are all very short and straightforward.

      Note that we use FGSM method to do adversarial attack, and for ordinary noise test, we use standard Gaussian noise added to images.

* Auxiliary utilities

  * `Trainer.py`

    A framework for training models and recording logs

  * `model_tools.py` and `metric.py`

    Defines some functions to initialize the model and calculate accuracy, etc.

  * `Padam.py`

    An Adam-like optimizer with better generalization abilities, which can sometimes beat original Adam. Please see https://github.com/thughost2/Padam

