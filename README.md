# Federated-Learning-with-Differential Privacy
[NTU MSAI Research Project]

This repo is built upon [AshwinRJ](https://github.com/AshwinRJ/Federated-Learning-PyTorch) 's  implementation of the [vanilla federated learning paper](https://arxiv.org/abs/1602.05629) on image classificaton task of MNIST dataset. Based on this, I fixed the problem on running FL experiment with CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Main component of this repo that I added, is the more complex task of image segmentation implementated with federated learning, with  option of IID and non-IID(equal and unequal data distribution), especially the option of training with differential privacy (DP) with Opacus library.

Many arguments are provided with this code for tuning all the necessary hyperparameters for training image segmentation with federated learning and differential privacy. Details are included in the `Options` section below.

## Requirements
Install all the packages from requirements.txt
* Python3
* Pytorch
* Torchvision
* Opacus
* Pycocotools

## Data
* The data for image classification: MNIST and CIFAR10 will be automatically downloaded into the `data` subfolder when runing the code in `classification' part.
* For image segmentation, COCO is the default dataset used, the image and annotation data  has to be manual downloaded in the `data/coco` before runing experiments on segmentation. Please follow the instructions in `data/README.md`.

To test other dataset for segmentation, the data needs to be processed into torchvision.datasets.CocoDetection format. Please refer to the COCO [website](https://cocodataset.org/#download) on how to handle the data. 

## Running the experiments
Code will automatically run on gpu if gpu and cuda is available on the machine. 

## Classification

* To run the baseline (no federated learning) experiment with MNIST (or CIFAR) on MLP:
```
python classification/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
-----

* To run the federated experiment with CIFAR on CNN (IID):
```
python classification/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=10
```
* To run the same experiment under non-IID equal data distribution condition:
```
python classification/federated_main.py --model=cnn --dataset=cifar --iid=0 --epochs=10
```
* To run the same experiment under non-IID unequal data distribution condition:
```
python classification/federated_main.py --model=cnn --dataset=cifar --iid=0 --unequal=1 --epochs=10
```

## Segmentation
### Training
* To run the baseline experiment with COCO (train2017 or val2017 dataset) on lraspp_mobilenetv3 model.
```
python segmentation/baseline_main.py --data=val2017 --model=lraspp_mobilenetv3 --num_classes=21 --epochs=10 --num_workers=4 --save_frequency=1  
```

* To run the federated learning experiments.
```
python segmentation/federated_main.py --model=lraspp_mobilenetv3  --num_classes=21 --activation=tanh --weight=0.5 --pretrained --iid=1 --unequal=0 --local_bs=8 --num_workers=4 --save_frequency=1 --data=val2017 --num_users=100 --local_test_frac=0.1 --local_ep=5  --epochs=5
```
* To run the federaated learning experiments with differential privacy.

(As Opacus is a developing library, the output generated many warnings during runtime, so I suggest ignore warning option as below in the command line.)
```
python -W ignore segmentation/federated_main.py --model=lraspp_mobilenetv3 --num_classes=21 --pretrained  --iid=1 --unequal=0 --local_bs=8 --virtual_bs=8 --num_workers=1 --save_frequency=1 --data=val2017 --num_users=50 --local_test_frac=0.1 --no_dropout --lr=0.01 --activation=tanh --weight=0.5 --noise_multiplier=0.2 --max_grad_norm=5 --dp  --local_ep=5  --epochs=5
```

### Inference
* To inference segmentation model checkpoint on images, first save the checkpoint and image in the folder same as the `root` argument before running below:
```
==model trained without differential privacy
python segmentation/inference.py --activation=relu --root=./test --checkpoint=checkpoint-relu.pth --filename=image1.jpg
==model trained with differential privacy
python segmentation/inference.py --activation=tanh --root=./test --checkpoint=checkpoint-dp-tanh.pth --dp --filename=image1.jpg
```
Do note that for model trained with `dp` and `tanh` activation function, the appropriate arguments should be used to create the model with the same structure as the checkpoint. It's because that during DP training and according to my experiments, model is required to use Group Normalization for privacy preservation and tanh activation instead of default relu for best performance. 

### Experiments
Logs, training curve and checkpoint will be saved in `save` folder when run.
The checkpoint and logs are named as below according to experiment conditions:
```
fedDP_val2017_lraspp_mobilenetv3_c21_e5_C[0.1]_iid[0]_uneq[0]_E[5]_B[8v8]_lr0.01_noise0.2_norm5.0_w0.5_tanh_n100.pth
```
Above checkpoint shows the experiment condition as: `federared learning with DP, lraspp_mobilenetv3 model, 21 classes, 5 rounds, 0.1 fraction of clients, non-iid and equal distribution, 5 local epochs, batch size is 8 and virtual batch_size(DP param)  is 8, learning rate of 0.01, noise multiplier=0.2, max_grad_norm for sample gradient clipping is 5.0, background class loss weight is 0.5, tanh activation function, 100 number of clients.`

For details can refer to `get_exp_name()` in segmentation/federated_main.py for interpretation. 

## Options for Segmentation
Below shows the options for training image segmentation with federated learning and differential privacy, options for image classification task is much simpler and should refer in the `classification/options.py`.

The default values for various paramters parsed to the experiment are given in ```segmentation/options.py```. Details are as below:

```
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of rounds of training
  --num_users NUM_USERS
                        number of users: K
  --frac FRAC           the fraction of clients used for training: C
  --local_ep LOCAL_EP   the number of local epochs: E, default is 10
  --local_bs LOCAL_BS   local batch size: B, default=8, local gpu can only set 1
  --num_workers NUM_WORKERS
                        test colab gpu num_workers=1 is faster
  --lr LR               learning rate
  --momentum MOMENTUM   SGD momentum (default: 0.9)
  --model {fcn_mobilenetv2,deeplabv3_mobilenetv2,deeplabv3_mobilenetv3,lraspp_mobilenetv3,fcn_resnet50}
                        model name
  --num_classes NUM_CLASSES
                        number of classes max is 81, pretrained is 21
  --cpu_only            indicate to use cpu only
  --optimizer {sgd,adam}
                        type of optimizer
  -aux AUX_LR, --aux_lr AUX_LR
                        times of normal learning rate used for auxiliary classifier
  --lr_scheduler {lambda,step}
                        learning rate scheduler
  --checkpoint CHECKPOINT
                        full file name of the checkpoint
  --save_frequency SAVE_FREQUENCY
                        number of epochs to save checkpoint
  --test_frequency TEST_FREQUENCY
                        number of epochs to eval global test data
  --train_only
  --pretrained          only available for deeplab_mobilenetv3 and lraspp_mobilenetv3
  --activation {relu,tanh}
                        set activatition function in models as argument
  --dataset DATASET     name of dataset
  --iid IID             Default set to IID. Set to 0 for non-IID.
  --unequal UNEQUAL     whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)
  --verbose VERBOSE     verbose
  --seed SEED           random seed
  --root ROOT           home directory
  --data {val2017,train2017}
                        val has 5k images while train has 118k
  --sample_rate SAMPLE_RATE
                        fraction of large dataset to be used for training, can reduce training memory
  --local_test_frac LOCAL_TEST_FRAC
                        frac of num_users for local testing
  --freeze_backbone     choose to not train backbone
  --weight WEIGHT       the weight assigned to computing loss of background class
  --focus_class FOCUS_CLASS
                        the only class that affect loss function, other class weight set as 0 except background
  --dp                  must activate for training with differential privacy
  --virtual_bs VIRTUAL_BS
                        the bs for noise addition, to save memory
  --max_grad_norm MAX_GRAD_NORM
                        max per-sam norm to clip
  --noise_multiplier NOISE_MULTIPLIER
                        suggested range in 0.1~2
  --no_dropout          set dropout prob as 0
  --filename FILENAME   image filename for inference.
```


## Further Readings
### Papers:
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
* [Making the Shoe fit: Architectures, Initializations, and Tuning for Learning with Privacy](https://openreview.net/forum?id=rJg851rYwH)
### Blog Posts:
* [Federated learning comics](https://federated.withgoogle.com/)
* [Opacus (a pytorch DP library) FAQ](https://opacus.ai/docs/faq)

