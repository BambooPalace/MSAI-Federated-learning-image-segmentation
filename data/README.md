## Datasets

### Classification
MNIST and CIFAR will be automatically downloaded into their folders, no need to manual download.

### Segmentaion
To run segmentation experiments, download COCO dataset from [here](https://cocodataset.org/#download) into `data/coco` folder.

For val2017 dataset, download the images (5K) from [here](http://images.cocodataset.org/zips/val2017.zip) and unzip all images into `coco/val2017`;
download the segmentation annotations(including train/val/test) from [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and save the jason file into `coco/annotations`. The annotation file name for val2017 is `instances_val2017.json` and the naming convention follows.

For train2017 dataset, download the images (118K) from [here](http://images.cocodataset.org/zips/train2017.zip) and save images into `coco\train2017`; the train2017 annotations can be downloaded together with other annotations as mentioned above.

For speed of training, it is recommend to train segmentation model with val2017 first.

