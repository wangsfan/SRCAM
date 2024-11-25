# SRCAM
The Official PyTorch code for ["Superpixels Rectify CAMs: Contrastive Activation Maps with Superpixel Rectification for Weakly Supervised Semantic Segmentation"](), which is implemented based on the code of [L2G](https://github.com/PengtaoJiang/L2G). 
The segmentation framework is borrowed from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch).

## Installation
Use the following command to prepare your enviroment.
```
conda create --name <env> --file requirements.txt
```
Use the following command to activate your enviroment.
```
conda activate <env>
```

Download the PASCAL VOC dataset and MS COCO dataset, respectively. 
- [PASCAL VOC 2012 提取码cl1e](https://pan.baidu.com/s/1CCR840MJ3Rx7jQ-r1jLX9g)  
- [MS COCO 2014](https://cocodataset.org/#home)  
- [MS COCO 2014 Ground-Truth 提取码：0naq](https://pan.baidu.com/s/1O61VZG0CkyZ7fHoK0FfjAA)

SRCAM uses the off-the-shelf saliency maps generated from PoolNet. Download them and move to a folder named **Sal**.
- [Saliency maps for PASCAL VOC 2012](https://drive.google.com/file/d/1ZBLZ3YFw6yDIRWo0Apd4znOozg-Buj4A/view?usp=sharing)
- [Saliency maps for MS COCO 2014](https://drive.google.com/file/d/1IN6qQK0kL4_x8yzF7jvS6hNIFXsrR6XV/view?usp=sharing)  

The data folder structure should be like:
```
SRCAM
├── models
├── scripts
├── utils
├── data
│   ├── voc12
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── SegmentationClassAug
│   │   ├── Sal
│   ├── coco14
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── Sal

```
Download the [pretrained model 提取码：t4ce](https://pan.baidu.com/share/init?surl=vP8O0RPJKXM0HPhVCn7wXw) to initialize the classification network and put it to `./models/`.

## SRCAM
To train a SRCAM model on dataset VOC2012, you need to implement the following commands:
```
cd SRCAM/
./train_srcam_sal_voc.sh 
```
For COCO:
```
cd SRCAM/
./train_srcam_sal_coco.sh 
```
We provide the pretrained classification models on PASCAL VOC and MS COCO, respectively.
- [Pretrained models for VOC](https://drive.google.com/file/d/1Yc-LZ4bTM_1arpPBId6CMP9I2gOrDkdi/view?usp=sharing)
- [Pretrained models for COCO](https://drive.google.com/file/d/1i3b35g4GJO448kVdibBa5aL-yG6G2Huc/view?usp=sharing)  

After the training process, you will need the following command to generate pseudo labels 
and check their qualities.   
For VOC:
```
./test_srcam_voc.sh
```
For COCO:
```
./test_srcam_coco.sh
```
## Weakly Supervised Segmentation
To train a segmentation model, you need to generate pseudo segmentation labels first by 
```
./gen_gt_voc.sh
```
This code will generate pseudo segmentation labels in `./data/voc12/pseudo_seg_labels/`.  
For COCO
```
./gen_gt_coco.sh
```
This code will generate pseudo segmentation labels in `./data/coco14/pseudo_seg_labels/`.  


```
cd deeplab-pytorch
```
Download the [pretrained models](https://drive.google.com/file/d/1huoE5TcdUqLRFjVPaYSs2_sg2ehv9Z_s/view?usp=sharing) and put them into the `pretrained` folder.  In our paper, we also utilize the [caffe model](https://pan.baidu.com/share/init?surl=B77rPp_Kr1qEtdrTP6lVdw)（48kv）to initilize deeplab, which can usually obtain higher segmentation performance.

Train DeepLabv2-resnet101 model by
```
python main.py train \
      --config-path configs/voc12_resnet_dplv2.yaml
```
Test the segmentation model by 
```
python main.py test \
    --config-path configs/voc12_resnet_dplv2.yaml \
    --model-path data/models/voc12/voc12_resnet_v2/train_aug/checkpoint_final.pth
```
Apply the crf post-processing by 
```
python main.py crf \
    --config-path configs/voc12_resnet_dplv2.yaml
```

## Performance
Dataset | mIoU(val)   
--- |:---:|
PASCAL VOC  | 72.8 
MS COCO     | 44.8
