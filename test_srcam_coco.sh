#!/bin/sh
EXP=exp_coco
TYPE=ms
THR=0.25
NUM_CLASSES=80
DATASET=mscoco
GPU_ID=0

#CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./scripts/test_srcam_coco.py \
#    --img_dir=./data/coco14/JPEGImages/ \
#    --test_list=./data/coco14/val_cls2.txt \
#    --arch=vgg \
#    --batch_size=1 \
#    --dataset=${DATASET} \
#    --input_size=224 \
#	--num_classes=${NUM_CLASSES} \
#    --thr=${THR} \
#    --restore_from=./runs/${EXP}/model/${DATASET}_epoch_14.pth \
#    --save_dir=./runs/${EXP}/${TYPE}/attention/ \
#    --multi_scale \
#    --cam_png=./runs/${EXP}/cam_png/




CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./scripts/test_srcam_coco.py \
    --img_dir=./data/coco14/JPEGImages/ \
    --test_list=./data/coco14/val_cls2.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=${DATASET} \
    --input_size=224 \
	--num_classes=${NUM_CLASSES} \
    --thr=0.16 \
    --restore_from=./runs/${EXP}/model/${DATASET}_epoch_14.pth \
    --save_dir=./runs/${EXP}/${TYPE}/attention16/ \
    --multi_scale \
    --cam_png=./runs/${EXP}/cam_png16/





CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./scripts/test_srcam_coco.py \
    --img_dir=./data/coco14/JPEGImages/ \
    --test_list=./data/coco14/val_cls2.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=${DATASET} \
    --input_size=224 \
	--num_classes=${NUM_CLASSES} \
    --thr=0.18 \
    --restore_from=./runs/${EXP}/model/${DATASET}_epoch_14.pth \
    --save_dir=./runs/${EXP}/${TYPE}/attention18/ \
    --multi_scale \
    --cam_png=./runs/${EXP}/cam_png18/







CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./scripts/test_srcam_coco.py \
    --img_dir=./data/coco14/JPEGImages/ \
    --test_list=./data/coco14/val_cls2.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=${DATASET} \
    --input_size=224 \
	--num_classes=${NUM_CLASSES} \
    --thr=${THR} \
    --restore_from=./runs/${EXP}/model/${DATASET}_epoch_11_temp.pth \
    --save_dir=./runs/${EXP}/${TYPE}/attention11/ \
    --multi_scale \
    --cam_png=./runs/${EXP}/cam_png11/




CUDA_VISIBLE_DEVICES=${GPU_ID} python3 scripts/evaluate_mthr_coco.py \
   --dataset=mscoco \
   --datalist=./data/coco14/val_cls2.txt \
   --gt_dir=./data/coco14/SegmentationClass/ \
   --save_path=./runs/${EXP}/${TYPE}/result.txt \
   --save_label_path=./runs/${EXP}/${TYPE}/cam_png \
   --pred_dir=./runs/${EXP}/${TYPE}/attention/
