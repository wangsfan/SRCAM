#!/bin/sh
EXP=exp_voc
TYPE=ms_spx
THR=0.25

CUDA_VISIBLE_DEVICES=1 python3 ./scripts/test_srcam_voc.py \
    --img_dir=./data/voc12/JPEGImages/ \
    --test_list=./data/voc12/val_cls.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=224 \
	  --num_classes=20 \
    --thr=${THR} \
    --restore_from=./runs/${EXP}/model/256spx-pascal_voc_epoch_9.pth \
    --save_dir=./runs/${EXP}/${TYPE}/attention_256spx_only_sal/ \
    --multi_scale \
    --cam_png=./runs/${EXP}/cam_png_256spx_only_sal/


CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_mthr_voc.py \
    --datalist ./data/voc12/val_aug.txt \
    --gt_dir ./data/voc12/SegmentationClassAug/ \
    --save_path ./runs/${EXP}/${TYPE}/result_256spx_only_sal.txt \
    --pred_dir ./runs//${EXP}/${TYPE}/attention_256spx_only_sal/
