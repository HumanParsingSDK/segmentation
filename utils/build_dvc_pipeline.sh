#!/usr/bin/env bash

if [[ "$#" -ne 1 ]]; then
    echo "Usage: ./utils/build_dvc_pipeline.sh <experiment dir name>"
    echo "   For example: ./utils/build_dvc_pipeline.sh exp1"
    exit 0
fi

EXP_DIR="$1"

dvc run -d prepare_dataset.py \
  -f dvc_stages/prepare_dataset.dvc \
  -o data/indices/train_seg.npy -o data/indices/train_class.npy \
  -o data/indices/val_seg.npy -o data/indices/val_class.npy \
  -o data/indices/test_seg.npy -o data/indices/test_class.npy \
  --no-exec python prepare_dataset.py

######### TRAIN CLASS #########

dvc run -d train_classifier.py \
  -f dvc_stages/train_class_resnet18.dvc \
  -o experiments/$EXP_DIR/class/resnet18 \
  -d data/indices/train_class.npy \
  -d data/indices/val_class.npy \
  --no-exec python train_classifier.py -m resnet18

dvc run -d train_classifier.py \
  -f dvc_stages/train_class_resnet34.dvc \
  -o experiments/$EXP_DIR/class/resnet34 \
  -d data/indices/train_class.npy \
  -d data/indices/val_class.npy \
  --no-exec python train_classifier.py -m resnet34

######### PREDICT CLASS #########

dvc run -d class_eval.py \
  -f dvc_stages/predict_class.dvc \
  -d experiments/$EXP_DIR/class/resnet18 \
  -d experiments/$EXP_DIR/class/resnet34 \
  -d data/indices/test_class.npy \
  -o out/class/class_predict.csv \
  --no-exec python class_eval.py -m resnet18 resnet34 -o out/class/class_predict.csv

######### TRAIN SEG #########

dvc run -d train_segmentation.py \
  -f dvc_stages/train_resnet18seg.dvc \
  -o experiments/$EXP_DIR/seg/resnet18 \
  -d data/indices/train_seg.npy \
  -d data/indices/val_seg.npy \
  --no-exec python train_segmentation.py -m resnet18

dvc run -d train_segmentation.py \
  -f dvc_stages/train_resnet34seg.dvc \
  -o experiments/$EXP_DIR/seg/resnet34 \
  -d data/indices/train_seg.npy \
  -d data/indices/val_seg.npy \
  --no-exec python train_segmentation.py -m resnet34

######### PREDICT SEG #########

#dvc run -d predict_model_seg.py \
#  -f dvc_stages/predict_resnet18seg.dvc \
#  -d experiments/$EXP_DIR/seg/resnet18 \
#  -d data/indices/test_seg.npy \
#  -o out/seg/resnet18_out.csv \
#  --no-exec python predict_model_seg.py -m resnet18 -o out/resnet18_out.csv
#
#dvc run -d predict_model_seg.py \
#  -f dvc_stages/predict_resnet34seg.dvc \
#  -d experiments/$EXP_DIR/seg/resnet34 \
#  -d data/indices/test_seg.npy \
#  -o out/seg/resnet34_out.csv \
#  --no-exec python predict_model_seg.py -m resnet34 -o out/resnet34_out.csv
#
#dvc run -d detect_seg_best_predict_config.py \
#  -f dvc_stages/seg_best_config.dvc \
#  -d out/seg/resnet18_out.csv \
#  -d out/seg/resnet34_out.csv \
#  -o out/seg/seg_best_predict.json \
#  --no-exec python detect_seg_best_predict_config.py

dvc run -d final_predict.py \
  -f Dvcfile \
  -d experiments/$EXP_DIR/seg/resnet18 \
  -d experiments/$EXP_DIR/seg/resnet34 \
  -d out/class/class_predict.csv \
  -d final_predict.py \
  -o out/seg/final_predict.csv \
  --no-exec python final_predict.py -m resnet18 resnet34 -o out/seg/final_predict.csv -c out/class/class_predict.csv

git add dvc_stages/*.dvc
git add Dvcfile
