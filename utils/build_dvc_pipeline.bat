if [%1]==[] goto usage

set EXP_DIR=%1

dvc run -d prepare_dataset.py ^
  -o data/indices/train_seg.npy -o data/indices/train_class.npy ^
  -o data/indices/val_seg.npy -o data/indices/val_class.npy ^
  -o data/indices/test_seg.npy -o data/indices/test_class.npy ^
  --no-exec python prepare_dataset.py

dvc run -d train_classifier.py ^
  -o experiments/%EXP_DIR%/class/resnet18 -f train_resnet18class.dvc ^
  -d data/indices/train_class.npy ^
  -d data/indices/val_class.npy ^
  --no-exec python train_classifier.py -m resnet18

dvc run -d train_classifier.py ^
  -o experiments/%EXP_DIR%/class/resnet34 -f train_resnet34class.dvc ^
  -d data/indices/train_class.npy ^
  -d data/indices/val_class.npy ^
  --no-exec python train_classifier.py -m resnet34

dvc run -d predict_model_class.py ^
  -f predict_resnet18class.dvc ^
  -d experiments/%EXP_DIR%/class/resnet18 ^
  -d data/indices/test_class.npy ^
  -o out/class/resnet18_out.csv ^
  --no-exec python predict_model_class.py -m resnet18 -o out/class/resnet18_class_out.csv

dvc run -d predict_model_class.py ^
  -f predict_resnet34class.dvc ^
  -d experiments/%EXP_DIR%/class/resnet34 ^
  -d data/indices/test_class.npy ^
  -o out/class/resnet34_out.csv ^
  --no-exec python predict_model_class.py -m resnet34 -o out/class/resnet34_class_out.csv

dvc run -d detect_class_best_predict_config.py ^
  -d out/class/resnet18_out.csv ^
  -d out/class/resnet34_out.csv ^
  -o out/class/class_best_predict.json ^
  --no-exec python detect_class_best_predict_config.py

dvc run -d class_eval.py ^
  -d out/class/class_best_predict.json ^
  -o out/class/class_predict.csv ^
  --no-exec python class_eval.py

dvc run -d train_segmentation.py ^
  -o experiments/%EXP_DIR%/seg/resnet18 -f train_resnet18seg.dvc ^
  -d data/indices/train_seg.npy ^
  -d data/indices/val_seg.npy ^
  --no-exec python train_segmentation.py -m resnet18

dvc run -d train_segmentation.py ^
  -o experiments/%EXP_DIR%/seg/resnet34 -f train_resnet34seg.dvc ^
  -d data/indices/train_seg.npy ^
  -d data/indices/val_seg.npy ^
  --no-exec python train_segmentation.py -m resnet34

dvc run -d predict_model_seg.py ^
  -f predict_resnet18seg.dvc ^
  -d experiments/%EXP_DIR%/seg/resnet18 ^
  -d data/indices/test_seg.npy ^
  -o out/seg/resnet18_out.csv ^
  --no-exec python predict_model_seg.py -m resnet18 -o out/resnet18_out.csv

dvc run -d predict_model_seg.py ^
  -f predict_resnet34seg.dvc ^
  -d experiments/%EXP_DIR%/seg/resnet34 ^
  -d data/indices/test_seg.npy ^
  -o out/seg/resnet34_out.csv ^
  --no-exec python predict_model_seg.py -m resnet34 -o out/resnet34_out.csv

dvc run -d detect_seg_best_predict_config.py ^
  -d out/seg/resnet18_out.csv ^
  -d out/seg/resnet34_out.csv ^
  -o out/seg/seg_best_predict.json ^
  --no-exec python detect_seg_best_predict_config.py

dvc run -d final_predict.py ^
  -f Dvcfile ^
  -d out/seg/seg_best_predict.json ^
  -d out/class/class_predict.csv ^
  -o out/seg/final_predict.csv ^
  --no-exec python final_predict.py

git add *.dvc
git add Dvcfile

:usage
@echo "Usage: ./utils/build_dvc_pipeline.sh <experiment dir name>"
@echo "   For example: ./utils/build_dvc_pipeline.sh exp1"
exit /B 1