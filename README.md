# Neural Pipeline working directory template

# Directory structure
```
-| worklog <- directory with all experiments description
-| experiments <- checkpoints, logs and other training artefacts
-| inference <- inferences
-| train_config <- all training hyperparams
---| train_config.py <- file with TrainConfig class
---| dataset.py <- dataset(s) implementation
---| metrics.py <- metrics implementation
---| loss.py <- loss implementation
-| utils <- directory with utils
---| build_dvc_pipeline.sh <- script for build DVC pipeline
---| export_model.py <- script for export model to different formats
---| visualize_dataset.py <- script for dataset visualisation
-| prepare_dataset.py <- dataset preporation pipeline step
-| train.py <- training pipeline step
-| predict.py <- predict pipeline step
```
