#!/bin/bash


rm -rf savedmodeldir
python3 model_inspect.py \
  --runmode=saved_model \
  --model_name=efficientdet-d0 \
  --ckpt_path=/workspace/data/efficientdet-fashion3k \
  --saved_model_dir=savedmodeldir \
  --tflite_path=efficientdet-d0.tflite \
  --hparams=config/fashion3k-config.yaml
