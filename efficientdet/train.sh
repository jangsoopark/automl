#!/bin/bash


python3 main.py\
  --mode=train_and_eval\
  --training_file_pattern=/workspace/data/style3k/style3k-train*.tfrecord\
	--validation_file_pattern=/workspace/data/style3k/style3k-test*.tfrecord\
	--model_name=efficientdet-d0\
	--model_dir=/workspace/data/efficientdet-fashion3k\
	--ckpt=/workspace/data/efficientdet/backbone/efficientdet-d0\
	--train_batch_size=8\
	--eval_batch_size=8\
	--eval_samples=1024\
	--num_examples_per_epoch=5717\
	--hparams=config/fashion3k-config.yaml\
	--strategy=gpus


#python main.py --mode=train_and_eval --training_file_pattern=D:\ivs\dataset\tutorial\tfrecord\pascal*.tfrecord --validation_file_pattern=D:\ivs\dataset\tutorial\tfrecord\pascal*.tfrecord --model_name=efficientdet-d0 --model_dir=D:\ivs\models\automl\efficientdet-d0 --ckpt=D:\ivs\models\backbone\coco\efficientdet-d0 --train_batch_size=8 --eval_batch_size=8 --eval_samples=1024 --num_examples_per_epoch=5717 --hparams=voc_config.yaml --use_tpu=False
