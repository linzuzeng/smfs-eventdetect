Detect events for Constant Velocity SMFS experiments
=====

Paper under review

```
usage: train.py [-h] [--datafolder DATAFOLDER] [--modelfile MODELFILE]
                [--cuda 1 or 0] [--train TRAIN] [--predict_size PREDICT_SIZE]
                [--minibatches_per_step MINIBATCHES_PER_STEP]
                [--minibatch_size MINIBATCH_SIZE] [--epoch EPOCH]
                [--learning_rate LEARNING_RATE] [--data_split DATA_SPLIT]
                [--data_kept DATA_KEPT] [--source_scale SOURCE_SCALE]
                [--source_bias SOURCE_BIAS] [--downsampling DOWNSAMPLING]
                [--noiselevel NOISELEVEL] [--pdfheader PDFHEADER]

SMFS Event Detect

optional arguments:
  -h, --help            show this help message and exit
  --datafolder DATAFOLDER
                        folder for data (.np) files
  --modelfile MODELFILE
                        folder for model (.pt) files
  --cuda 1 or 0         use cuda
  --train TRAIN         set 1 to train the model, set 0 to test the trained
                        model
  --predict_size PREDICT_SIZE
                        predict_size for predicting
  --minibatches_per_step MINIBATCHES_PER_STEP
                        minibatches_per_step for training
  --minibatch_size MINIBATCH_SIZE
                        minibatch_size for training
  --epoch EPOCH         epochs for training
  --learning_rate LEARNING_RATE
                        learning_rate for training
  --data_split DATA_SPLIT
                        data_split for truncating dataset
  --data_kept DATA_KEPT
                        data_kept for truncating dataset
  --source_scale SOURCE_SCALE
                        source_scale in nm and pN for transforming input
                        signals in the dataset
  --source_bias SOURCE_BIAS
                        source_bias after applying source_scale for
                        transforming input signals in the dataset
  --downsampling DOWNSAMPLING
                        perform downsampling using averaging filter on input
                        data
  --noiselevel NOISELEVEL
                        add extra Gaussian noise (level in nm and pN) into
                        input dataset
  --report REPORT       folder for saving repots
  --report_note REPORT_NOTE
                        add prefix to each report file
```