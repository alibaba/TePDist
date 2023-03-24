# TePDist Benchmark for Wide-ResNet
This implementation is only for benchmark for training, which  takes fake data as input.

## Configuration and Parameters

* The entry script is `run.sh`. Users can modifies `train_batch_size` and `model_type` to
  benchmark for different scenarios.

* We define 6 versions of Wide-ResNet with different model size and layers. The parameter
  `model_type` indicates the different configurations of Wide-ResNet. It accepts integer
   value from 0 to 6. The corresponding configuration is defined at the beginning in
  `train_imagenet.py`.

## Experiment nodes

As default, TePDist explores SPMD and Pipeline Strategy automatically and select the best
according to the cost. For this implementation, TePDist will give SPMD Strategy without
pipeline as the best strategy for the clusters with only two GPU devices.

## Wide-ResNet configurations

```
 model_type  model size
    0        250M (50 layers)
    1        500M (50 layers)
    2        1B   (50 layers)
    3        2B   (50 layers)
    4        4B   (50 layers)
    5        7B   (50 layers)
    6        13B  (101 layers)
```
