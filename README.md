# ML caching with guarantees

## Introduction
The cache replacement pipeline has been cloned and slightly differed from https://github.com/google-research/google-research/tree/master/cache_replacement.

## Getting the traces
To avoid problems with reproduction, we have saved all the traces used in our work to [Dropbox](https://www.dropbox.com/sh/h9lsrxgmofl2oso/AABgLHNNmla2X2ipqTGiL9Oua?dl=0). To move forward, please copy the traces you would like to work
 on to the `./cache_replacement/policy_learning/cache/traces/` folder.
 
## Setting up the environment
This project uses `python3`. Please install the required packages with
  
  `pip install -r requirements.txt`
   
Afterwards, please install OpenAI baselines with 
 
 ```pip install -e git+https://github.com/openai/baselines.git@ea25b9e8b234e6ee1bca43083f8f3cf974143998#egg=baselines```
 
## Training models with evaluation and parsing args
To run training, use this command:

```bash run.sh <DATASET> <DEVICE> <FRACTION> <DAGGER> <RESULT_DIR>```

where:
- DATASET - the name of used dataset (do not forget to download all three splits: train, valid, test)
- DEVICE - the GPU device to train on
- FRACTION - the fratcion of the train set to use (eg. `1`, `0.01`)
- DAGGER (`True` or `False`) - whether to use DAgger
- RESULT_DIR - data folder

For example:

```bash run.sh astar 0 0.33 True ./results```

This example script will do the following:
* Train a Parrot model with DAgger on 33% of the astar dataset, and several things save to the `./results/astar__dagger=true__fraction=0.33` folder:
    * `evictions` and `predictions` folders will contain easily readible evictions and predictions of the model during training
    * three `.json` files containing the configs used in the training
    * `tensorboard` folder with visualization data
    * `checkpoints` folder with saved models
    * `logs.txt` with the cache hit rates per full validation (this will be used in evaluation to get the best checkpoint)
* Evaluate the trained model on the test set, saving the results to `./results/astar__dagger=true__fraction=0.33/test` folder.
* Create a `./results/parsed/astar__dagger=true__fraction=0.33` folder that will contain the crucial files (parsed outs + logs with scores)
* Parse the evictions and predictions from the evaluation, to a more leightweight format.