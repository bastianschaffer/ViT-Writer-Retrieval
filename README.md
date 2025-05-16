# ViT-Writer-Retrieval (In-Progress)

## Prerequisites
* device with a GPU
* some python libraries

## Prepare Dataset

* source: https://zenodo.org/records/1324999
* trainset for vlad codebook: `icdar17-historicalwi-training-binarized.zip` (was also used for training the ViT)
* testset: `ScriptNet-HistoricalWI-2017-binarized.zip`
* unzip both sets into `your-unzip-directory`
* create or determine `your-dataset-directory` of your choice
### Seperate Train Images:
```sh
./dataset-class-seperator.sh <your-unzip-directory/icdar2017-training-binary> <your-dataset-directory/train>
```
### Seperate Test Images:
```sh
./dataset-class-seperator.sh <your-unzip-directory/ScriptNet-HistoricalWI-2017-binarized> <your-dataset-directory/val>
```
### Resulting File Structure:
```
# both empty since original images were removed by the script
your-unzip-directory/
├─ icdar2017-training-binary/
├─ ScriptNet-HistoricalWI-2017-binarized/

# files newly seperated into directories
your-dataset-directory/
├─ train/
│  ├─ 7/
│  │  ├─ some-image.png
│  │  ├─ some-image.png
│  │  ├─ ...
│  ├─ 11/
│  │  ├─ ...
│  ├─ ...
├─ val/
│  ├─ 1/
│  │  ├─ some-image.png
│  │  ├─ some-image.png
│  │  ├─...
│  ├─ 2/
│  │  ├─...
│  ├─ ...
```

## Download Checkpoint

* download `checkpoint-devout-grass-739.pth` from [my personal google drive](https://drive.google.com/drive/folders/1hC2-RbrwInoC-oqDwCTY6RBY8fnoXSR7?usp=sharing)
* see its corresponding learning curve in the google drive directory as well

## Run Vlad Training and Evaluation

### Setup Arguments
* for code simplicity, for now, arguments are not read from the command line but must be set inside the source of [main_vlad.py](main_vlad.py) manually
#### Arguments to be changed:
* `CHECKPOINT = "<your-checkpoint-path/checkpoint-devout-grass-739.pth>"`
* `TRAIN_PATH = "<your-dataset-directory/train>"`
* `VAL_PATH = "<your-dataset-directory/val>"`
* any other "constants" as you wish, like `K_TRAIN_EVAL`, ... 


### Run

* training and evaluation are both done in [main_vlad.py](main_vlad.py)
* in order to keep it running after exiting from ssh, use [run_vlad.sh](run_vlad.sh), which runs [main_vlad.py](main_vlad.py) in a detached tmux terminal
* console output will be redirected into `./vlad.log` to examine during and after the run

```sh
./run_vlad.sh
```