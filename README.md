# ViT-Writer-Retrieval

This repository reproduces `Self-Supervised Vision Transformers for Writer Retrieval` by Tim Raven et al.

## Dataset

Source: https://zenodo.org/records/1324999

Both train and test set must be unzipped to be ready for use.
To decrease runtime, optionally convert images of `ScriptNet-HistoricalWI-2017-binarized` to PNG:
```
cd <your-scripnet-directory>
mogrify -format png *.jpg
rm ./*.jpg
```

## Download Checkpoint

If you want to test the pipeline without manually training the ViT, download a checkpoint from [my personal google drive](https://drive.google.com/drive/folders/1hC2-RbrwInoC-oqDwCTY6RBY8fnoXSR7?usp=sharing).

The vlad code works with both 1-channel and 3-channel weights without modifications necessary. The difference is that internally the single binary image is duplicated 3 times when the 3-channel weights are used.
We found no reason to use the 3-channel weights since both yield the same accuracy while the pipeline runs slighly faster with the single-channel weights.

## Train the ViT

The ViT can be trained by running [main_attmask.py](attmask/main_attmask.py).
See [job-script.sbatch](attmask/job-script.sbatch) to run on a SLURM cluster.
See [run_without_cluster.sh](attmask/run_without_cluster.sh) to run AttMask with a single GPU on a normal system without a SLURM cluster.

There is a [sweep config](attmask/sweep-config.yml) for parameter sweeping with 'Weights and Biasis', but to use it, the code of [main_attmask.py](attmask/main_attmask.py) must be slighly adjusted at the bottom.


## Run Vlad Training and Evaluation

Feature extraction, aggregation and evaluation are all done in [main_vlad.py](vlad/main_vlad.py).
In order to keep it running after exiting from ssh, use [run_vlad_wth_tmux.sh](vlad/run_vlad_with_tmux.sh), which runs [main_vlad.py](main_vlad.py) in a detached tmux terminal.
The console output will be redirected into a `vlad.log` file to examine during and after the run.

Arguments for the vlad pipeline must be set inside the source of [main_vlad.py](vlad/main_vlad.py):

    * CHECKPOINT = "<your-checkpoint-path/your-checkpoint.pth>"
    * TRAIN_PATH = "<your-trainset-directory>"
    * VAL_PATH = "<your-testset-directory>"
    * any other "constants" as you wish, like `S_EVAL`, ... 


```sh
sh run_vlad_with_tmux.sh
```