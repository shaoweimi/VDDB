# Dependencies
This code is developed with Python3, and we recommend PyTorch >=1.11. Install the dependencies with [Anaconda](https://www.anaconda.com/products/individual) and activate the environment `vddb` with
```bash
conda env create --file requirements.yaml python=3
conda activate vddb
```
# Pre-trained models

We provide pretrained checkpoints via Google cloud [here](https://drive.google.com/drive/folders/1B98Qe8_nb2IefDkJDYtBGolnH0hjQ1r2?usp=sharing).

# Datasets
For Edges2Handbags, please follow instructions from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md).
For DIODE, please download appropriate datasets from [here](https://diode-dataset.org/).

# Model training and sampling
We provide bash files [train.sh](VDDB/scripts/train.sh) and [sample.sh](VDDB/scripts/sample.sh) for model training and sampling. 
## Acknowledgement
Our code is implemented based on I2SB and DDBM.

[I2SB](https://github.com/NVlabs/I2SB)

[DDBM](https://github.com/alexzhou907/DDBM)
