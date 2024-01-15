## Sample-rate Independent RNNs

### Getting started
Create conda environment:

`conda env create -f environment.yml`

### Training networks
Training the networks is controlled via the config files in `/configs`. Each file contains config parameters for
three key elements: trainer, model and data. To train a network from one of the config files:

`python3 main.py fit --config configs/<config_file>`


