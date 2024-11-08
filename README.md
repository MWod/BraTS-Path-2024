# BraTS-Path-2024
Contribution to the BraTS-Path 2024 Challenge organized during MICCAI 2024 conference.

The repository contains the source code used to prepare the submission to the 1st edition of the [BraTS-Path Challenge](https://www.synapse.org/Synapse:syn53708249/wiki/).

The submission scored the 1st place in the competition - outperforming the fine-tuned general-purpose computer vision models.

# How to reproduce the training?

* Download the ProvGigaPath pretrained model [ProvGigaPath](https://huggingface.co/prov-gigapath/prov-gigapath).
You need to download only the patch-level encoder. The slide-level encoder is not used. Remember about giving credit to the ProvGigaPath authors! [Click](https://www.nature.com/articles/s41586-024-07441-w).
* Navigate to the [Paths](./src/paths/pc_paths.py) and set the path to the BraTS-Path dataset and the ProvGigaPath model.
* Parse the BraTS-Path dataset (available upon request from the challenge organizers): [Parse](./src/parsers/parse_dataset.py).
* Modify the [training parameters](./src/experiments/pc_experiments/experiments.py) to start the training without any previous checkpoints.
* Wait until the training finishes. Then you can use the [evaluation scripts](./src/evaluation/) to test the performance.

# How to reproduce the evaluation?
* Register to the [BraTS-Path Challenge](https://www.synapse.org/Synapse:syn53708249/wiki/) and follow the guidelines there related to the MLCube preparation.


# License
The source code is released under Apache-2.0 license. However, note that the source code for ProvGigaPath and the associated pretrained model follow a diferent license.
