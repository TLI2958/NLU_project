## Cross-Lingual Data Augmentation: Enhancing Slot Labeling with XLM-R

### DS-GA 1012 Project

- This repository contains a PyTorch implementation of a multi-task learning model for intent classification and slot filling, utilizing the [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base) model.
- The implementation is inspired by [Xu et. al., (2020)](https://aclanthology.org/2020.emnlp-main.410/).

### Table of Contents
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Debugging](#debugging)
- [Acknowledgments](#acknowledgments)

## Usage

### Training
Here is a conda environment configuration file with the necessary dependencies that you may choose to use.
```
conda env create -f environment.yml
conda activate NLU_project
```
Be sure to modify the `prefix` as needed.

To train the model on the training data, run:

```bash
python main.py --train --save_dir /path/to/your/directory/
```

### Evaluation

To evaluate the model on the test set, run:

```bash
python main.py --eval --save_dir /path/to/your/directory/
```

### Debugging

For debugging purposes, you can train and evaluate on smaller subsets of the data:

```bash
python main.py --debug --save_dir /path/to/your/directory/
python main.py --debug_eval --save_dir /path/to/your/directory/
```


## Acknowledgments
- [MASSIVE](https://github.com/alexa/massive)
- [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [Multiatis](https://github.com/amazon-science/multiatis)
