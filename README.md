# KKA: Improving Vision Anomaly Detection through Anomaly-related Knowledge from Large Language Models


This repository provides a PyTorch implementation of the Key Knowledge Augmentation (KKA) method. KKA aims to improve the performance of anomaly detection models 
by leveraging large language models to generate realistic anomaly samples, thus enabling better boundary learning between normal and anomalous data.

## Abstract

>> Vision anomaly detection, particularly in unsupervised settings, often struggles to distinguish between normal samples and anomalies due to the wide variability in anomalies. Recently, an increasing number of studies have focused on generating anomalies to help detectors learn more effective boundaries between normal samples and anomalies. However, as the generated anomalies are often derived from random factors, they frequently lack realism. Additionally, randomly generated anomalies typically offer limited support in constructing effective boundaries, as most differ substantially from normal samples and lie far from the boundary. To address these challenges, we propose Key Knowledge Augmentation (KKA), a method that extracts anomaly-related knowledge from large language models (LLMs). More speciﬁcally, KKA leverages the extensive prior knowledge of LLMs to generate meaningful anomalies based on normal samples. Then, KKA classiﬁes the generated anomalies as easy anomalies and hard anomalies according to their similarityto normal samples. Easy anomalies exhibit signiﬁcant differences from normal samples, whereas hard anomalies closely resemble normal samples. KKA iteratively updates the generated anomalies, and gradually increasing the proportion of hard anomalies to enable the detector to learn a more effective boundary. Experimental results show that the proposed method significantly improves the performance of various vision anomaly detectors while maintaining low generation costs.
>>

## The need for semi-supervised anomaly detection

![fig1](imgs/fig1.jpg?raw=true "fig1")

## Installation

This code is written in `Python 3.8` and requires the packages listed in `requirements.txt`.

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`

```
# pip install virtualenv
cd <path-to-KKA-Anomaly-Detection>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`

```
cd <path-to-KKA-Anomaly-Detection>
conda create --name myenv
source activate myenv
conda install --yes --file requirements.txt
```

## Running experiments

We have implemented the [`CIFAR-100`](https://www.cs.toronto.edu/%7Ekriz/cifar.html),
[`Oxford-102`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), and
[`UCM-Caption`](https://github.com/201528014227051/RSICD_optimal) datasets.

Have a look into `main.py` for all possible arguments and options.

### Create anomaly
Please use `use_glm4.py` and `create_image.py` in the `create` folder to create anomaly samples. After generating the anomaly samples, place them into a folder `0`.

### Oxford-102 example
```
cd <path-to-KAA-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/Oxford-102

# change to source directory
cd src

# run experiment
python main.py
```
In this dataset, `26` folder is considered to be the normal class. 
`0` folder is considered to be the anomaly class.
The autoencoder is provided through Deep SAD pre-training using `--pretrain True` with `main.py`. 
Autoencoder pretraining is used for parameter initialization.

### Iterate anomaly samples
Please modify the `user_dict` in the `datasets/flower.py` file to:

```python
user_dict = {0: [26, 0], 1: [26, 0]}
```

Then run `python main.py`, and in the output directory locate the `d_samples.json` and `e_samples.json` files. 
Use these two files with the `dpo_train.py` and `dpo_test.py` scripts located in the `dpo` folder. Finally, use the `create` folder scripts to generate new anomaly samples for iteration.

## License

MIT