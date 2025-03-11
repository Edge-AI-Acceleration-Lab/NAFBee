<a href="https://istd.sutd.edu.sg/people/phd-students/tomomasa-yamasaki">
    <img src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/logo.png" alt="Tomo logo" title="Tomo" align="right" height="110" />
</a>

# NAFBee: Neural Network Activation Function Benchmark
This is a benchmark for networks with a variety of activation functions. NAFBee provides network information and accuracy. User can obtain the accuracy without training. NAFBee is used for RBFleX-NAS.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![](https://img.shields.io/github/repo-size/tomomasayamasaki/NAFBee)
![](https://img.shields.io/github/commit-activity/y/tomomasayamasaki/NAFBee)
![](https://img.shields.io/github/last-commit/tomomasayamasaki/NAFBee)
![](https://img.shields.io/github/languages/count/tomomasayamasaki/NAFBee)

## 🟨 Creator Info
### Tomomasa Yamasaki  
Singapore University of Technology and Design (Sep 2021 intake)  
Linkedin: https://www.linkedin.com/in/tomomasa/  
Github: https://github.com/tomomasayamasaki


## 🟨 Requirement
- python 3.x
- PyTorch
```
conda create -n myenv python=3.8
conda activate myenv
conda install -c conda-forge transformers=4.5.0 tokenizers=0.10.3
conda install -c conda-forge huggingface_hub
conda install pandas
conda install scikit-learn
```

## 🟨 Datasets for this neural network benchmark
- CIFAR-10 for VGG19
- SST-2 for BERT

## 🟨 How to use
### 0. Import packages
```python
# VGG19
import json
from models import *
```
```python
# BERT
import json
from BERT_model import BertModel
```

### 1. Load NAFBee.json
```python
file_path = "NAFBee_VGG19.json" #or "NAFBee_BERT.json"
with open(file_path, "r") as json_file:
    nafbee = json.load(json_file)
```

### 2. Get the network information
```python
info = nafbee["1"] #you can input numbers from 1 to 11.
print(info)

#{'network': 'VGG19', 'activation': 'ReLU', 'accuracy': 91.06}
```
### 3. All information of the No.1 network
```python
info_network = info["network"]
info_activation = info["activation"]
info_accuracy = info["accuracy"]
```

### 4. Define the model on PyTorch
```python
if "VGG" in info_network:
    model = VGG(info_network, info_activation)
```

## 🟨 Demo
You can see a program to get the model. You can add any program using the model on Pytorch such as training or scoring.
```python
python NAFBee_VGG19.py #VGG19
```
```python
python NAFBee_BERT.py #BERT
```

## 🟨 Citing RBFleX-NAS
If you use NAFBee, please cite the following paper:
```
XXXX
```

## 🟨 Licence
[MIT Licence](https://en.wikipedia.org/wiki/MIT_License)
