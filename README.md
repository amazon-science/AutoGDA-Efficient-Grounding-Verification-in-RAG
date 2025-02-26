# Automatic Generative Domain Adaptation (Auto-GDA)

This repository contains code for the Automatic Generative Domain Adaption (Auto-GDA) framework published at ICLR 2025.
It can be used to generate high-quality synthetic NLI data, which is useful for fine-tuning efficient NLI models to specific domains.

Please refer to our paper

[Auto-GDA: Automatic Domain Adaptation for Efficient Grounding Verification in Retrieval Augmented Generation](https://openreview.net/forum?id=w5ZtXOzMeJ) 
by Tobias Leemann, Periklis Petridis, Giuseppe Vietri, Dionysis Manousakas, Aaron Roth, and Sergul Aydore

for more details about the method.


# Installation


```
git clone git@github.com:amazon-science/AutoGDA-Efficient-Grounding-Verification-in-RAG.git
cd AutoGDA-Efficient-Grounding-Verification-in-RAG
```

Install mysql server which is used to manage hyperparameter configuration and venv or conda (if not already installed)

```
sudo apt-get install mysql-server pkg-config default-libmysqlclient-dev build-essential python3.10-venv
```

## Create virtual environment
In the project folder (DomainAdaptationForFactualConsistencyDetection), run

```
python3 -m venv autogda
source autogda/bin/activate
```

or using conda run

```
conda create -n autogda python=3.10
conda activate autogda
```


## Install dependencies via pip
Verify that the nvidia-gpus are recognized by running
``nvidia-smi`` before installation of requirements
if not you can try installing the proper Nvidia Driver like
```
sudo apt install nvidia-driver-535
```

Then install die remaining requirements using pip (preferred) or poetry
```
pip install -r requirements.txt
```

Optionally, activate the enviorment for usage with Jupyter by running
```
python -m ipykernel install --user --name autogda
```
## Setting up the MySQL-DB for hyperparameter search
Install mysql server via apt get as described above. Then log into the mysql console:
```sudo mysql -u root```
and run the comments
```
CREATE USER optuna@"%";
CREATE DATABASE optuna;
GRANT ALL ON optuna.* TO optuna@"%";
```

# Usage
**Datasets.** First, download the datasets by exectuing the script ```./src/scripts/download_datasets.sh``` from the main directory as working directory.

**LLM APIs.** For using the LLM APIs either make sure you have valid credentials for Amazon Bedrock in your ```~/.aws``` folder or set the OpenAI key in ```data/openai.json```.

## Quickstart

## Details on experiments and files in the repository:
Please confer ```experiment_reproduction.md``` for details on how to use the method and to run experiments.

# Reference
Please cite our work if you use this codebase, for instance using the following BibTeX-Entry:

```
@inproceedings{
leemann2025autogda,
title={Auto-{GDA}: Automatic Domain Adaptation for Efficient Grounding Verification in Retrieval Augmented Generation},
author={Tobias Leemann and Periklis Petridis and Giuseppe Vietri and Dionysis Manousakas and Aaron Roth and Sergul Aydore},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=w5ZtXOzMeJ}
}
```

# License

This project is licensed under the Apache-2.0 License.
