# Automatic Generative Domain Adaptation (Auto-GDA)

This repository contains code for the Automatic Generative Domain Adaption (Auto-GDA) framework.
It can be used to generate high-quality synthetic data, which is useful for fine-tuning efficient NLI models.
The codebase should accompany our ICLR submission.

## Install (Tested on new Ubuntu AWS instance)

To install this package first make sure you run `mwinit -o`. Then run the following commands:
```
git clone ssh://git.amazon.com/pkg/DomainAdaptationForFactualConsistencyDetection
cd DomainAdaptationForFactualConsistencyDetection
```
(or sync to remote machine via PyCharm Deployment)

Install mysql server which is used to manage hyperparameter configuration and venv (if not already installed)
```
sudo apt-get install mysql-server pkg-config default-libmysqlclient-dev build-essential python3.10-venv
```

# Create virtual environment
In the project folder (DomainAdaptationForFactualConsistencyDetection), run
```
python3 -m venv autogda
source autogda/bin/activate
```

# Install via pip
Verify that the nvidia-gpus are recognized by running
``nvidia-smi`` before installation of requirements.
if not you can try
```
sudo apt install nvidia-driver-535
```
Then do
```
pip install -r requirements.txt
```

## Setting up th MySQL-DB for hyperparameter search
Install mysql server via apt get as described above. Then log into the mysql console:
```sudo mysql -u root```
and run the comments
```
CREATE USER optuna@"%";
CREATE DATABASE optuna;
GRANT ALL ON optuna.* TO optuna@"%";
```

## Details on individual files in the repository and their usage:
Please confer ```experiment_reproduction.md```.


## License

This project is licensed under the Apache-2.0 License.
