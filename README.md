<p align="center"><img width="70%" src="static/img/guesswhich.png"/></p>

---------------------------------------------------------------------------

## Introduction

**[Evaluating Visual Conversational Agents via Cooperative Human-AI Games][]**  
Prithvijit Chattopadhyay*, Deshraj Yadav*, Viraj Prabhu, Arjun Chandrashekharan, Abhishek Das, Stefan Lee, Dhruv Batra, Devi Parikh
[arxiv.org/abs/1611.08669][1]  
[HCOMP 2017][4] (Oral)

## Abstract

As AI continues to advance, human-AI teams are inevitable. However, progress in AI is routinely measured in isolation, without a human in the loop. It is important to measure how progress in AI translates to humans being able to accomplish tasks better; i.e., the performance of human-AI teams. In this work, we design a cooperative game – GuessWhich to measure human-AI team performance in the specific context of the AI being a visual conversational agent. The AI, which we call ALICE, is provided an image which is unseen by the human. The human then asks ALICE questions aboutthis secret image to identify it from a fixed pool of images.

We measure performance of the human-ALICE team by the number of guesses it takes the human to correctly identify the secret image after a fixed number of dialog rounds with ALICE. We compare performance of the human-ALICE teams for two versions of ALICE. While AI literature shows that one version outperforms the other when paired with another AI, we find that this improvement in AI-AI performance does not translate to improved human-AI performance.


## Installation Instructions

### Installing the essential requirements

```shell
sudo apt-get install -y git python-pip python-dev
sudo apt-get install -y python-dev
sudo apt-get install -y autoconf automake libtool curl make g++ unzip
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
```

### Install Torch

```shell
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc
```

### Install PyTorch(Python Lua Wrapper)

```shell
git clone https://github.com/hughperkins/pytorch.git
cd pytorch
source ~/torch/install/bin/torch-activate
./build.sh
```

### Install RabbitMQ and Redis Server

```shell
sudo apt-get install -y redis-server rabbitmq-server
sudo rabbitmq-plugins enable rabbitmq_management
sudo service rabbitmq-server restart 
sudo service redis-server restart
```

### Lua dependencies

```shell
luarocks install loadcaffe
```

The below two dependencies are only required if you are going to use GPU

```shell
luarocks install cudnn
luarocks install cunn
```

### Cuda Installation

Note: CUDA and cuDNN is only required if you are going to use GPU

Download and install CUDA and cuDNN from [nvidia website](https://developer.nvidia.com/cuda-downloads) 

### Install dependencies

```shell
git clone https://github.com/Cloud-CV/GuessWhich.git
cd GuessWhich
sh download_models.sh
pip install -r requirements.txt
```

### Create the database

```shell
python manage.py makemigrations amt
python manage.py migrate
```

### Running the RabbitMQ workers and Development Server

Open 3 different terminal sessions and run the following commands:

```shell
cd chatbot && python sl_evalaute.py
cd chatbot && python rl_evalaute.py
python manage.py runserver
```

You are all set now. Visit http://127.0.0.1:8000 and you will have your demo running successfully.

## Contributors

* [Deshraj Yadav][2] (deshraj@gatech.edu)

## License

BSD

[1]: https://arxiv.org/abs/1611.08669
[2]: http://deshraj.github.io
[4]: http://www.humancomputation.com/2017/