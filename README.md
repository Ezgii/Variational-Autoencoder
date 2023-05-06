
### Project description
In this project, an Auto Encoder (AE) and a Variational Auto Encoder (VAE) are designed in Python.

### Goals of the project
* Explore the gym framework for training RL agents.
* Apply my knowledge on VAE to learn image generation.
* Train generative models to produce sample pixel observation images from gym environments.

### Environment
[OpenAI's Gym](https://gym.openai.com/) is a framework for training reinforcement 
learning agents. It provides a set of environments and a
standardized interface for interacting with those.   
In this project, I used the [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment from gym.

### Installation

#### Using conda (recommended)    
1. [Install Anaconda](https://www.anaconda.com/products/individual)

2. Create the env    
`conda create a1 python=3.8` 

3. Activate the env     
`conda activate a1`    

4. install torch ([steps from pytorch installation guide](https://pytorch.org/)):    
- if you don't have an nvidia gpu or don't want to bother with cuda installation:    
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`    
  
- if you have an nvidia gpu and want to use it:    
[install cuda](https://docs.nvidia.com/cuda/index.html)   
install torch with cuda:   
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. other dependencies   
`conda install -c conda-forge matplotlib gym opencv pyglet`

#### Using pip
`python3 -m pip install -r requirements.txt`

### Code
[MyVAE.py](https://github.com/Ezgii/Variational-Autoencoder/blob/master/MyVAE.py) - my VAE model   
[train_vae.py](https://github.com/Ezgii/Variational-Autoencoder/blob/master/train_vae.py) - script to collect pixel observations from gym environments using a random policy, and train the VAE model     
[sample_vae.py](https://github.com/Ezgii/Variational-Autoencoder/blob/master/sample_vae.py) - samples from the VAE trained by train_vae.py    

### How to run the code
On terminal, write:

`python3 MyVae.py`

`python3 train_vae.py`

`python3 sample_vae.py`

### Results
