# 2048-PyTorch-DQN

Implementing [Deep Q-Network](https://deepmind.com/research/dqn/) with `PyTorch` to solve 2048. 

#### Requirements:
`NVIDIA GPU` and `CUDA`, of course...  and `PyTorch`, along with some others in `requirements.txt`


#### 2048

[2048](https://github.com/gabrielecirulli/2048) is a popular game by Gabriele Cirulli, 
and I am using an [implemented version of Python](https://github.com/yangshun/2048-python) by [Tay Yang Shun](http://github.com/yangshun)
and [Emmanuel Goh](http://github.com/emman27).

To start the training, which takes horribly long to get to an expert level, run the following:
    
    $ python3 train_dqn.py

![screenshot](img/2048.gif)


