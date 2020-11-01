## Openai Gym Wrapper for Shallow Reinforcment Leanring on Atari Features:

This codebase contains an Openai gym wrapper for the handcrafted features used to train a linear model to play atari introduced in 
[State of the Art Control of Atari Games Using Shallow Reinforcement Learning](https://arxiv.org/pdf/1512.01563.pdf) paper.
The original codebase for the paper can be found using this [[Link](https://github.com/mcmachado/b-pro)].


**To use the wrapper first compile either `BPROSLibrary.cpp` or `BPROSTLibrary.cpp` c++ binary using the command below:**

```
g++ -std=c++11 -shared -fPIC BPROSLibrary.cpp -o BPROSLibrary.so -lboost_python -lpython2.7 -I/usr/local/include/python2.7
```

**After compiling the proper c++ binary install this package:**

```
pip install -e .
```

**Below is example code:**
```
from gym_shallow_atari import WrapperBPROST

env = WrapperBPROST(gym.make('AirRaid-v0'))
env.reset()
while True:
    a = env.action_space.sample()
    obs, reward, done, _ = env.step(a)
    if done:
        break
```
