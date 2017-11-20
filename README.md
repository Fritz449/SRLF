# SRLF
### Simple Reinforcement Learning Framework
-------------------------------------------------

SRLF helps to set up and train agent using one of these four algorithms:

1) Evolution strategies [(paper)](https://arxiv.org/abs/1703.03864)
2) Deep Deterministic Policy Gradient [(paper)](https://arxiv.org/abs/1509.02971)
3) TRPO [(paper)](https://arxiv.org/abs/1502.05477)
4) Rainbow [(paper)](https://arxiv.org/abs/1710.02298)

I didn't make very user-friendly interface, but if you know how the algorigthm you want to use works, you can do it more or less easily.

To use, follow these steps:
1) Choose algorithm you want to use
2) Be sure you know how it works
3) Study this implementation and how hyperparameters are used
4) Use **prepare_config.py** as an example to make your config. Also check helpers/utils.py to know how to set some of non-obvious hyperparameters (like **trainer**)
5) Launch ```python3 prepare_config.py``` to make config-file
6) Launch ```python3 run_experiment.py``` to run experiment (you should also check how **run_experiment.py** works)
7) Enjoy!

I also hope that framework can be helpful for people who want to study how to implement some of 4 algorithms that can be found here.


