Pseudo Random Number Generation through Reinforcement Learning and Recurrent Neural Networks
********************************************************************************************

Luca Pasqualini, Maurizio Parton
################################################################

GitHub for a reinforcement learning research project consisting in simulating a novel Random Number Generator (RNG) by Deep Reinforcement Learning and Recurrent Neural Networks.

A PRNG is an algorithm generating pseudo-random numbers and in this research project it is approximated by a Deep Recurrent Neural Network using Reinforcement Learning.
The network is trained to "randomly" generate a novel algorithm by Reinforcement Learning, using a deep agent to solve a partially observable Markov Decision Process.
The formulation decouples the size of the action space and the length of the sequence, improving by a wide margin previous work's approach.
Starting from a seed state (usually of zeros) the agent learns how to "move" in the highly dimensional environment by appending a sequence of bits with fixed length N to its current sequence, in order to reach states with high rewards.
The reward is given by the result of the `NIST test battery <https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf>`_ on the entire sequence computed at the last time step.
The agent can only observe the last appended sequence at each time step and uses a RNN, specifically a LSTM, to memorize previous appended sequences, to devise an appropriate policy.

Link to previous work on `arXiv <https://arxiv.org/abs/1912.11531?context=cs.AI>`_.
Link to previous work `published article <https://www.sciencedirect.com/science/article/pii/S1877050920304944?via%3Dihub>`_.

Additional information about the current approach can be found in the `arXiv article <TODO>`_.

The algorithm used is Proximal Policy Optimization (PPO) with rewards-to-go and Generalized Advantage Estimation (GAE-Lambda) buffer.

**License**

The same of the article.

**Framework used**

- To run the NIST test battery `NistRng <https://github.com/InsaneMonster/NistRng>`_ (`nistrng package <https://pypi.org/project/nistrng/>`_) python implementation framework is used.
- To execute reinforcement learning the framework `USienaRL <https://github.com/InsaneMonster/USienaRL>`_ (`usienarl package <https://pypi.org/project/usienarl/>`_) is used.

**Compatible with Usienarl v0.7.6**

**Backend**

- *Python 3.6*
- *Tensorflow 1.15*