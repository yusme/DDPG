# DDPG

Continuous control with deep reinforcement learning

[http://arxiv.org/abs/1509.02971](http://arxiv.org/abs/1509.02971)

The goal of these algorithms is to perform policy iteration
by alternatively performing policy evaluation 
on the current policy with Q-learning, and then improving upon the
current policy by following the policy gradient

## TODO

- Batch Normalization 
- Prioritized Experience Replay (https://arxiv.org/abs/1511.05952): to replay important transitions from reply Memory more frequently
