# Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies
This repository is an implementation of our [ICLR 2020 Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies](https://arxiv.org/abs/2001.00248) in PyTorch. If you use this work, please cite:
```
@inproceedings{
        Sohn2020Meta,
        title={Meta Reinforcement Learning with Autonomous Inference of Subtask Dependencies},
        author={Sungryull Sohn and Hyunjae Woo and Jongwook Choi and Honglak Lee},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=HkgsWxrtPB}
}
```

Our codebase is built on top of [subtask-graph-execution](https://github.com/srsohn/subtask-graph-execution) and [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) for Mining & Playground and the older (deprecated) version of the current [Reaver](https://github.com/inoryy/reaver) for StarCraft II domain.

We have provided two separate codebases for each of our grid-world and StarCraft II domains inside the `grid-world` and `sc2` directories. Please refer to README in each directory for instructions on installation and running the code.
