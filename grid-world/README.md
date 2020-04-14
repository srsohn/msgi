## Dependencies

* PyTorch
* gym
* numpy
* [imageio](https://imageio.readthedocs.io/en/stable/installation.html)
* [tensorboardX](https://github.com/lanpa/tensorboardX)


## Installation
```shell
pip install torch gym numpy imageio tensorboardX
```


## Running the code

Run the following commands to run the experiments from the paper:

**MSGI-Meta**
```shell
bash scripts/train_eval_msgi_meta.sh
```

**MSGI-Random**
```shell
bash scripts/eval_msgi_random.sh
```

**Hierarchical RL**
```shell
bash scripts/train_eval_hrl.sh
```

**RL<sup>2**
```shell
bash scripts/train_eval_rl2.sh
```

**Random** and **GRProp + Oracle**
```shell
bash scripts/eval_baselines.sh
```

