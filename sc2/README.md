## Dependencies

* PyTorch
* pySC2
* numpy
* sklearn
* scikit-learn




## Installation

#### Install StarCraft II

In order to run experiments on StarCraft II environment, please follow the [official pySC2 documentation](https://github.com/deepmind/pysc2#get-starcraft-ii) on installing the environment (For Linux users, make sure to download the SC2 version 4.7.1. Also note that the Linux package requires GLIBC version >= 2.18)

#### Add custom StarCraft II scenarios

After the installation, move the custom SC2 scenarios to `StarCraftII/Maps/mini_games` directory:

```shell
cp custom_maps/*.SC2Map <path_to_sc2>/StarCraftII/Maps/mini_games
```

#### Install PySC2

Extract, unzip, and install our modified pySC2 from `data` directory:

```shell
cp data/pysc2.zip <path_to_pysc2> && cd <path_to_pysc2>
unzip pysc2.zip
cd pysc2
pip install -e .
```



## Running the saved ILP model

To run the saved ILP models for the SC2 scenarios, extract the ILP models from `data` directory:

```shell
cd <path_to_msgi>/msgi/sc2
cp data/ILP_models.zip . && unzip ILP_models.zip
python main.py --map BuildBattleCruiser_20 --meta MSGI --load_ilp --tr_epi 20 --num_timesteps 25000 --run_id 1
```



## Running the code from scratch

In order to run MSGI, first run the following to save ILP:

```shell
python main.py --map BuildBattleCruiser_20 --meta MSGI --save_ilp --tr_epi 20 --num_timesteps 25000 --run_id 1
```

And then, run the following:

```shell
python main.py --map BuildBattleCruiser_20 --meta MSGI --load_ilp --tr_epi 20 --num_timesteps 25000 --run_id 1
```



## Demo videos

https://bit.ly/msgi-videos

