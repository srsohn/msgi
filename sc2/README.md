## Dependencies

* PyTorch
* pySC2
* numpy
* sklearn
* scikit-learn
* tqdm


## Installation

#### Install StarCraft II

In order to run experiments on StarCraft II environment, please follow the [official pySC2 documentation](https://github.com/deepmind/pysc2#get-starcraft-ii) on installing the environment (For Linux users, make sure to download the SC2 version 4.7.1. Also note that the Linux package requires GLIBC version >= 2.18)

#### Add custom StarCraft II scenarios

After the installation, move the custom SC2 scenarios to `StarCraftII/Maps/mini_games` directory:

```shell
cd MSGI
cp custom_maps/*.SC2Map [SC2_dir]/StarCraftII/Maps/mini_games/
```

#### Install PySC2

Extract, unzip, and install our modified pySC2 from `data` directory:

```shell
cp data/pysc2.zip ../ && unzip ../pysc2.zip
cd ../pysc2
pip install -e .
```



## Running the code

In order to run MSGI, first run the following to save ILP:

```shell
python main.py --map BuildBattleCruiser_20 --meta MSGI --save_ilp --tr_epi 20 --num_iter 1 --num_timesteps 25000 --run_id 1
```

And then, run the following:

```shell
python main.py --map BuildBattleCruiser_20 --meta MSGI --load_ilp --tr_epi 20 --num_iter 1 --num_timesteps 25000 --run_id 1
```



