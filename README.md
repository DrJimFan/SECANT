# SECANT: Self-Expert Cloning for Zero-Shot Generalization of Visual Policies

This repository contains the environments in our paper:

> [SECANT: Self-Expert Cloning for Zero-Shot Generalization of Visual Policies](https://linxifan.github.io/secant-site/)<br/>
> <b>ICML 2021</b><br/>
> <b>Linxi "Jim" Fan, Guanzhi Wang, De-An Huang, Zhiding Yu, Li Fei-Fei, Yuke Zhu, Animashree Anandkumar</b><br/>

**Quick Links**:
[[Project Website](https://linxifan.github.io/secant-site/)]
[[Arxiv](http://arxiv.org/abs/2106.09678)]
[[Demo Video](https://linxifan.github.io/secant-site/assets/videos/secant.mp4)]
[[ICML Page](https://proceedings.mlr.press/v139/fan21c.html)]

## Abstract

![framework](https://linxifan.github.io/secant-site/assets/images/algorithm.png)

Generalization has been a long-standing challenge for reinforcement learning (RL). Visual RL, in particular, can be easily distracted by irrelevant factors in high-dimensional observation space.

In this work, we consider robust policy learning which targets zero-shot generalization to unseen visual environments with large distributional shift. We propose <b>SECANT</b>, a novel self-expert cloning technique that leverages image augmentation in two stages to decouple robust representation learning from policy optimization.

Specifically, an expert policy is first trained by RL from scratch with weak augmentations. A student network then learns to mimic the expert policy by supervised learning with strong augmentations, making its representation more robust against visual variations compared to the expert.

Extensive experiments demonstrate that <b>SECANT</b> significantly advances the state of the art in zero-shot generalization across 4 challenging domains. Our average reward improvements over prior SOTAs are: DeepMind Control (<b>+26.5%</b>), robotic manipulation (<b>+337.8%</b>), vision-based autonomous driving (<b>+47.7%</b>), and indoor object navigation (<b>+15.8%</b>).

## Installation

### Install MuJoCo
Please refer to [dm_control](https://github.com/deepmind/dm_control#requirements-and-installation) and [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco) for how to download and set up MuJoCo for [Deepmind Control Suite](https://github.com/deepmind/dm_control#requirements-and-installation) and [RoboSuite](https://github.com/wangguanzhi/robosuite/blob/master/docs/installation.md), respectively. Make sure environment variables, `LD_LIBRARY_PATH` and `MJLIB_PATH` are set correctly.

### Install CARLA

Please refer to the [CARLA doc](docs/carla.md) for installation instructions.

### Install SECANT
```bash
# Create your virtual environment
conda create --name secant python=3.7
conda activate secant

# Install dm_control
pip install dm_control

# Install robosuite adapted for SECANT
pip install git+git://github.com/wangguanzhi/robosuite.git

# Clone this repo
git clone https://github.com/LinxiFan/SECANT
cd SECANT

# Install SECANT
pip install -e .
```


## Environments

* [Deepmind Control Suite](docs/dm_control.md) (`dm_control`): simple vision-based robotic control
* [CARLA](docs/carla.md): autonomous driving simulator
* [RoboSuite](docs/robosuite.md): dexterous robotic manipulation tasks.


## Usage

SECANT follows the Gym API. 

Basic usage:

```python
from secant.envs.dm_control import make_dmc

env = make_dmc(task="walker_walk")
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)    
```

You can try other environments easily.

```python
from secant.envs.robosuite import make_robosuite
from secant.envs.carla import make_carla
```

Please see [examples](examples) and [docs](docs) on how to use each environment.

## Training and Test Environments


<table>
<tr>
<th>environment</th>
<th>make_env</th>
<th>training</th>
<th>test</th>
</tr>
<tr>
<td>
dm_control
</td>
<td>

```python
secant.envs.robosuite.make_dmc
```

</td>
<td>

```python
background="original"
```

</td>
<td>

```python
background="color_easy"
background="color_hard"
background="video[0-9]"
```

</td>
</tr>
<tr>
<td>
carla
</td>
<td>

```python
secant.envs.carla.make_carla
```

</td>
<td>

```python
weather="clear_noon"
```

</td>
<td>

```python
weather="wet_sunset"
weather="wet_cloudy_noon"
weather="soft_rain_sunset"
weather="mid_rain_sunset"
weather="hard_rain_noon"
```

</td>
</tr>
<tr>
<td>
robosuite
</td>
<td>

```python
secant.envs.robosuite.make_robosuite
```

</td>
<td>

```python
mode="train"
    
scene_id=0
```

</td>
<td>

```python
mode="eval-easy"
mode="eval-hard"
mode="eval-extreme"
    
scene_id=[0-9]
```

</td>
</tr>
</table>


## Citation

Thank you so much for your interest in our work! For your convenience, we provide the BibTeX code to cite our ICML 2021 paper:

```bibtex
@InProceedings{pmlr-v139-fan21c,
  title = 	 {SECANT: Self-Expert Cloning for Zero-Shot Generalization of Visual Policies},
  author =       {Fan, Linxi and Wang, Guanzhi and Huang, De-An and Yu, Zhiding and Fei-Fei, Li and Zhu, Yuke and Anandkumar, Animashree},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3088--3099},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/fan21c/fan21c.pdf},
  url = 	 {https://proceedings.mlr.press/v139/fan21c.html},
}
```
