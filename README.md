# HetGPPO & SND

<img src="https://github.com/matteobettini/vmas-media/blob/main/hetgppo/HETGIPPO_fill.png?raw=true" alt="drawing"/>  

This repository contains the code for the papers:
- [Heterogeneous Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2301.07137)
- [System Neural Diversity: Measuring Behavioral Heterogeneity in Multi-Agent Learning](https://arxiv.org/abs/2305.02128) 


### Cite

If you use HetGPPO in your research, **cite** it using:
```
@inproceedings{bettini2023hetgppo,
  title = {Heterogeneous Multi-Robot Reinforcement Learning},
  author = {Bettini, Matteo and Shankar, Ajay and Prorok, Amanda},
  year = {2023},
  booktitle = {Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems},
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
  series = {AAMAS '23}
}
```
If you use SND in your research, **cite** it using:
```
@article{bettini2023snd,
  title={System Neural Diversity: Measuring Behavioral Heterogeneity in Multi-Agent Learning},
  author={Bettini, Matteo and Shankar, Ajay and Prorok, Amanda},
  journal={arXiv preprint arXiv:2305.02128},
  year={2023}
}
```

### Videos
Watch the presentation video of HetGPPO.

<p align="center">

[![HetGPPO Video](https://img.youtube.com/vi/J81IVQEy-zw/0.jpg)](https://www.youtube.com/watch?v=J81IVQEy-zw)
</p>
Watch the talk at AAMAS 2023 about HetGPPO.
<p align="center">

[![HetGPPO Talk]()]()
</p>

### How to use

#### Install

Clone the repository using
```bash
git clone --recursive https://github.com/proroklab/HetGPPO.git
cd HetGPPO
```
Create a conda environment and install the dependencies
```bash
pip install "ray[rllib]"==2.1.0
pip install -r requirements.txt
```

#### Run

The training scripts to run HetGPPO in the various [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) scenarios can be found in the `train` folder.

These scripts use the HetGPPO model in `models/gppo.py` to train multiple agents in different scenarios. The scripts log to wandb.

You can run them with:
```bash
python train/train_flocking.py
```

The hyperparameters for each script can be changed according to the user needs.

Several util tools can be found in `utils.py`, including the callback to compute the SND heterogeneity metric.

Several evaluation tools can be found in the `evaluate` folder.



