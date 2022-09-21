# Policy Dissection

[NeurIPS 2022] Official implementation of the paper: **Human-AI Shared Control via Frequency-based Policy Dissection** 

[**Webpage**](https://metadriverse.github.io/policydissect/) | 
[**Code**](https://github.com/metadriverse/policydissect) | 
[**Video**](https://youtu.be/2Shqhwgom3A) |
[**Paper**](https://arxiv.org/pdf/2206.00152.pdf) |

[comment]: <> ([**Poster**]&#40;https://github.com/decisionforce/HACO/blob/main/docs/iclr_poster.pdf&#41; )

[//]: # (![alt text]&#40;https://github.com/metadriverse/policydissect/blob/main/docs/assets/overview.png?raw=true&&#41;)

Currently, we provide some interactive neural controllers enabled by *Policy Dissection*. 
The policy dissection method and training code will be updated soon.


**Environments**:

- [x] MetaDrive
- [x] Pybullet-Quadrupedal Robot (Forked from: https://github.com/Mehooz/vision4leg.git)
- [ ] Isaacgym-Cassie
- [ ] Isaacgym-ANYmal
- [ ] Gym-Walker
- [ ] Gym-BipedalWalker
- [ ] Gym-Ant

## Installation
```bash
# Clone the code to local
git clone https://github.com/metadriverse/policydissect.git
cd policydissect

# Create virtual environment
conda create -n haco python=3.7
conda activate policydissect

# Install basic dependency
pip install -e .
```

## Play 
### MetaDrive
Run```python policydissect/play/play_metadriv.py``` to collaborate with the AI driver. 
Press ```w```,```a```,```s```,```d``` for triggering lane following, left/right lane changing and braking, and
```r``` for resetting the environment.

### Quadrupedal Robot
The quadrupedal robot is trained with the code provided by https://github.com/Mehooz/vision4leg.git.
For playing with legged robot, run```python policydissect/play/play_quadrupedal.py```.
Press ```w```,```a```,```s```,```d``` for changing the moving direction. and ```r``` for reset.
Also, you can collaborate with AI and challenge the hard environment consisting of obstacles and challenging terrains by running ```python policydissect/play/play_quadrupedal.py --hard```
You can change to a different environment by adding ```--seed your_seed_int_type```.
*tips: Avoid running fast!*

## Reference
```
@article{li2022human,
  title={Human-AI Shared Control via Frequency-based Policy Dissection},
  author={Li, Quanyi and Peng, Zhenghao and Wu, Haibin and Feng, Lan and Zhou, Bolei},
  journal={arXiv preprint arXiv:2206.00152},
  year={2022}}
```