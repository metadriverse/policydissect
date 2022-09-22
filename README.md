# Policy Dissection

[NeurIPS 2022] Official implementation of the paper: **Human-AI Shared Control via Frequency-based Policy Dissection**

[**Webpage**](https://metadriverse.github.io/policydissect/) |
[**Code**](https://github.com/metadriverse/policydissect) |
[**Video**](https://youtu.be/2Shqhwgom3A) |
[**Paper**](https://arxiv.org/pdf/2206.00152.pdf) |

[comment]: <> ([**Poster**]&#40;https://github.com/decisionforce/HACO/blob/main/docs/iclr_poster.pdf&#41; )

Currently, we provide some interactive neural controllers enabled by *Policy Dissection*.
The policy dissection method and training code will be updated soon.

**Environments**:

- [x] MetaDrive
- [x] Pybullet-Quadrupedal Robot (Forked from: https://github.com/Mehooz/vision4leg.git)
- [x] Isaacgym-Cassie (Forked from: https://github.com/leggedrobotics/legged_gym)
- [x] Isaacgym-ANYmal (Forked from: https://github.com/leggedrobotics/legged_gym)
- [ ] Gym-Walker
- [ ] Gym-BipedalWalker
- [ ] Gym-Ant

## Installation
### Basic Installation
```bash
# Clone the code to local
git clone https://github.com/metadriverse/policydissect.git
cd policydissect

# Create virtual environment
conda create -n policydissect python=3.7
conda activate policydissect

# install torch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install basic dependency
pip install -e .
```

### IsaacGym Installation (Optional) 
For playing with agents trained in IsaacGym, follow the instructions below to install IsaacGym 
- Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
- cd ```isaacgym/python && pip install -e .``` 


## Play

### MetaDrive

Run```python policydissect/scripts/play_metadriv.py``` to collaborate with the AI driver.
Press ```w```,```a```,```s```,```d``` for triggering lane following, left/right lane changing and braking, and
```r``` for resetting the environment.

### Pybullet Quadrupedal Robot

The quadrupedal robot is trained with the code provided by https://github.com/Mehooz/vision4leg.git.
For playing with legged robot, run```python policydissect/scripts/play_quadrupedal.py```.
Press ```w```,```a```,```s```,```d``` for changing the moving direction. and ```r``` for reset.
Also, you can collaborate with AI and challenge the hard environment consisting of obstacles and challenging terrains by
running ```python policydissect/scripts/play_quadrupedal.py --hard```
You can change to a different environment by adding ```--seed your_seed_int_type```.
*tips: Avoid running fast!*

### IsaacGym Cassie
The Cassie robot is trained with the code provided by https://github.com/leggedrobotics/legged_gym with a fixed forward 
command ```[1, 0, 0]```, and thus can only move forward. By applying *Policy Dissection*, primitives related to yaw
rate, forward speed, height control and torque force can be identified. Activating these primitives
enable various skills like crouching, forward jumping, back-flipping and so on.
Run```python policydissect/scripts/play_quadrupedal.py``` for playing with the robot. Add flag```--parkour```to launch 
a challenging parkour environment.
```
Keymap:
- KEY_W:Forward
- KEY_A:Left
- KEY_S:Stop
- KEY_C:Crouch
- KEY_X:Tiptoe
- KEY_Q:Jump
- KEY_D:Right
- KEY_SPACE:Back Flip
- KEY_R:Reset
```
*tips: Switch to Tiptoe state before pressing Key_Q for increasing the distance of jump*

### Comparison with explicit goal-conditioned control
To measure the coarseness of the control approach enabled by *Policy Dissection*, we train a goal-conditioned
quadrupedal ANYmal robot controller with code provided by https://github.com/leggedrobotics/legged_gym. We build 
primitive-activation conditional control system on this controller with a PID 
controller determining the unit output according to the tracking error. As a result, it can track the target yaw command
and can achieve the similar control precision, compared to explicitly indicating the goal in the network input. 

The experiment script can be found at ```policydissect/scripts/run_tracking_experiment.py```. The default yaw tracking is 
achieved by explicit goal-conditioned control, while running ```python policydissect/scripts/run_tracking_experiment.py --primitive_activation```
will change to primitive-activation conditional control.


## Reference

```
@article{li2022human,
  title={Human-AI Shared Control via Frequency-based Policy Dissection},
  author={Li, Quanyi and Peng, Zhenghao and Wu, Haibin and Feng, Lan and Zhou, Bolei},
  journal={arXiv preprint arXiv:2206.00152},
  year={2022}}
```
