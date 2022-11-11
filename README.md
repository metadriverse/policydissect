# Policy Dissection

[NeurIPS 2022] Official implementation of the paper: **Human-AI Shared Control via Policy Dissection**

[**Webpage**](https://metadriverse.github.io/policydissect/) |
[**Code**](https://github.com/metadriverse/policydissect) |
[**Video**](https://youtu.be/7UmScmKMFE4) |
[**Paper**](https://arxiv.org/pdf/2206.00152.pdf) |

[comment]: <> ([**Poster**]&#40;https://github.com/decisionforce/HACO/blob/main/docs/iclr_poster.pdf&#41; )

Currently, we provide some interactive neural controllers enabled by *Policy Dissection*.
The policy dissection method and training code will be updated soon.

**Supported Environments**:

- [x] MetaDrive
- [x] Pybullet-Quadrupedal Robot (Forked from: https://github.com/Mehooz/vision4leg.git)
- [x] Isaacgym-Cassie (Forked from: https://github.com/leggedrobotics/legged_gym)
- [x] Isaacgym-ANYmal (Forked from: https://github.com/leggedrobotics/legged_gym)
- [x] Gym-Walker (Mujoco-200)
- [x] Gym-BipedalWalker (Box2D)
- [x] Gym-Ant (Mujoco-200)

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
- Download and install Isaac Gym Preview 3 from https://developer.nvidia.com/isaac-gym
- cd ```isaacgym/python && pip install -e .```

Please review the file `isaacgym/docs/install.html` for more information on installation.
See the [Troubleshooting](#troubleshooting) section for debugging.


### Mujoco Installation (Optional)

For playing with the Mujoco-Ant and Mujoco-Walker, please
- install **mujoco200** according to https://www.roboti.us/download.html (Mujoco licence can be found at **https://www.roboti.us/license.html**)
- copy contents in the folder to `~/.mujoco/mujoco200`
- copy licence from https://www.roboti.us/license.html to `~/.mujoco/`
- add this line to `.bashrc`: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhenghao/.mujoco/mujoco200/bin`
- run ```pip install mujoco-py==2.0.2.7``` (Solutions of compiling error can be easily found at https://github.com/openai/mujoco-py/issues)

## Play with AI

### MetaDrive

To collaborate with the AI driver in [MetaDrive environment](https://github.com/metadriverse/metadrive), run:

```bash
# MetaDrive
# Keymap:
# - KEY_W: lane following
# - KEY_A: left lane changing
# - KEY_S: braking
# - KEY_D: right lane changing
# - KEY_R:Reset
python play/play_metadrive.py
``` 


### Pybullet Quadrupedal Robot

The quadrupedal robot is trained with the code provided by https://github.com/Mehooz/vision4leg.git.
For playing with legged robot, run:

```bash
# Pybullet Quadrupedal Robot
# Keymap:
# - KEY_W: forward
# - KEY_A: moving left
# - KEY_S: stop
# - KEY_D: moving right
# - KEY_R: reset
python play/play_pybullet_a1.py
python play/play_pybullet_a1.py --hard
python play/play_pybullet_a1.py --hard --seed 1001
```


Also, you can collaborate with AI and challenge the hard environment consisting of obstacles and challenging terrains by
adding `--hard` flag. You can change to a different environment by adding ```--seed your_seed_int_type```.

*tips: Avoid running fast!*

### IsaacGym Cassie

The Cassie robot is trained with the code provided by https://github.com/leggedrobotics/legged_gym with a fixed forward
command ```[1, 0, 0]```, and thus can only move forward. By applying *Policy Dissection*, primitives related to yaw
rate, forward speed, height control and torque force can be identified. Activating these primitives
enable various skills like crouching, forward jumping, back-flipping and so on.
Run the following command to play with the robot. Add flag```--parkour```to launch a challenging parkour environment.

```bash
# Keymap:
# - KEY_W:Forward
# - KEY_A:Left
# - KEY_S:Stop
# - KEY_C:Crouch
# - KEY_X:Tiptoe
# - KEY_Q:Jump
# - KEY_D:Right
# - KEY_SPACE:Back Flip
# - KEY_R:Reset
python play/play_cassie.py
python play/play_cassie.py --parkour
```

*tips: Switch to Tiptoe state before pressing Key_Q to increase the distance of jump.*


> **Note**
> Do not draw the windows or close the pygame window during running.


### Gym Environments

We also discover motor primitives in three gym environments: Box2d-BipedalWalker, Mujoco-Ant and Mujoco-Walker. 
You can try them via:


```bash
# BipedalWalker
# Keymap:
# - KEY_W: jump
# - KEY_A: stand up from split
# - KEY_S: restore running after jumping
# - KEY_R: reset
python play/play_gym_bipedalwalker.py

# Mujoco-Ant
# Keymap:
# - KEY_W: move up
# - KEY_A: move left
# - KEY_S: move down
# - KEY_D: move right
# - KEY_Q: rotation
# - KEY_R: reset
python play/play_mujoco_ant.py
    
# Mujoco-Walker
# Keymap:
# - KEY_R: reset
# - KEY_A: stop
# - KEY_W: freeze red knee
# - KEY_D: restore running
python play/play_mujoco_walker.py
```

### Comparison with explicit goal-conditioned control

To measure the coarseness of the control approach enabled by *Policy Dissection*, we train a goal-conditioned
quadrupedal ANYmal robot controller with code provided by https://github.com/leggedrobotics/legged_gym. We build
primitive-activation conditional control system on this controller with a PID
controller determining the unit output according to the tracking error. As a result, it can track the target yaw command
and can achieve the similar control precision, compared to explicitly indicating the goal in the network input.
Video is available [here](https://metadriverse.github.io/policydissect/#Tracking%20Demo).

The experiment script can be found at ```play/run_tracking_experiment.py```. 
The default yaw tracking is achieved by explicit goal-conditioned control, while
running ```python play/run_tracking_experiment.py --primitive_activation```
will change to primitive-activation conditional control.

## Policy Dissection Examples
In ```example``` folder, we provide two examples showing how to dissect policy. The results can be read by opening
```read_result.ipynb``` with jupyter notebook. Also, the identified units are chosen as motor primitives for evoking 
behaviors of Anymal and the MetaDrive agents. Check previous section about how to play with them.

## Troubleshooting

### Installing IsaacGym

If you encounter `ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory`, run
this:

```bash
export LD_LIBRARY_PATH=/path/to/libpython/directory
# If you are using Conda, the path should be /path/to/conda/envs/your_env/lib.
# For example:
export LD_LIBRARY_PATH=/home/zhenghao/anaconda3/envs/policydissect/lib
```

If you encounter `CalledProcessError: Command '['which', 'c++']' returned non-zero exit status 1.`, try this:
```bash
sudo apt-get install build-essential
```


If you encounter `AttributeError: module 'distutils' has no attribute 'version'` from tensorboard,
try this:
```bash
pip install -U setuptools==50.0.0
```

### Installing Mujoco


If you encounter: `fatal error: GL/osmesa.h: No such file or directory`:

```bash
sudo apt-get install libosmesa6-dev
```

If you encounter: `error: [Errno 2] No such file or directory: 'patchelf': 'patchelf'`:

```bash
sudo apt-get install patchelf
```

If you encounter: `ERROR: GLEW initalization error: Missing GL version`:

```bash
sudo apt-get install -y libglew-dev
```


## Reference

```
@inproceedings{
    li2022humanai,
    title={Human-{AI} Shared Control via Policy Dissection},
    author={Quanyi Li and Zhenghao Peng and Haibin Wu and Lan Feng and Bolei Zhou},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=LCOv-GVVDkp}
}
```