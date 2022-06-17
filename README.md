# Policy Dissection

Official implementation of the paper: **Human-AI Shared Control via Frequency-based Policy Dissection** 

[**Webpage**](https://metadriverse.github.io/policydissect/) | 
[**Code**](https://github.com/metadriverse/policydissect) | 
[**Video**](https://youtu.be/2Shqhwgom3A) |
[**Paper**](https://arxiv.org/pdf/2206.00152.pdf) |

[comment]: <> ([**Poster**]&#40;https://github.com/decisionforce/HACO/blob/main/docs/iclr_poster.pdf&#41; )

Currently, we provide some interactive neural controllers enabled by *Policy Dissection*. 
The policy dissection method and training code will be updated soon.
Please run ```python policydissect/play/play_[env].py``` to play with these neural controllers.


**Environments**:

- [x] MetaDrive
- [x] Pybullet-Quadrupedal Robot (Forked from: https://github.com/Mehooz/vision4leg.git)
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

