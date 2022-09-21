import torch as th
import numpy as np

if __name__=="__main__":
    path = "/home/quanyi/legged_gym/logs/flat_anymal_c/Aug01_16-05-26_/model_300.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("anymal_heading_tanh.npz", **ret)
    print("model is converted and saved!")
