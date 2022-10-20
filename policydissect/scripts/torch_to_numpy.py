import torch as th
import numpy as np

if __name__ == "__main__":
    path = "../logs/flat_anymal_c/Oct20_11-25-34_/model_300.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("anymal_only_forward.npz", **ret)
    print("model is converted and saved!")
