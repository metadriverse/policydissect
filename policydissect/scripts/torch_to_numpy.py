import torch as th
import numpy as np

if __name__ == "__main__":
    path = "../logs/flat_anymal_c/Oct20_12-09-57_/model_300.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("../weights/anymal_forward_tanh.npz", **ret)
    print("model is converted and saved!")
