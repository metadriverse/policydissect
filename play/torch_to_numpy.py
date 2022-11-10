import torch as th
import numpy as np

if __name__ == "__main__":
    path = "../policydissect/logs/flat_anymal_c/Nov10_15-04-03_/model_300.pt"
    weights = th.load(path)["model_state_dict"]
    ret = {}
    for layer, weight in weights.items():
        ret[layer] = weight.detach().cpu().numpy()
    np.savez_compressed("../policydissect/weights/anymal_forward_tanh.npz", **ret)
    print("model is converted and saved!")
