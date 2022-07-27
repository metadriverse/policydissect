import os
import datetime
import os.path as osp
import sys

import numpy as np

from policydissect.utils.legged_utils import get_single_hrl_env

sys.path.append(os.path.join(os.path.dirname(__file__), "../../quadrupedal"))
from policydissect.quadrupedal.vision4leg.get_env import get_subprocvec_env
import random
from policydissect.quadrupedal.torchrl.collector.on_policy import VecOnPolicyCollector
from policydissect.quadrupedal.torchrl.algo import PPO
import policydissect.quadrupedal.torchrl.networks as networks
import policydissect.quadrupedal.torchrl.policies as policies
from policydissect.quadrupedal.torchrl.utils import Logger
from policydissect.quadrupedal.torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from policydissect.quadrupedal.torchrl.utils import get_args
from policydissect.utils.legged_config import hrl_param
import torch
from policydissect.utils.legged_hrl_env import HRLWrapper

args = get_args()


def experiment():
    params = hrl_param
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    params["env"]["env_build"]["enable_rendering"] = False
    params["env"]["env_build"]["terrain_type"] = "random_blocks_sparse_and_heightfield"
    HRLWrapper.set_repeat(args.action_repeat)
    env = get_subprocvec_env(
        params["env_name"],
        params["env"],
        args.vec_env_nums,
        args.proc_nums,
        env_func=get_single_hrl_env
    )

    eval_env = get_subprocvec_env(
        params["env_name"],
        params["env"],
        max(2, args.vec_env_nums),
        max(2, args.proc_nums),
        env_func=get_single_hrl_env
    )

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    buffer_param = params['replay_buffer']

    experiment_name = "hrl_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M"))
    logger = Logger(
        experiment_name, params['env_name'],
        args.seed, params, args.log_dir, args.overwrite)
    params['general_setting']['env'] = env

    replay_buffer = OnPolicyReplayBuffer(
        env_nums=args.vec_env_nums,
        max_replay_buffer_size=int(buffer_param['size']),
        time_limit_filter=buffer_param['time_limit_filter']
    )
    params['general_setting']['replay_buffer'] = replay_buffer

    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type'] = networks.MLPBase
    # params['net']['activation_func'] = torch.nn.Tanh

    encoder = networks.LocoTransformerEncoder(
        in_channels=env.image_channels,
        state_input_dim=env.observation_space.shape[0],
        **params["encoder"]
    )

    pf = policies.GaussianContPolicyLocoTransformer(
        encoder=encoder,
        state_input_shape=env.observation_space.shape[0],
        visual_input_shape=(env.image_channels, 64, 64),
        output_shape=env.action_space.shape[0],
        **params["net"],
        **params["policy"]
    )

    vf = networks.LocoTransformer(
        encoder=encoder,
        state_input_shape=env.observation_space.shape[0],
        visual_input_shape=(env.image_channels, 64, 64),
        output_shape=1,
        **params["net"]
    )

    print(pf)
    print(vf)

    params['general_setting']['collector'] = VecOnPolicyCollector(
        vf, env=env, eval_env=eval_env, pf=pf,
        replay_buffer=replay_buffer, device=device,
        train_render=False,
        **params["collector"]
    )
    params['general_setting']['save_dir'] = osp.join(
        logger.work_dir, "model")
    agent = PPO(
        pf=pf,
        vf=vf,
        **params["ppo"],
        **params["general_setting"]
    )
    agent.train()


if __name__ == "__main__":
    # python hrl_locotransformer.py --device=cuda:0 --log_dir=xxx --repeat=yyy > 0.log 2>&1 &
    experiment()
