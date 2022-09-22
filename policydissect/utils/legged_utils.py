import random

import numpy as np

params = {
    'env_name': 'A1MoveGround',
    'env': {
        'reward_scale': 1,
        'obs_norm': True,
        'horizon': 2500,
        'env_build': {
            'z_constrain': False,
            'motor_control_mode': 'POSITION',
            'other_direction_penalty': 0,
            'z_penalty': 1,
            'clip_num': [0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5],
            'time_step_s': 0.0025,
            'num_action_repeat': 16,
            'add_last_action_input': True,
            'no_displacement': False,
            'diagonal_act': True,
            'get_image': False,
            'depth_image': False,
            'depth_norm': False,
            'rgbd': False,
            'grayscale': False,
            'alive_reward': -0.05,
            'fall_reward': -20,
            'fric_coeff': [1, 0.01, 0.01],
            'target_vel': 1.0,
            'random_init_range': 1.0,
            'domain_randomization': False,
            'enable_action_interpolation': False,
            'enable_action_filter': False,
            'terrain_type': 'random_blocks_sparse_and_heightfield',
            'frame_extract': 1,
            'get_image_interval': 1,
            'enable_rendering': True,
            'record_video': False
        }
    },
    'replay_buffer': {
        'size': 16384,
        'time_limit_filter': True
    },
    'policy': {
        'tanh_action': True
    },
    'encoder': {
        'hidden_shapes': [256, 256]
    },
    'net': {
        'append_hidden_shapes': [256, 256],
        'hidden_shapes': [256, 256]
    },
    'collector': {
        'epoch_frames': 16384,
        'max_episode_frames': 999,
        'eval_episodes': 2
    },
    'general_setting': {
        'discount': 0.99,
        'num_epochs': 1500,
        'batch_size': 1024,
        'gae': True,
        'save_interval': 50,
        'eval_interval': 10
    },
    'ppo': {
        'plr': 0.0001,
        'vlr': 0.0001,
        'clip_para': 0.2,
        'opt_epochs': 3,
        'tau': 0.95,
        'shuffle': True,
        'entropy_coeff': 0.005
    }
}


def seed_env(env, seed):
    env.eval()
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
