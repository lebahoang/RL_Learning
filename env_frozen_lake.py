from typing import cast
import numpy as np
import numpy.typing as npt
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import gymnasium as gym

def get_env() -> tuple[FrozenLakeEnv, int]:
    board_size = 4
    # m = generate_random_map(board_size)
    m = [
        "SHFF",
        "FFHF",
        "HFFH",
        "FHFG"
    ]
    obj = gym.make(
        'FrozenLake-v1',
        desc=m,
        is_slippery=False,
        success_rate=1.0/3.0,
        reward_schedule=(1, 0, 0)
    )
    env = cast(FrozenLakeEnv, obj.unwrapped)
    env.reset(seed=0)
    return env, board_size

def init_model(env: FrozenLakeEnv, board_size: int) -> tuple[list[int],
                                                             np.ndarray[tuple[int,int], np.dtype[np.float64]],
                                                             np.ndarray[tuple[int,], np.dtype[np.float64]],
                                                             float, float, int]:
    # actions: left, down, right, up
    actions = [0, 1, 2, 3]
    # policy
    pi = np.full((board_size**2, len(actions)), 0.25, dtype=np.float64)

    # Value func
    value_func = np.empty((board_size**2, ), dtype=np.float64)
    max_espisode = 1000
    theta = 0.001
    gamma = 0.9
    return actions, pi, value_func, theta, gamma, max_espisode

def step(env: FrozenLakeEnv, s: int, a: int) -> tuple[int, float, bool, float]:
    p_sa_s_next, new_state, reward, terminated = env.P[s][a][0]
    return new_state, float(reward), terminated, p_sa_s_next
