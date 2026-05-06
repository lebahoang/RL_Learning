import numpy as np
import numpy.typing as npt
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import env_frozen_lake

# def print_pi(pi: npt.NDArray[np.float64]):
#     env.reset()
#     init_state = 0
#     while True:

def policy_evaluation(env: FrozenLakeEnv,
                      actions: list[int],
                      value_func: npt.NDArray[np.float64],
                      pi: npt.NDArray[np.float64],
                      theta: float,
                      gamma: float,
                      max_espisode: int) -> npt.NDArray[np.float64]:
    espisode = 0
    new_value_func = np.zeros(value_func.shape, dtype=np.float64)
    while True:
        delta = -1.0
        for s in range(value_func.shape[0]):
            t = 0.0
            for a in actions:
                next_state, r, _, p_sa_s_next = env_frozen_lake.step(env, s, a)
                # print(done, 'dasds', s, a, next_state)
                if s != next_state:
                    value = p_sa_s_next * (r + gamma * value_func[next_state])
                    t += pi[s][a] * value
            delta = max(delta, abs(t-value_func[s]))
            new_value_func[s] = t
        value_func = new_value_func
        espisode += 1
        if (delta >= 0 and delta < theta) or espisode == max_espisode:
            print('FINAL', delta, espisode, value_func)
            return value_func

def policy_improvement(board_size: int,
                       env: FrozenLakeEnv,
                       actions: list[int],
                       value_func: npt.NDArray[np.float64],
                       gamma: float):
    new_pi = np.zeros((board_size**2, len(actions)), dtype=np.float64)
    for s in range(value_func.shape[0]):
        maxV = 0.0
        max_a = []
        for a in actions:
            next_state, r, done, p_sa_s_next = env_frozen_lake.step(env, s, a)
            if s != next_state:
                v = p_sa_s_next * (r + gamma * value_func[next_state])
                if v > maxV:
                    maxV = v
                    max_a = [a]
                elif v == maxV:
                    max_a.append(a)
        for a in max_a:
            new_pi[s][a] = 1.0 / len(max_a)
    return new_pi

def policy_iteration():
    env, board_size = env_frozen_lake.get_env()
    actions, pi, value_func, theta, gamma, max_espisode = env_frozen_lake.init_model(env, board_size)
    while True:
        value_func = policy_evaluation(env, actions, value_func, pi, theta, gamma, max_espisode)
        new_pi = policy_improvement(board_size, env, actions, value_func, gamma)
        # print(pi)
        # print(new_pi)
        # print("--------------------")
        if (new_pi == pi).all():
            break
        pi = new_pi
    return pi

pi = policy_iteration()
print(pi)