import numpy as np
import env_frozen_lake

def value_iteration():
    env, board_size = env_frozen_lake.get_env()
    actions, pi, value_func, theta, gamma, max_espisode = env_frozen_lake.init_model(env, board_size)
    espisode = 0
    while True:
        new_value_func = np.zeros(value_func.shape, dtype=np.float64)
        new_pi = np.zeros((board_size**2, len(actions)), dtype=np.float64)
        delta = -1.0
        espisode += 1
        for s in range(value_func.shape[0]):
            t = 0.0
            max_a = []
            for a in actions:
                next_state, r, _, p_sa_s_next = env_frozen_lake.step(env, s, a)
                # print(done, 'dasds', s, a, next_state)
                if s != next_state:
                    value = p_sa_s_next * (r + gamma * value_func[next_state])
                    if value > t:
                        t = value
                        max_a = [a]
                    elif value == t:
                        max_a.append(a)
            delta = max(delta, abs(t-value_func[s]))
            new_value_func[s] = t
            for a in max_a:
                new_pi[s][a] = 1.0 / len(max_a)
        value_func = new_value_func
        pi = new_pi
        if (delta >= 0 and delta < theta) or espisode == max_espisode:
            print('FINAL', delta, espisode, value_func)
            break
    return pi

pi = value_iteration()
print(pi)
    