import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and
# https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = x
        batch_var = 0
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var,
            batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var,
                                       batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self,
                 max_size=1e6,
                 obs_shape=None,
                 discount=0.99,
                 norm_obs=False,
                 norm_ret=False):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

        if norm_obs:
            self.obs_rms = RunningMeanStd(shape=obs_shape)

        if norm_ret:
            self.discount = discount
            self.returns = 0.0
            self.ret_rms = RunningMeanStd(shape=())

    def filter_obs(self, obs):
        if hasattr(self, 'obs_rms'):
            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -5, 5)
        else:
            return obs

    def filter_ret(self, rew):
        if hasattr(self, 'ret_rms'):
            return rew / np.sqrt(self.ret_rms.var + 1e-8)
        else:
            return rew

    def add(self, data):
        if hasattr(self, 'obs_rms'):
            self.obs_rms.update(data[0])

        if hasattr(self, 'ret_rms'):
            self.returns = self.returns * (1 - data[-1]) * self.discount + data[-2]
            self.ret_rms.update(self.returns)

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            states.append(np.array(state, copy=False))
            next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        return self.filter_obs(np.array(states)), self.filter_obs(
            np.array(next_states)), np.array(actions), self.filter_ret(
                np.array(rewards).reshape(-1, 1)), np.array(dones).reshape(
                    -1, 1)
