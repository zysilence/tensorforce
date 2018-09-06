import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class XauusdEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, start=None, end=None, period=1800, window_size=31):
        self.start = start
        self.end = end
        self.period = period
        self.window_size = window_size

        self.action_space = spaces.Discrete(3)
        # 'asset': (cash, asset)
        self.observation_space = spaces.Dict({
            'history': spaces.Box(low=0, high=10, shape=(self.window_size, 4), dtype=np.float32),
            'asset': spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        })

        self.seed()
        self.viewer = None
        # self.state: (open, close, high, low, cash, asset);
        #             type: tuple
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :param action: 0 hold; 1 buy 2%; 0 sell 2%
        :return: state, reward, done, info
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        done = False

        if 0 == action:
            self.state = ()

        elif 1 == action:
            pass
        else:
            pass
        self.state = ()

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                            "You should always call 'reset()' once you receive 'done = True' -- any further steps are "
                            "undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
