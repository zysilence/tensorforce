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

    def __init__(self, start=None, end=None, period=1800):
        self.start = start
        self.end = end
        self.period = period

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 10.0)

        self.seed()
        self.viewer = None
        self.state = None  # type: tuple

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
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
