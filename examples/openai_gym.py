# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import json
import logging
import os
import time
import sys

import numpy as np
from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000

# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-i', '--import-modules', help="Import module(s) required for environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('-sl', '--sleep', type=float, default=None, help="Slow down simulation by sleeping for x seconds (fractions allowed).")
    parser.add_argument('--job', type=str, default=None, help="For distributed mode: The job type of this agent.")
    parser.add_argument('--task', type=int, default=0, help="For distributed mode: The task index of this agent.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.import_modules is not None:
        for module in args.import_modules.split(','):
            importlib.import_module(name=module)

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor=args.monitor,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video,
        visualize=args.visualize
    )
    # [sfan] Set mode for env. Env data will be loaded after the mode being set.
    mode = 'train'
    if args.test:
        mode = 'test'
        # [sfan] Env is unwrapped.
        environment.gym = environment.gym.unwrapped
        environment.gym.env = None
    env = environment.gym.env or environment.gym
    if hasattr(env, 'set_mode'):
        env.set_mode(mode)

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
        )
    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r, id_):
        """
        [sfan] Callback function for runner.run() if episode is terminated
        :param r: Runner instance, e.g. runner
        :param id_: runner.id, i.e. the worker's ID in a distributed run (default=0)
        :return: True
        """
        # if r.episode % report_episodes == 0:
        if args.test or r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info('='*50)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.episode, r.episode_timestep, steps_per_second
            ))
            logger.info('-'*50)
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average rewards: {}".
                        format(sum(r.episode_rewards) / len(r.episode_rewards)))
            logger.info("Average of last 500 rewards: {}".
                        format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {}".
                        format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))

            # [sfan] Logging episode stats from the user defined environment
            # [sfan] Profits
            if r.episode_profits:
                logger.info('-'*50)
                logger.info("Episode profit: {}".format(r.episode_profits[-1]))
                logger.info("Average profits: {}".
                            format(sum(r.episode_profits) / len(r.episode_profits)))
                logger.info('-'*50)
                logger.info("Average profits of last 500 episodes: {}".
                            format(sum(r.episode_profits[-500:]) / min(500, len(r.episode_profits))))
                logger.info("Max profits of last 500 episodes: {}".
                            format(max(r.episode_profits[-500:])))
                logger.info("Min profits of last 500 episodes: {}".
                            format(min(r.episode_profits[-500:])))
                logger.info('-'*25)
                hist, bin_edges = np.histogram(r.episode_profits[-500:])
                logger.info("Hist and bin-edges of last 500 profits:")
                logger.info(hist)
                logger.info(bin_edges)
                logger.info('-'*50)
                logger.info("Average profits of last 100 episodes: {}".
                            format(sum(r.episode_profits[-100:]) / min(100, len(r.episode_profits))))
                logger.info("Max profits of last 100 episodes: {}".
                            format(max(r.episode_profits[-100:])))
                logger.info("Min profits of last 100 episodes: {}".
                            format(min(r.episode_profits[-100:])))
                logger.info('-'*25)
                hist, bin_edges = np.histogram(r.episode_profits[-100:])
                logger.info("Hist and bin-edges of last 100 profits:")
                logger.info(hist)
                logger.info(bin_edges)

            # [sfan] Holds
            if r.episode_action_holds:
                logger.info('-'*50)
                logger.info("Episode action-'hold' cnt: {}".format(r.episode_action_holds[-1]))
                logger.info("Average action-'hold' cnt: {:0.4f}".
                            format(sum(r.episode_action_holds) / len(r.episode_action_holds)))
                logger.info("Average of last 500 action-'hold' cnt: {:0.4f}".
                            format(sum(r.episode_action_holds[-500:]) / min(500, len(r.episode_action_holds))))
                logger.info("Average of last 100 action-'hold' cnt: {:0.4f}".
                            format(sum(r.episode_action_holds[-100:]) / min(100, len(r.episode_action_holds))))
            # [sfan] Empties
            if r.episode_action_empties:
                logger.info('-'*50)
                logger.info("Episode action-'empty' cnt: {}".format(r.episode_action_empties[-1]))
                logger.info("Average action-'empty' cnt: {:0.4f}".
                            format(sum(r.episode_action_empties) / len(r.episode_action_empties)))
                logger.info("Average of last 500 action-'empty' cnt: {:0.4f}".
                            format(sum(r.episode_action_empties[-500:]) / min(500, len(r.episode_action_empties))))
                logger.info("Average of last 100 action-'empty' cnt: {:0.4f}".
                            format(sum(r.episode_action_empties[-100:]) / min(100, len(r.episode_action_empties))))

        if args.save and args.save_episodes is not None and not r.episode % args.save_episodes:
            logger.info("Saving agent to {}".format(args.save))
            r.agent.save_model(args.save)

        return True

    runner.run(
        num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished,
        testing=args.test,
        sleep=args.sleep
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
