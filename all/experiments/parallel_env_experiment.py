
import torch
import time
import numpy as np
from all.core import State
from .writer import ExperimentWriter, CometWriter
from .experiment import Experiment
from all.environments import VectorEnvironment
from all.agents import ParallelAgent
import gym


class ParallelEnvExperiment(Experiment):
    '''An Experiment object for training and testing agents that use parallel training environments.'''

    def __init__(
            self,
            preset,
            env,
            name=None,
            train_steps=float('inf'),
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=True,
            writer="tensorboard"
    ):
        self._name = name if name is not None else preset.name
        super().__init__(self._make_writer(logdir, self._name, env.name, write_loss, writer), quiet)
        self._n_envs = preset.n_envs
        if isinstance(env, VectorEnvironment):
            assert self._n_envs == env.num_envs
            self._env = env
        else:
            self._env = env.duplicate(self._n_envs)
        self._preset = preset
        self._agent = preset.agent(writer=self._writer, train_steps=train_steps)
        self._render = render

        # training state
        self._returns = []
        self._frame = 1
        self._episode = 1
        self._episode_start_times = [] * self._n_envs
        self._episode_start_frames = [] * self._n_envs

        # test state
        self._test_episodes = 100
        self._test_episodes_started = self._n_envs
        self._test_returns = []
        self._should_save_returns = [True] * self._n_envs

        if render:
            for _env in self._envs:
                _env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        num_envs = int(self._env.num_envs)
        returns = np.zeros(num_envs)
        state_array = self._env.reset()
        start_time = time.time()
        completed_frames = 0
        while not self._done(frames, episodes):
            action = self._agent.act(state_array)
            state_array = self._env.step(action)
            self._frame += num_envs
            episodes_completed = state_array.done.type(torch.IntTensor).sum().item()
            completed_frames += num_envs
            returns += state_array.reward.cpu().detach().numpy()
            if episodes_completed > 0:
                dones = state_array.done.cpu().detach().numpy()
                cur_time = time.time()
                fps = completed_frames / (cur_time - start_time)
                completed_frames = 0
                start_time = cur_time
                for i in range(num_envs):
                    if dones[i]:
                        self._log_training_episode(returns[i], fps)
                        returns[i] = 0
            self._episode += episodes_completed

    def test(self, episodes=100):
        test_agent = self._preset.test_agent()
        returns = 0
        first_state = self._env.reset()[0]
        eps_returns = []
        while len(eps_returns) < episodes:
            first_action = test_agent.act(first_state)
            if isinstance(self._env.action_space, gym.spaces.Discrete):
                action = torch.tensor([first_action] * self._env.num_envs)
            else:
                action = torch.tensor(first_action).reshape(1, -1).repeat(self._env.num_envs, 1)
            state_array = self._env.step(action)
            dones = state_array.done.cpu().detach().numpy()
            rews = state_array.reward.cpu().detach().numpy()
            first_state = state_array[0]
            returns += rews[0]
            for i in range(1):
                if dones[i]:
                    episode_return = returns
                    esp_index = len(eps_returns)
                    eps_returns.append(episode_return)
                    returns = 0
                    self._log_test_episode(esp_index, episode_return)

        self._log_test(eps_returns)
        return eps_returns

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        if writer == "comet":
            return CometWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
