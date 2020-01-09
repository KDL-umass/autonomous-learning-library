import torch
import numpy as np
from all.logging import DummyWriter
from ._agent import Agent


class C51(Agent):
    """
    Implementation of C51, a categorical DQN agent

    The 51 refers to the number of atoms used in the
    categorical distribution used to estimate the
    value distribution. Thought this is the canonical
    name of the agent, this agent is compatible with
    any number of atoms.

    Also note that this implementation uses a "double q"
    style update, which is believed to be less prone
    towards overestimation.
    """

    def __init__(
            self,
            q_dist,
            replay_buffer,
            exploration=0.02,
            discount_factor=0.99,
            minibatch_size=32,
            replay_start_size=5000,
            update_frequency=1,
            eps=1e-5, # stability parameter for loss
            writer=DummyWriter(),
    ):
        # objects
        self.q_dist = q_dist
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.eps = eps
        self.exploration = exploration
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        if self._should_explore():
            return torch.randint(
                self.q_dist.n_actions, (len(state),), device=self.q_dist.device
            )
        return self._best_actions(state)

    def _should_explore(self):
        return (
            len(self.replay_buffer) < self.replay_start_size
            or np.random.rand() < self.exploration
        )

    def _best_actions(self, states):
        probs = self.q_dist.eval(states)
        q_values = (probs * self.q_dist.atoms).sum(dim=2)
        return torch.argmax(q_values, dim=1)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            states, actions, rewards, next_states, weights = self.replay_buffer.sample(self.minibatch_size)
            actions = torch.cat(actions)
            # forward pass
            dist = self.q_dist(states, actions)
            # compute target distribution
            target_dist = self._compute_target_dist(next_states, rewards)
            # compute loss
            kl = self._kl(dist, target_dist)
            loss = (weights * kl).mean()
            # backward pass
            self.q_dist.reinforce(loss)
            # update replay buffer priorities
            self.replay_buffer.update_priorities(kl.detach())
            # debugging
            self.writer.add_loss(
                "q_mean", (dist.detach() * self.q_dist.atoms).sum(dim=1).mean()
            )

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

    def _compute_target_dist(self, states, rewards):
        actions = self._best_actions(states)
        dist = self.q_dist.target(states, actions)
        shifted_atoms = (
            rewards.view((-1, 1)) + self.discount_factor * self.q_dist.atoms
        )
        return self.q_dist.project(dist, shifted_atoms)

    def _kl(self, dist, target_dist):
        log_dist = torch.log(torch.clamp(dist, min=self.eps))
        log_target_dist = torch.log(torch.clamp(target_dist, min=self.eps))
        return (target_dist * (log_target_dist - log_dist)).sum(dim=-1)