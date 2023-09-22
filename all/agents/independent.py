from ._multiagent import Multiagent
import numpy as np

class IndependentMultiagent(Multiagent):
    def __init__(self, agents):
        self.agents = agents
        self._prev_action = None
        self._action = None

    def act(self, state):
        self._action = self.agents[state['agent']].act(state)
        if self._prev_action != None:
            self._action = np.random.choice([self._action, self._prev_action], p=[0.75,0.25])
        self._prev_action = self._action
        return self._action
