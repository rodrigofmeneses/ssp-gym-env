#%%
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#%%
class SSP(gym.Env):
    metadata = {'render.modes': ['human']}

    NUM_NODES = 10
    DENSITY = 0.7
    MAX_STEPS = 50

    def __init__(self):
        self.action_space = gym.spaces.Discrete(self.NUM_NODES)
        self.observation_space = gym.spaces.Discrete(self.NUM_NODES)

        self.seed()

        # Possible positions to chose on 'reset()'
        self.init_states = list(range(self.NUM_NODES))
        self.goal = self.np_random.randint(self.NUM_NODES)
        self.init_states.remove(self.goal)

        self.terminal_states = [self.goal]
        self._createModel()
        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.count = 0

        self.state = self.np_random.choice(self.init_states)
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : Discrete
        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        if self.done:
            # should never reach this point
            print('EPISODE DONE!!!')
        elif self.count == self.MAX_STEPS:
            self.done = True
        else:
            assert self.action_space.contains(action)
            self.count += 1

            # insert simulation logic to handle an action...
            
            if action in self.possibleActions(self.state):
                self.reward = -self.rewards[action, self.state]
                if action in self.terminal_states:
                    self.done = True
                    self.reward += 1
                    self.state = action
                else:
                    self.state = np.random.choice(self.NUM_NODES, 
                                            p=self.transitions_prob[action, self.state])
            else:
                self.reward = -100
        try:
            assert self.observation_space.contains(self.state)
        
        except AssertionError:
            print('INVALID STATE', self.state)
        
        return [self.state, self.reward, self.done, self.info]
    
    def _createModel(self):
        """
            Create Graph representation and MDP variables with number of states designated
        """
        # Graph with self.DENSITY probability of create a edge
        self.Graph = nx.erdos_renyi_graph(self.NUM_NODES, self.DENSITY, directed=False)

        # Guarantee of connectivy
        for node in range(self.NUM_NODES):
            if not self.Graph[node]:
                except_node_list = [x for x in range(self.NUM_NODES) if x != node]
                self.Graph.add_edge(node, np.random.choice(except_node_list))

        # Random weights
        for u, v in self.Graph.edges():
            self.Graph.edges[u, v]['weight'] = self.np_random.randint(1, 11)

        self.states = np.array(self.Graph.nodes) # S = State list
        self.possibleActions = lambda s: list(self.Graph[s].keys()) # A(s) = Action function
        self.rewards = np.zeros((self.NUM_NODES, self.NUM_NODES)) # R = Reward list

        for s in self.states:
            for a in self.possibleActions(s):
                self.rewards[s, a] = self.Graph[s][a]['weight']

        self.transitions_prob = np.zeros((self.NUM_NODES, self.NUM_NODES, self.NUM_NODES))

        for s in self.states:
            for s_ in self.states:
                for a in self.possibleActions(s):
                    if s_ == a:
                        self.transitions_prob[s_, s, a] = 1
        
        self.pos = nx.spring_layout(self.Graph)
    

    def render(self):
        # pos = nx.spring_layout(self.Graph)
        nx.draw(self.Graph, self.pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(self.Graph, 'weight')
        nx.draw_networkx_edge_labels(self.Graph, self.pos, edge_labels)
        nx.draw_networkx(self.Graph.subgraph(self.state), pos=self.pos, node_color='red')
        nx.draw_networkx(self.Graph.subgraph(self.goal), pos=self.pos, node_color='green')
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


# %%


# %%
