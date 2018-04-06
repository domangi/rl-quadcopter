import numpy as np
from task import Task

class RedBaron_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.score = self.best_score
        self.noise_scale = 0.1

        # Episodes group used to find new best
        self.episodes_used_for_each_learning_step = 10
        self.current_try = []

        # Discretize actions
        self.num_possible_actions = 50
        self.action_unit_size = self.action_range / self.num_possible_actions

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            # add experience to buffer
            self.current_try.append([self.w, self.total_reward])

            # change current weights
            self.reset_episode()
            if self.best_score == -np.inf:
                self.w = self.noise_scale * np.random.normal(size=self.w.shape)
            else:
                self.w = self.best_w + self.noise_scale * np.random.normal(size=self.w.shape)

        if len(self.current_try) == self.episodes_used_for_each_learning_step:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return self.discreteAct(action)

    def discreteAct(self, action):
        integer_action = action / self.action_unit_size
        action = self.action_unit_size * integer_action
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        current_best_reward = -np.inf
        current_best_weights = []

        for weights, reward in self.current_try:
            if reward > current_best_reward:
                current_best_reward = reward
                current_best_weights = weights
        old_best = max(self.best_score, -1000)
        self.score = current_best_reward
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = current_best_weights
        # set noise based on improvement, low improvement high noise
        print("\n")
        print("current best: ", current_best_reward)
        print("old best: ", old_best)
        print("diff: ", (current_best_reward - old_best))
        print("Best weights: ", self.best_w)
        print("\n")

        self.noise_scale = min(10, abs(current_best_reward-old_best)+0.1)
        self.current_try = []
