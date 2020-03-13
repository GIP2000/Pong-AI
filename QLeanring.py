import numpy as np 

class QLearningHandler:
    def __init__(self,
                actions,
                max_values,
                min_values,
                discrete_observation_size,
                learning_rate=0.1,
                discount=0.95,
                epsilon = 1,
                episodes=25000,
                start_epsilon_decaying=1,
                use_epsilon=True,
                min_reward=-2,
                max_reward=0):

        self.actions = [i for i in range(actions)]
        self.min_value = np.asarray(min_values)
        max_values = np.asarray(max_values)
        self.discrete_observation_size = np.asarray(discrete_observation_size)
        self.size = (max_values - self.min_value)/self.discrete_observation_size
        print(tuple(self.discrete_observation_size) + (actions,))
        self.q_table = np.random.uniform(low=int(min_reward),high=int(max_reward),size=(tuple(self.discrete_observation_size) + (actions,)))
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.end_epsilon_decaying = episodes // 2
        self.start_epsilon_decaying = start_epsilon_decaying
        self.use_epsilon = use_epsilon
        self.episode = 0

    
    def get_discrete_state(self,state):
        discrete_state = (np.asarray(state) - self.min_value)/self.size
        return tuple(discrete_state.astype(np.int))
    
    def choose_action(self,state):
        """
        @param state = this is the non discrete state 
        """

        if np.random.random() > self.epsilon or not self.use_epsilon:
            action = np.argmax(self.q_table[self.get_discrete_state(state)])
        else:
            action = np.random.randint(low=self.actions[0],high=self.actions[-1])
        return action 
    
    def update_q(self,state,next_state,action,reward):
        discrete_state = self.get_discrete_state(state)
        next_discrete_state = self.get_discrete_state(next_state)
        max_future_q = np.max(self.q_table[next_discrete_state])
        current_q = self.q_table[discrete_state + (action, )]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
        self.q_table[discrete_state + (action, )] = new_q
    
    def epsilon_decay(self):
        if self.end_epsilon_decaying >= self.episode >= self.start_epsilon_decaying:
            self.epsilon -= self.epsilon_decay_value
            self.episode+=1
        
    def winning_move(self,state,action):
        discrete_state = self.get_discrete_state(state)
        self.q_table[discrete_state + (action, )] = 0 

    
    

