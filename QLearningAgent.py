from BaseAgent import BaseAgent
import numpy as np

class QLearningAgent(BaseAgent):
    """
    A Tabular Q-Learning Agent.
    Implements the logic requested to start with, extending BaseAgent.
    """
    def __init__(self, action_space_size, state_bins, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = gamma # Discount factor [cite: 95]
        self.epsilon = epsilon
        self.action_space_size = action_space_size
        
        # Q-Table initialization
        # Note: Since state is continuous in the PDF, we use discretization for Q-Learning
        self.state_bins = state_bins 
        # For simplicity, calculating table size based on bins provided
        # Size = (bins_soc * bins_time * ...) x action_space
        self.q_table = {} 

    def _discretize_state(self, state):
        """
        Maps continuous state s_{i,t}  to a tuple key for Q-table.
        """
        # Example discretization logic (simplified)
        soc, t_rem, grid, volt = state[0], state[1], state[2], state[3]
        
        soc_bin = int(soc * 10) # 0-10
        t_bin = int(t_rem)
        grid_bin = int(grid > 0.5) # High/Low congestion
        
        return (soc_bin, t_bin, grid_bin)

    def get_action(self, state, eval_mode=False):
        """
        Ajout du flag eval_mode.
        Si True : Epsilon = 0 (Action purement optimale selon Q-Table).
        """
        state_key = self._discretize_state(state)
        
        # En mode évaluation, on ne fait pas d'exploration aléatoire
        effective_epsilon = 0.0 if eval_mode else self.epsilon
        
        if np.random.random() < effective_epsilon:
            return np.random.randint(0, self.action_space_size)
        
        if state_key not in self.q_table:
            # En test, si état inconnu, on reste safe (ex: action 1 = idle)
            return 1 # Ou logique par défaut
            
        return np.argmax(self.q_table[state_key])

    def update(self, state, action_idx, reward, next_state):
        """
        Updates Q-values.
        Equation: Q(s,a) = Q(s,a) + lr * [r + gamma * max Q(s', a') - Q(s,a)]
        Matches the goal: Maximiser sum(gamma^t * r_{i,t}) [cite: 94]
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)
            
        prediction = self.q_table[state_key][action_idx]
        target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        # Gradient descent step (scalar)
        self.q_table[state_key][action_idx] += self.lr * (target - prediction)

    def get_parameters(self):
        # In a tabular setting, we send the Q-table (or gradients of it)
        return self.q_table