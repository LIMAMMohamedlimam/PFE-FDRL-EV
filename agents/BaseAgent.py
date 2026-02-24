class BaseAgent:
    """
    Abstract base class to ensure extensibility for Federated Learning.
    Future LoRA/DQN agents will inherit from this.
    """
    def get_action(self, state):
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state):
        raise NotImplementedError
        
    def get_parameters(self):
        """For FL Aggregation: Returns weights (NN) or Q-table"""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """For FL Aggregation: Updates local model with global weights"""
        raise NotImplementedError