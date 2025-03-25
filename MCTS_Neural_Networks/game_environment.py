from abc import ABC, abstractmethod
import numpy as np


class GameEnvironment(ABC):
    """
    Abstract class for game environments to use with MCTS
    Any game that wants to work with the MCTS algorithm should implement this interface
    """
    
    @abstractmethod
    def get_initial_state(self):
        """
        Returns the initial state of the game
        
        Returns:
            Initial state
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, state):
        """
        Returns a binary mask of valid actions for the given state
        
        Args:
            state: Current game state
            
        Returns:
            Binary mask of valid actions (1 = valid, 0 = invalid)
        """
        pass
    
    @abstractmethod
    def step(self, state, action):
        """
        Applies an action to the state and returns the new state
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            New state after applying the action
        """
        pass
    
    @abstractmethod
    def is_terminal(self, state):
        """
        Checks if the state is terminal (game over)
        
        Args:
            state: Current state
            
        Returns:
            True if the state is terminal, False otherwise
        """
        pass
    
    @abstractmethod
    def get_reward(self, state):
        """
        Returns the reward for the current state
        
        Args:
            state: Current state
            
        Returns:
            Reward value (e.g., 1 for win, -1 for loss, 0 for draw)
        """
        pass
    
    @abstractmethod
    def get_encoded_state(self, state):
        """
        Encodes the state for neural network input
        
        Args:
            state: Current state
            
        Returns:
            Encoded state as numpy array
        """
        pass
    
    @abstractmethod
    def action_space_size(self):
        """
        Returns the size of the action space
        
        Returns:
            Integer representing the number of possible actions
        """
        pass
    
    def get_next_player(self, state):
        """
        Returns the player to move in the given state
        
        Args:
            state: Current state
            
        Returns:
            Player index (0, 1, etc.)
        """
        # Default implementation assumes alternating players
        # Override this method if your game has different player turn logic
        return 1 - self.get_current_player(state)
    
    @abstractmethod
    def get_current_player(self, state):
        """
        Returns the current player in the given state
        
        Args:
            state: Current state
            
        Returns:
            Player index (0, 1, etc.)
        """
        pass
    
    def clone_state(self, state):
        """
        Creates a deep copy of the state
        
        Args:
            state: State to clone
            
        Returns:
            Cloned state
        """
        # Default implementation uses numpy copy
        # Override this method if your state requires special copying
        return np.copy(state)
    
    def get_state_str(self, state):
        """
        Converts state to string representation
        
        Args:
            state: State to convert
            
        Returns:
            String representation of state
        """
        # Default implementation uses string conversion
        # Override this method for custom string representation
        return str(state)
    
    def render(self, state):
        """
        Renders the state for visualization
        
        Args:
            state: State to render
        """
        # Default implementation does nothing
        # Override this method for custom rendering
        pass 