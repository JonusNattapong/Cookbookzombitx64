import numpy as np
from game_environment import GameEnvironment


class TicTacToeEnv(GameEnvironment):
    """
    TicTacToe environment that implements the GameEnvironment interface
    """
    
    def __init__(self, board_size=3):
        """
        Initialize the TicTacToe environment
        
        Args:
            board_size: Size of the board (default is 3x3)
        """
        self.board_size = board_size
        
    def get_initial_state(self):
        """
        Returns an empty board with player 1 to move
        
        Returns:
            Initial state (board, player)
        """
        # Initialize empty board (0 = empty, 1 = player 1, 2 = player 2)
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # Player 1 starts (0 = player 1, 1 = player 2)
        player = 0
        return (board, player)
    
    def get_valid_actions(self, state):
        """
        Returns a binary mask of valid actions
        
        Args:
            state: (board, player)
            
        Returns:
            Binary mask of valid actions (1 = valid, 0 = invalid)
        """
        board, _ = state
        valid_actions = np.zeros(self.board_size * self.board_size, dtype=np.int8)
        
        # Empty cells are valid moves
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:
                    valid_actions[i * self.board_size + j] = 1
                    
        return valid_actions
    
    def step(self, state, action):
        """
        Apply action to state and return new state
        
        Args:
            state: (board, player)
            action: Action index (0-8 for 3x3 board)
            
        Returns:
            New state after applying action
        """
        board, player = state
        
        # Make a copy of the board
        new_board = np.copy(board)
        
        # Apply action
        row = action // self.board_size
        col = action % self.board_size
        new_board[row, col] = player + 1  # 1 for player 1, 2 for player 2
        
        # Switch player
        new_player = 1 - player
        
        return (new_board, new_player)
    
    def is_terminal(self, state):
        """
        Check if the game is over
        
        Args:
            state: (board, player)
            
        Returns:
            True if the game is over, False otherwise
        """
        board, _ = state
        
        # Check rows, columns, and diagonals for a win
        for player_id in [1, 2]:
            # Check rows
            for i in range(self.board_size):
                if np.all(board[i, :] == player_id):
                    return True
            
            # Check columns
            for i in range(self.board_size):
                if np.all(board[:, i] == player_id):
                    return True
            
            # Check diagonals
            if np.all(np.diag(board) == player_id):
                return True
            if np.all(np.diag(np.fliplr(board)) == player_id):
                return True
        
        # Check for a draw (board full)
        if np.all(board != 0):
            return True
        
        return False
    
    def get_reward(self, state):
        """
        Calculate reward for state
        
        Args:
            state: (board, player)
            
        Returns:
            1 if player 1 wins, -1 if player 2 wins, 0 for draw or ongoing
        """
        board, player = state
        
        # Note: player is the next player to move
        # So the previous player is 1 - player
        
        # Check if previous player won
        prev_player = 1 - player
        prev_player_id = prev_player + 1  # 1 or 2
        
        # Check rows
        for i in range(self.board_size):
            if np.all(board[i, :] == prev_player_id):
                return 1.0 if prev_player == 0 else -1.0
        
        # Check columns
        for i in range(self.board_size):
            if np.all(board[:, i] == prev_player_id):
                return 1.0 if prev_player == 0 else -1.0
        
        # Check diagonals
        if np.all(np.diag(board) == prev_player_id):
            return 1.0 if prev_player == 0 else -1.0
        if np.all(np.diag(np.fliplr(board)) == prev_player_id):
            return 1.0 if prev_player == 0 else -1.0
        
        # Draw or ongoing
        return 0.0
    
    def get_encoded_state(self, state):
        """
        Encode state for neural network input
        
        Args:
            state: (board, player)
            
        Returns:
            Encoded state as numpy array with shape (3 * board_size * board_size,)
        """
        board, player = state
        
        # Create 3 planes:
        # - Plane 0: 1 where player 1 has pieces
        # - Plane 1: 1 where player 2 has pieces
        # - Plane 2: 1 everywhere if it's player 1's turn, 0 everywhere if it's player 2's turn
        encoded = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        encoded[0] = (board == 1)
        encoded[1] = (board == 2)
        encoded[2] = player == 0  # 1 if player 1's turn, 0 if player 2's turn
        
        # Flatten the array
        return encoded.flatten()
    
    def action_space_size(self):
        """
        Returns the size of the action space
        
        Returns:
            board_size * board_size (9 for 3x3 board)
        """
        return self.board_size * self.board_size
    
    def get_current_player(self, state):
        """
        Returns the current player
        
        Args:
            state: (board, player)
            
        Returns:
            Player index (0 or 1)
        """
        _, player = state
        return player
    
    def render(self, state):
        """
        Print the board
        
        Args:
            state: (board, player)
        """
        board, player = state
        
        symbols = ['.', 'X', 'O']
        player_names = ['X', 'O']
        
        print(f"Player {player_names[player]}'s turn")
        
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                row.append(symbols[board[i, j]])
            print(' '.join(row))
        print() 