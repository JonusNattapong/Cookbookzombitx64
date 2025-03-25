import argparse
import torch
import os
import numpy as np
import random

from neural_network import MCTSNet, EnhancedMCTSNet
from tictactoe_env import TicTacToeEnv
from mcts import MCTS
from parallel_mcts import ParallelMCTS
from trainer import SelfPlayTrainer
from knowledge_graph import KnowledgeGraph


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and play with MCTS + Neural Networks')
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'evaluate'], 
                       default='train', help='Mode: train, play, or evaluate')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to model checkpoint for play/evaluate mode')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced neural network architecture')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel MCTS for self-play')
    parser.add_argument('--num_iterations', type=int, default=100, 
                       help='Number of training iterations')
    parser.add_argument('--hidden_size', type=int, default=256, 
                       help='Hidden layer size for neural network')
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--use_knowledge', action='store_true', 
                       help='Use knowledge graph')
    parser.add_argument('--knowledge_size', type=int, default=10, 
                       help='Knowledge embedding size')
    parser.add_argument('--num_self_play', type=int, default=100, 
                       help='Number of self-play games per iteration')
    parser.add_argument('--mcts_sims', type=int, default=800, 
                       help='Number of MCTS simulations per move')
    parser.add_argument('--num_processes', type=int, default=4, 
                       help='Number of processes for parallel MCTS')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(args, input_size, action_size):
    """Create neural network model"""
    if args.enhanced:
        return EnhancedMCTSNet(
            input_size=input_size,
            action_size=action_size,
            hidden_size=args.hidden_size,
            knowledge_size=args.knowledge_size if args.use_knowledge else 0
        )
    else:
        return MCTSNet(
            input_size=input_size,
            action_size=action_size,
            hidden_size=args.hidden_size,
            knowledge_size=args.knowledge_size if args.use_knowledge else 0
        )


def load_model(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def play_game(model, game_env, num_simulations=800, human_player=0):
    """
    Play a game against the model
    
    Args:
        model: Neural network model
        game_env: Game environment
        num_simulations: Number of MCTS simulations per move
        human_player: Player index for human (0 = first player, 1 = second player)
    """
    # Create MCTS
    mcts = MCTS(
        model=model,
        game_env=game_env,
        num_simulations=num_simulations,
        dirichlet_noise=False,  # No exploration noise in human games
        temperature=0.1  # Low temperature for deterministic play
    )
    
    # Initialize game
    state = game_env.get_initial_state()
    game_env.render(state)
    
    # Play game
    while not game_env.is_terminal(state):
        current_player = game_env.get_current_player(state)
        
        if current_player == human_player:
            # Human player's turn
            while True:
                try:
                    if game_env.board_size == 3:
                        print("Enter move (0-8):")
                    else:
                        print(f"Enter row (0-{game_env.board_size-1}) and column (0-{game_env.board_size-1}) separated by space:")
                    
                    user_input = input().strip()
                    
                    if ' ' in user_input:
                        row, col = map(int, user_input.split())
                        action = row * game_env.board_size + col
                    else:
                        action = int(user_input)
                    
                    # Check if action is valid
                    valid_actions = game_env.get_valid_actions(state)
                    if action >= 0 and action < len(valid_actions) and valid_actions[action] == 1:
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Try again.")
        else:
            # AI player's turn
            print("AI thinking...")
            action_probs = mcts.search(state)
            action = np.argmax(action_probs)
            print(f"AI chose move: {action}")
        
        # Apply action
        state = game_env.step(state, action)
        game_env.render(state)
    
    # Game over
    reward = game_env.get_reward(state)
    
    if reward == 0:
        print("Game ended in a draw!")
    elif (reward == 1 and human_player == 0) or (reward == -1 and human_player == 1):
        print("You won!")
    else:
        print("AI won!")


def main():
    """Main function"""
    args = parse_args()
    set_seed(args.seed)
    
    # Create game environment
    game_env = TicTacToeEnv(board_size=3)
    
    # Determine input and action sizes
    state = game_env.get_initial_state()
    encoded_state = game_env.get_encoded_state(state)
    input_size = len(encoded_state)
    action_size = game_env.action_space_size()
    
    # Create or load model
    model = create_model(args, input_size, action_size)
    if args.model_path and os.path.exists(args.model_path):
        model = load_model(model, args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Created new model")
    
    # Create knowledge graph if needed
    knowledge_graph = None
    if args.use_knowledge:
        knowledge_graph = KnowledgeGraph(embedding_size=args.knowledge_size)
        # Here you would initialize the knowledge graph with domain knowledge
        # This is just a placeholder
        print("Created knowledge graph")
    
    # Different modes
    if args.mode == 'train':
        # Configure training
        config = {
            'num_iterations': args.num_iterations,
            'num_self_play_games': args.num_self_play,
            'num_mcts_simulations': args.mcts_sims,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'use_parallel_mcts': args.parallel,
            'num_processes': args.num_processes
        }
        
        # Create trainer and train
        trainer = SelfPlayTrainer(model, game_env, config)
        trainer.train()
        
    elif args.mode == 'play':
        # Play against the model
        play_game(model, game_env, num_simulations=args.mcts_sims)
        
    elif args.mode == 'evaluate':
        # Evaluate the model
        if args.model_path:
            # Configure training (for evaluation only)
            config = {
                'num_iterations': 1,
                'num_self_play_games': 1,
                'num_mcts_simulations': args.mcts_sims,
                'eval_games': 100,  # More evaluation games
                'use_parallel_mcts': args.parallel,
                'num_processes': args.num_processes
            }
            
            # Create trainer and evaluate
            trainer = SelfPlayTrainer(model, game_env, config)
            trainer.evaluate()
        else:
            print("Model path must be provided for evaluation")


if __name__ == "__main__":
    main() 