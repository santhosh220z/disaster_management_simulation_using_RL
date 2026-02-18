"""
Main Entry Point for Disaster Management Simulation
Command-line interface for training, evaluation, and simulation
"""

import argparse
import sys
from pathlib import Path

from environment import DisasterEnvironment
from agent import QLearningAgent, ManualPolicy, AdaptiveManualPolicy
from trainer import Trainer, Evaluator, quick_train, compare_with_manual
from visualization import TrainingVisualizer, load_and_visualize
from config import SIMULATION_CONFIG, DISASTER_SCENARIOS


def train_command(args):
    """Handle train command"""
    print("\n" + "="*60)
    print("DISASTER MANAGEMENT RL - TRAINING MODE")
    print("="*60)
    
    # Initialize environment
    env = DisasterEnvironment(
        scenario_name=args.scenario,
        seed=args.seed if args.seed else None
    )
    
    # Initialize agent
    agent = QLearningAgent(
        n_actions=env.n_actions,
        learning_rate=args.alpha,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
    )
    
    # Initialize trainer
    trainer = Trainer(
        n_episodes=args.episodes,
        scenarios=[args.scenario] if args.scenario else None,
        save_dir=args.output,
        verbose=not args.quiet,
    )
    
    # Train
    results = trainer.train(
        agent, 
        env,
        render_frequency=args.render_freq if args.render else 0
    )
    
    # Visualize if requested
    if args.visualize:
        visualizer = TrainingVisualizer(save_dir=args.output)
        visualizer.plot_training_history(trainer.training_history, show=True)
        visualizer.plot_agent_statistics(agent.get_statistics(), show=True)
    
    return results


def evaluate_command(args):
    """Handle evaluate command"""
    print("\n" + "="*60)
    print("DISASTER MANAGEMENT RL - EVALUATION MODE")
    print("="*60)
    
    # Load trained agent
    agent = QLearningAgent(n_actions=25)  # Will be overwritten by load
    agent.load(args.model)
    
    # Initialize environment
    env = DisasterEnvironment(
        scenario_name=args.scenario,
        seed=args.seed if args.seed else None
    )
    
    # Evaluate
    evaluator = Evaluator(n_episodes=args.episodes)
    results = evaluator.evaluate(agent, env, "Trained RL Agent")
    
    print(f"\nEvaluation Results ({args.episodes} episodes):")
    print(f"  Average Reward: {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Average Discharged: {results['avg_discharged']:.1f}")
    print(f"  Average Deaths: {results['avg_deaths']:.1f}")
    
    return results


def compare_command(args):
    """Handle compare command"""
    print("\n" + "="*60)
    print("DISASTER MANAGEMENT RL - COMPARISON MODE")
    print("="*60)
    
    # Load trained agent
    agent = QLearningAgent(n_actions=25)
    agent.load(args.model)
    
    # Compare with manual policies
    results = compare_with_manual(agent, n_episodes=args.episodes)
    
    # Visualize if requested
    if args.visualize:
        visualizer = TrainingVisualizer(save_dir="plots")
        visualizer.plot_comparison(results, show=True)
    
    return results


def simulate_command(args):
    """Handle simulate command (interactive simulation)"""
    print("\n" + "="*60)
    print("DISASTER MANAGEMENT RL - SIMULATION MODE")
    print("="*60)
    
    # Initialize environment
    env = DisasterEnvironment(
        scenario_name=args.scenario,
        seed=args.seed if args.seed else None
    )
    
    # Initialize agent
    if args.model:
        agent = QLearningAgent(n_actions=env.n_actions)
        agent.load(args.model)
        print(f"Loaded agent from {args.model}")
    elif args.manual:
        agent = ManualPolicy(strategy=args.manual)
        print(f"Using manual policy: {args.manual}")
    else:
        agent = QLearningAgent(n_actions=env.n_actions)
        print("Using untrained RL agent")
    
    # Run simulation
    state = env.reset()
    total_reward = 0
    
    print(f"\nRunning simulation for {args.steps} steps...")
    print(env.render())
    
    for step in range(args.steps):
        state_tuple = tuple(state.tolist())
        action = agent.get_action(state_tuple, training=False)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if args.verbose or (step + 1) % 10 == 0:
            print(f"\nStep {step + 1}:")
            print(f"  Action: {action} | Reward: {reward:.1f} | Total Reward: {total_reward:.1f}")
            print(f"  Discharged: {info['discharged_this_step']} | Deaths: {info['deaths_this_step']}")
        
        if args.render and (step + 1) % args.render_freq == 0:
            print(env.render())
        
        if done:
            print("\nSimulation ended - episode complete!")
            break
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total Steps: {env.time_step}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Total Discharged: {env.total_discharged}")
    print(f"Total Deaths: {env.total_deaths}")
    print(env.render())


def dashboard_command(args):
    """Launch Streamlit dashboard"""
    import subprocess
    print("Launching Streamlit dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", 
                   "--server.port", str(args.port)])


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Disaster Management Simulation using Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --episodes 50 --scenario Earthquake
  python main.py evaluate --model models/best_agent.pkl
  python main.py compare --model models/best_agent.pkl --visualize
  python main.py simulate --model models/best_agent.pkl --steps 100
  python main.py dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an RL agent')
    train_parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    train_parser.add_argument('--scenario', type=str, default='Earthquake', 
                             choices=[s['name'] for s in DISASTER_SCENARIOS],
                             help='Disaster scenario to train on')
    train_parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate')
    train_parser.add_argument('--gamma', type=float, default=0.7, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=0.3, help='Initial exploration rate')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--output', type=str, default='models', help='Output directory')
    train_parser.add_argument('--render', action='store_true', help='Render during training')
    train_parser.add_argument('--render-freq', type=int, default=10, help='Render frequency')
    train_parser.add_argument('--visualize', action='store_true', help='Show visualizations after training')
    train_parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to saved agent')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    eval_parser.add_argument('--scenario', type=str, default='Earthquake', help='Scenario to evaluate on')
    eval_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare RL agent with manual policies')
    compare_parser.add_argument('--model', type=str, required=True, help='Path to saved agent')
    compare_parser.add_argument('--episodes', type=int, default=10, help='Episodes per agent')
    compare_parser.add_argument('--visualize', action='store_true', help='Show comparison charts')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run interactive simulation')
    sim_parser.add_argument('--model', type=str, help='Path to saved agent')
    sim_parser.add_argument('--manual', type=str, choices=['balanced', 'hospital_priority', 'even_distribution'],
                           help='Use manual policy instead')
    sim_parser.add_argument('--scenario', type=str, default='Earthquake', help='Disaster scenario')
    sim_parser.add_argument('--steps', type=int, default=52, help='Number of simulation steps')
    sim_parser.add_argument('--seed', type=int, help='Random seed')
    sim_parser.add_argument('--render', action='store_true', help='Render environment')
    sim_parser.add_argument('--render-freq', type=int, default=5, help='Render frequency')
    sim_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    dash_parser.add_argument('--port', type=int, default=8501, help='Dashboard port')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'compare':
        compare_command(args)
    elif args.command == 'simulate':
        simulate_command(args)
    elif args.command == 'dashboard':
        dashboard_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
