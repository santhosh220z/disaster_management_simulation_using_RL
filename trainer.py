"""
Training Pipeline for Disaster Management RL Agent
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json
from datetime import datetime

from environment import DisasterEnvironment
from agent import QLearningAgent, ManualPolicy, AdaptiveManualPolicy
from config import SIMULATION_CONFIG, RL_CONFIG, DISASTER_SCENARIOS


class Trainer:
    """
    Training pipeline for RL agents in disaster management simulation
    """
    
    def __init__(
        self,
        n_episodes: int = None,
        scenarios: List[str] = None,
        save_dir: str = "models",
        verbose: bool = True,
    ):
        """
        Initialize trainer
        
        Args:
            n_episodes: Number of training episodes
            scenarios: List of disaster scenarios to train on
            save_dir: Directory to save models
            verbose: Print training progress
        """
        self.n_episodes = n_episodes or SIMULATION_CONFIG["max_episodes"]
        self.scenarios = scenarios or [s["name"] for s in DISASTER_SCENARIOS]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Training results
        self.training_history = {
            "episode_rewards": [],
            "episode_discharged": [],
            "episode_deaths": [],
            "episode_lengths": [],
            "scenarios_used": [],
            "timestamps": [],
        }
        
        # Best model tracking
        self.best_reward = float("-inf")
        self.best_episode = 0
    
    def train(
        self,
        agent: QLearningAgent,
        env: Optional[DisasterEnvironment] = None,
        render_frequency: int = 0,
    ) -> Dict:
        """
        Train agent on disaster scenarios
        
        Args:
            agent: RL agent to train
            env: Environment (creates new if not provided)
            render_frequency: Render every N episodes (0 = never)
        
        Returns:
            Training results dictionary
        """
        if env is None:
            env = DisasterEnvironment(scenario_name=self.scenarios[0])
        
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("STARTING TRAINING")
            print(f"Episodes: {self.n_episodes}")
            print(f"Scenarios: {self.scenarios}")
            print(f"Learning Rate: {agent.alpha}")
            print(f"Discount Factor: {agent.gamma}")
            print(f"Initial Epsilon: {agent.epsilon}")
            print(f"{'='*60}\n")
        
        for episode in range(self.n_episodes):
            # Select scenario (rotate through available scenarios)
            scenario = self.scenarios[episode % len(self.scenarios)]
            
            # Reset environment
            state = env.reset(scenario_name=scenario)
            state_tuple = tuple(state.tolist())
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Select action
                action = agent.get_action(state_tuple, training=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                next_state_tuple = tuple(next_state.tolist())
                
                # Update agent
                agent.update(state_tuple, action, reward, next_state_tuple, done)
                
                # Update state
                state_tuple = next_state_tuple
                episode_reward += reward
                episode_steps += 1
            
            # Record episode results
            agent.end_episode(episode_reward, episode_steps)
            self._record_episode(
                episode_reward,
                env.total_discharged,
                env.total_deaths,
                episode_steps,
                scenario
            )
            
            # Check for best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_episode = episode
                agent.save(str(self.save_dir / "best_agent.pkl"))
            
            # Render if requested
            if render_frequency > 0 and (episode + 1) % render_frequency == 0:
                print(env.render())
            
            # Progress logging
            if self.verbose and (episode + 1) % 5 == 0:
                avg_reward = np.mean(self.training_history["episode_rewards"][-5:])
                avg_discharged = np.mean(self.training_history["episode_discharged"][-5:])
                print(
                    f"Episode {episode + 1}/{self.n_episodes} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Avg(5): {avg_reward:.1f} | "
                    f"Discharged: {env.total_discharged} | "
                    f"Deaths: {env.total_deaths} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )
        
        # Save final model
        agent.save(str(self.save_dir / "final_agent.pkl"))
        
        # Training summary
        training_time = time.time() - start_time
        results = self._get_training_summary(training_time, agent)
        
        if self.verbose:
            self._print_summary(results)
        
        # Save training history
        self._save_history()
        
        return results
    
    def _record_episode(
        self,
        reward: float,
        discharged: int,
        deaths: int,
        steps: int,
        scenario: str
    ):
        """Record episode results"""
        self.training_history["episode_rewards"].append(reward)
        self.training_history["episode_discharged"].append(discharged)
        self.training_history["episode_deaths"].append(deaths)
        self.training_history["episode_lengths"].append(steps)
        self.training_history["scenarios_used"].append(scenario)
        self.training_history["timestamps"].append(datetime.now().isoformat())
    
    def _get_training_summary(self, training_time: float, agent: QLearningAgent) -> Dict:
        """Get training summary statistics"""
        rewards = self.training_history["episode_rewards"]
        discharged = self.training_history["episode_discharged"]
        deaths = self.training_history["episode_deaths"]
        
        return {
            "total_episodes": len(rewards),
            "training_time_seconds": training_time,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "final_reward": rewards[-1] if rewards else 0,
            "avg_reward": np.mean(rewards) if rewards else 0,
            "avg_reward_last_10": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            "total_discharged": sum(discharged),
            "total_deaths": sum(deaths),
            "avg_discharged_per_episode": np.mean(discharged) if discharged else 0,
            "avg_deaths_per_episode": np.mean(deaths) if deaths else 0,
            "final_epsilon": agent.epsilon,
            "q_table_size": len(agent.q_table),
            "agent_stats": agent.get_statistics(),
        }
    
    def _print_summary(self, results: Dict):
        """Print training summary"""
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Episodes: {results['total_episodes']}")
        print(f"Training Time: {results['training_time_seconds']:.1f} seconds")
        print(f"Best Reward: {results['best_reward']:.1f} (Episode {results['best_episode']})")
        print(f"Final Reward: {results['final_reward']:.1f}")
        print(f"Average Reward: {results['avg_reward']:.1f}")
        print(f"Avg Reward (Last 10): {results['avg_reward_last_10']:.1f}")
        print(f"Total Patients Discharged: {results['total_discharged']}")
        print(f"Total Deaths: {results['total_deaths']}")
        print(f"Final Epsilon: {results['final_epsilon']:.4f}")
        print(f"Q-Table Size: {results['q_table_size']} states")
        print(f"{'='*60}\n")
    
    def _save_history(self):
        """Save training history to file"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")


class Evaluator:
    """
    Evaluation pipeline for comparing agents
    """
    
    def __init__(self, n_episodes: int = 10, scenarios: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            n_episodes: Number of evaluation episodes per agent
            scenarios: Scenarios to evaluate on
        """
        self.n_episodes = n_episodes
        self.scenarios = scenarios or [s["name"] for s in DISASTER_SCENARIOS]
    
    def evaluate(
        self,
        agent,
        env: Optional[DisasterEnvironment] = None,
        agent_name: str = "Agent"
    ) -> Dict:
        """
        Evaluate an agent
        
        Args:
            agent: Agent to evaluate
            env: Environment
            agent_name: Name for logging
        
        Returns:
            Evaluation results
        """
        if env is None:
            env = DisasterEnvironment(scenario_name=self.scenarios[0])
        
        results = {
            "agent_name": agent_name,
            "episodes": [],
            "rewards": [],
            "discharged": [],
            "deaths": [],
        }
        
        for episode in range(self.n_episodes):
            scenario = self.scenarios[episode % len(self.scenarios)]
            state = env.reset(scenario_name=scenario)
            state_tuple = tuple(state.tolist())
            
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(state_tuple, training=False)
                next_state, reward, done, info = env.step(action)
                state_tuple = tuple(next_state.tolist())
                episode_reward += reward
            
            results["episodes"].append(episode)
            results["rewards"].append(episode_reward)
            results["discharged"].append(env.total_discharged)
            results["deaths"].append(env.total_deaths)
        
        # Summary statistics
        results["avg_reward"] = np.mean(results["rewards"])
        results["std_reward"] = np.std(results["rewards"])
        results["avg_discharged"] = np.mean(results["discharged"])
        results["avg_deaths"] = np.mean(results["deaths"])
        
        return results
    
    def compare_agents(
        self,
        agents: List[Tuple[str, any]],
        env: Optional[DisasterEnvironment] = None
    ) -> Dict:
        """
        Compare multiple agents
        
        Args:
            agents: List of (name, agent) tuples
            env: Environment
        
        Returns:
            Comparison results
        """
        if env is None:
            env = DisasterEnvironment(scenario_name=self.scenarios[0])
        
        comparison = {}
        
        for name, agent in agents:
            results = self.evaluate(agent, env, name)
            comparison[name] = results
            
            print(f"\n{name}:")
            print(f"  Avg Reward: {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
            print(f"  Avg Discharged: {results['avg_discharged']:.1f}")
            print(f"  Avg Deaths: {results['avg_deaths']:.1f}")
        
        # Determine winner
        best_agent = max(comparison.keys(), key=lambda x: comparison[x]["avg_reward"])
        comparison["winner"] = best_agent
        
        print(f"\n🏆 Best Agent: {best_agent}")
        
        return comparison


def quick_train(n_episodes: int = 50, verbose: bool = True) -> Tuple[QLearningAgent, Dict]:
    """
    Quick training function for convenience
    
    Args:
        n_episodes: Number of episodes
        verbose: Print progress
    
    Returns:
        Trained agent and results
    """
    env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
    agent = QLearningAgent(n_actions=env.n_actions)
    trainer = Trainer(n_episodes=n_episodes, verbose=verbose)
    results = trainer.train(agent, env)
    return agent, results


def compare_with_manual(
    trained_agent: QLearningAgent,
    n_episodes: int = 10
) -> Dict:
    """
    Compare trained agent with manual policies
    
    Args:
        trained_agent: Trained RL agent
        n_episodes: Episodes for evaluation
    
    Returns:
        Comparison results
    """
    env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
    
    # Create manual policies
    manual_balanced = ManualPolicy(strategy="balanced")
    manual_hospital = ManualPolicy(strategy="hospital_priority")
    adaptive_manual = AdaptiveManualPolicy()
    
    # Compare
    evaluator = Evaluator(n_episodes=n_episodes)
    comparison = evaluator.compare_agents([
        ("RL Agent", trained_agent),
        ("Manual (Balanced)", manual_balanced),
        ("Manual (Hospital Priority)", manual_hospital),
        ("Adaptive Manual", adaptive_manual),
    ], env)
    
    return comparison


if __name__ == "__main__":
    # Run quick training
    agent, results = quick_train(n_episodes=50)
    
    # Compare with manual policies
    print("\n" + "="*60)
    print("COMPARING WITH MANUAL POLICIES")
    print("="*60)
    comparison = compare_with_manual(agent, n_episodes=10)
