"""
Reinforcement Learning Agent for Disaster Management
Implements Q-Learning with epsilon-greedy exploration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import json
from pathlib import Path

from config import RL_CONFIG, ACTION_CONFIG


class QLearningAgent:
    """
    Q-Learning agent for disaster response decision making
    
    Uses a Q-table to learn optimal resource allocation policies
    through temporal difference learning.
    """
    
    def __init__(
        self,
        n_actions: int,
        learning_rate: float = None,
        discount_factor: float = None,
        epsilon: float = None,
        epsilon_decay: float = None,
        epsilon_min: float = None,
    ):
        """
        Initialize Q-Learning agent
        
        Args:
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
        """
        self.n_actions = n_actions
        self.alpha = learning_rate or RL_CONFIG["learning_rate_alpha"]
        self.gamma = discount_factor or RL_CONFIG["discount_factor_gamma"]
        self.epsilon = epsilon or RL_CONFIG["exploration_rate_epsilon"]
        self.epsilon_decay = epsilon_decay or RL_CONFIG["epsilon_decay"]
        self.epsilon_min = epsilon_min or RL_CONFIG["epsilon_min"]
        
        # Q-table: state -> action values
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )
        
        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_history": [],
            "q_value_history": [],
            "action_distribution": defaultdict(int),
        }
        
        # Current episode tracking
        self.current_episode = 0
        self.total_steps = 0
        self.exploration_steps = 0
        self.exploitation_steps = 0
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state tuple
            training: Whether in training mode (enables exploration)
        
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(self.n_actions)
            self.exploration_steps += 1
        else:
            # Exploitation: best known action
            q_values = self.q_table[state]
            action = np.argmax(q_values)
            self.exploitation_steps += 1
        
        # Track action distribution
        self.training_stats["action_distribution"][action] += 1
        
        return action
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool
    ):
        """
        Update Q-value using temporal difference learning
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        current_q = self.q_table[state][action]
        
        if done:
            # No future rewards if done
            target = reward
        else:
            # Maximum Q-value for next state
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        
        # Temporal difference update
        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        self.total_steps += 1
    
    def decay_epsilon(self):
        """Decay exploration rate after episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats["epsilon_history"].append(self.epsilon)
    
    def end_episode(self, episode_reward: float, episode_length: int):
        """Record episode statistics"""
        self.current_episode += 1
        self.training_stats["episode_rewards"].append(episode_reward)
        self.training_stats["episode_lengths"].append(episode_length)
        
        # Record average Q-value
        if self.q_table:
            avg_q = np.mean([np.mean(q) for q in self.q_table.values()])
            self.training_stats["q_value_history"].append(avg_q)
        
        # Decay epsilon
        self.decay_epsilon()
    
    def get_q_values(self, state: Tuple) -> np.ndarray:
        """Get Q-values for a state"""
        return self.q_table[state].copy()
    
    def get_best_action(self, state: Tuple) -> int:
        """Get best action for a state (greedy)"""
        return np.argmax(self.q_table[state])
    
    def get_policy(self) -> Dict[Tuple, int]:
        """Get learned policy (state -> best action)"""
        return {state: np.argmax(q_values) for state, q_values in self.q_table.items()}
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        stats = dict(self.training_stats)
        stats["action_distribution"] = dict(stats["action_distribution"])
        stats["total_steps"] = self.total_steps
        stats["exploration_steps"] = self.exploration_steps
        stats["exploitation_steps"] = self.exploitation_steps
        stats["current_episode"] = self.current_episode
        stats["current_epsilon"] = self.epsilon
        stats["q_table_size"] = len(self.q_table)
        
        if stats["episode_rewards"]:
            stats["avg_reward_last_10"] = np.mean(stats["episode_rewards"][-10:])
            stats["max_reward"] = max(stats["episode_rewards"])
            stats["min_reward"] = min(stats["episode_rewards"])
        
        return stats
    
    def save(self, filepath: str):
        """Save agent to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to regular dict for saving
        q_table_dict = {str(k): v.tolist() for k, v in self.q_table.items()}
        
        save_data = {
            "q_table": q_table_dict,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "n_actions": self.n_actions,
            "training_stats": {
                **self.training_stats,
                "action_distribution": dict(self.training_stats["action_distribution"])
            },
            "current_episode": self.current_episode,
            "total_steps": self.total_steps,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file"""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)
        
        # Restore Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in save_data["q_table"].items():
            # Convert string key back to tuple
            key = eval(k)
            self.q_table[key] = np.array(v)
        
        self.alpha = save_data["alpha"]
        self.gamma = save_data["gamma"]
        self.epsilon = save_data["epsilon"]
        self.epsilon_decay = save_data["epsilon_decay"]
        self.epsilon_min = save_data["epsilon_min"]
        self.n_actions = save_data["n_actions"]
        self.training_stats = save_data["training_stats"]
        self.training_stats["action_distribution"] = defaultdict(
            int, self.training_stats["action_distribution"]
        )
        self.current_episode = save_data["current_episode"]
        self.total_steps = save_data["total_steps"]
        
        print(f"Agent loaded from {filepath}")
    
    def reset_exploration_stats(self):
        """Reset exploration/exploitation counters"""
        self.exploration_steps = 0
        self.exploitation_steps = 0


class ManualPolicy:
    """
    Manual policy for comparison with RL agent
    Implements simple rule-based resource allocation
    """
    
    def __init__(self, strategy: str = "balanced"):
        """
        Initialize manual policy
        
        Args:
            strategy: 'balanced', 'hospital_priority', 'even_distribution'
        """
        self.strategy = strategy
        self.action_map = {
            "balanced": 1,  # Balanced distribution
            "hospital_priority": 0,  # Prioritize hospitals
            "even_distribution": 4,  # Even distribution
            "emergency": 3,  # Emergency hospital mode
        }
    
    def get_action(self, state: Tuple, training: bool = False) -> int:
        """Get action based on fixed strategy"""
        electricity_action = self.action_map.get(self.strategy, 1)
        water_action = self.action_map.get(self.strategy, 1)
        
        # Combine into single action
        n_water_actions = len(ACTION_CONFIG["water_distribution_ratios"])
        return electricity_action * n_water_actions + water_action
    
    def update(self, *args, **kwargs):
        """No-op for manual policy"""
        pass
    
    def end_episode(self, *args, **kwargs):
        """No-op for manual policy"""
        pass


class AdaptiveManualPolicy:
    """
    Adaptive manual policy that changes based on situation
    More sophisticated rule-based approach
    """
    
    def __init__(self):
        self.n_water_actions = len(ACTION_CONFIG["water_distribution_ratios"])
    
    def get_action(self, state: Tuple, training: bool = False) -> int:
        """
        Get action based on current state
        
        Adapts strategy based on observed damage levels
        """
        # Parse state to understand current situation
        # State format: [hospital1_damage, hospital1_resource, hospital2_damage, ...]
        
        # Check if any hospital is in critical condition
        hospital_critical = False
        for i in range(0, len(state), 2):
            if i < 6:  # First 3 pairs are hospitals (assuming 3 hospitals)
                damage_level = state[i]
                resource_level = state[i + 1]
                if damage_level >= 3 or resource_level <= 1:
                    hospital_critical = True
                    break
        
        if hospital_critical:
            # Emergency mode: prioritize hospitals
            electricity_action = 3  # Emergency hospital mode
            water_action = 3
        else:
            # Balanced distribution
            electricity_action = 1
            water_action = 1
        
        return electricity_action * self.n_water_actions + water_action
    
    def update(self, *args, **kwargs):
        """No-op"""
        pass
    
    def end_episode(self, *args, **kwargs):
        """No-op"""
        pass


class DQNAgent:
    """
    Deep Q-Network Agent (placeholder for future enhancement)
    
    Replaces Q-table with neural network for better scalability
    to larger state spaces.
    """
    
    def __init__(self, state_size: int, n_actions: int):
        self.state_size = state_size
        self.n_actions = n_actions
        
        # Placeholder - would implement neural network here
        raise NotImplementedError(
            "DQN Agent is planned for future enhancement. "
            "Currently using Q-Learning agent."
        )
