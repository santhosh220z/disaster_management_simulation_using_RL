"""
Visualization Module for Disaster Management Simulation
Provides real-time and post-training visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class TrainingVisualizer:
    """
    Visualize training progress and results
    """
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'reward': '#2ecc71',
            'discharged': '#3498db',
            'deaths': '#e74c3c',
            'epsilon': '#9b59b6',
            'q_value': '#f39c12',
        }
    
    def plot_training_history(
        self,
        history: Dict,
        show: bool = True,
        save: bool = True
    ):
        """
        Plot comprehensive training history
        
        Args:
            history: Training history dictionary
            show: Display plot
            save: Save plot to file
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History - Disaster Management RL', fontsize=14, fontweight='bold')
        
        episodes = range(1, len(history['episode_rewards']) + 1)
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, history['episode_rewards'], 
                color=self.colors['reward'], alpha=0.6, linewidth=1)
        # Moving average
        window = min(10, len(history['episode_rewards']))
        if window > 1:
            moving_avg = np.convolve(history['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(history['episode_rewards']) + 1), 
                    moving_avg, color=self.colors['reward'], 
                    linewidth=2, label=f'Moving Avg ({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Patients Discharged vs Deaths
        ax2 = axes[0, 1]
        ax2.plot(episodes, history['episode_discharged'], 
                color=self.colors['discharged'], linewidth=2, label='Discharged')
        ax2.plot(episodes, history['episode_deaths'], 
                color=self.colors['deaths'], linewidth=2, label='Deaths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Count')
        ax2.set_title('Healthcare Outcomes per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Metrics
        ax3 = axes[1, 0]
        cumulative_discharged = np.cumsum(history['episode_discharged'])
        cumulative_deaths = np.cumsum(history['episode_deaths'])
        ax3.fill_between(episodes, 0, cumulative_discharged, 
                        alpha=0.3, color=self.colors['discharged'])
        ax3.plot(episodes, cumulative_discharged, 
                color=self.colors['discharged'], linewidth=2, label='Total Discharged')
        ax3.fill_between(episodes, 0, cumulative_deaths, 
                        alpha=0.3, color=self.colors['deaths'])
        ax3.plot(episodes, cumulative_deaths, 
                color=self.colors['deaths'], linewidth=2, label='Total Deaths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Count')
        ax3.set_title('Cumulative Healthcare Outcomes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scenario Distribution
        ax4 = axes[1, 1]
        scenarios = history.get('scenarios_used', [])
        if scenarios:
            scenario_counts = {}
            for s in scenarios:
                scenario_counts[s] = scenario_counts.get(s, 0) + 1
            ax4.bar(scenario_counts.keys(), scenario_counts.values(), 
                   color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(scenario_counts)])
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Count')
        ax4.set_title('Training Scenarios Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_agent_statistics(
        self,
        agent_stats: Dict,
        show: bool = True,
        save: bool = True
    ):
        """
        Plot agent learning statistics
        
        Args:
            agent_stats: Agent statistics dictionary
            show: Display plot
            save: Save plot to file
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Agent Learning Statistics', fontsize=14, fontweight='bold')
        
        # Plot 1: Epsilon Decay
        ax1 = axes[0, 0]
        epsilon_history = agent_stats.get('epsilon_history', [])
        if epsilon_history:
            ax1.plot(epsilon_history, color=self.colors['epsilon'], linewidth=2)
            ax1.axhline(y=agent_stats.get('current_epsilon', 0), 
                       color='red', linestyle='--', label='Current Epsilon')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Epsilon')
        ax1.set_title('Exploration Rate Decay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Value Evolution
        ax2 = axes[0, 1]
        q_history = agent_stats.get('q_value_history', [])
        if q_history:
            ax2.plot(q_history, color=self.colors['q_value'], linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Q-Value')
        ax2.set_title('Q-Value Learning Progress')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Action Distribution
        ax3 = axes[1, 0]
        action_dist = agent_stats.get('action_distribution', {})
        if action_dist:
            actions = list(action_dist.keys())
            counts = list(action_dist.values())
            ax3.bar(range(len(actions)), counts, color='#3498db')
            ax3.set_xlabel('Action')
            ax3.set_ylabel('Count')
        ax3.set_title('Action Selection Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Exploration vs Exploitation
        ax4 = axes[1, 1]
        exploration = agent_stats.get('exploration_steps', 0)
        exploitation = agent_stats.get('exploitation_steps', 0)
        total = exploration + exploitation
        if total > 0:
            labels = ['Exploration', 'Exploitation']
            sizes = [exploration/total * 100, exploitation/total * 100]
            colors = [self.colors['epsilon'], self.colors['q_value']]
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
        ax4.set_title('Exploration vs Exploitation')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'agent_statistics.png', dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(
        self,
        comparison_results: Dict,
        show: bool = True,
        save: bool = True
    ):
        """
        Plot comparison between agents
        
        Args:
            comparison_results: Comparison results from evaluator
            show: Display plot
            save: Save plot to file
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Agent Comparison', fontsize=14, fontweight='bold')
        
        agents = [k for k in comparison_results.keys() if k != 'winner']
        
        # Plot 1: Average Rewards
        ax1 = axes[0]
        avg_rewards = [comparison_results[a]['avg_reward'] for a in agents]
        std_rewards = [comparison_results[a]['std_reward'] for a in agents]
        bars = ax1.bar(agents, avg_rewards, yerr=std_rewards, 
                      capsize=5, color='#3498db', alpha=0.7)
        # Highlight winner
        winner_idx = agents.index(comparison_results['winner'])
        bars[winner_idx].set_color('#2ecc71')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average Discharged
        ax2 = axes[1]
        avg_discharged = [comparison_results[a]['avg_discharged'] for a in agents]
        bars = ax2.bar(agents, avg_discharged, color='#3498db', alpha=0.7)
        bars[winner_idx].set_color('#2ecc71')
        ax2.set_ylabel('Average Patients Discharged')
        ax2.set_title('Healthcare Outcome Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Average Deaths
        ax3 = axes[2]
        avg_deaths = [comparison_results[a]['avg_deaths'] for a in agents]
        bars = ax3.bar(agents, avg_deaths, color='#e74c3c', alpha=0.7)
        ax3.set_ylabel('Average Deaths')
        ax3.set_title('Mortality Comparison (Lower is Better)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'agent_comparison.png', dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


class SimulationVisualizer:
    """
    Real-time visualization of simulation state
    """
    
    def __init__(self):
        self.fig = None
        self.axes = None
        self.initialized = False
    
    def setup(self):
        """Setup the visualization figure"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 10))
        self.fig.suptitle('Real-Time Disaster Simulation', fontsize=14, fontweight='bold')
        self.initialized = True
        plt.ion()
    
    def update(self, metrics: Dict, history: Dict):
        """
        Update visualization with current metrics
        
        Args:
            metrics: Current simulation metrics
            history: Historical data
        """
        if not self.initialized:
            self.setup()
        
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        # Plot 1: Hospital Status
        ax1 = self.axes[0, 0]
        hospitals = metrics['hospital_statuses']
        names = [h['name'] for h in hospitals]
        patients = [h['patients'] for h in hospitals]
        capacities = [h['capacity'] for h in hospitals]
        
        x = np.arange(len(names))
        width = 0.35
        ax1.bar(x - width/2, patients, width, label='Current', color='#3498db')
        ax1.bar(x + width/2, capacities, width, label='Capacity', color='#95a5a6', alpha=0.5)
        ax1.set_ylabel('Patients')
        ax1.set_title('Hospital Occupancy')
        ax1.set_xticks(x)
        ax1.set_xticklabels([n.replace('Hospital_', 'H') for n in names])
        ax1.legend()
        
        # Plot 2: Power Output
        ax2 = self.axes[0, 1]
        power_stations = metrics['power_statuses']
        power_names = [ps['name'].replace('PowerStation_', 'PS') for ps in power_stations]
        outputs = [ps['output'] for ps in power_stations]
        capacities = [ps['capacity'] for ps in power_stations]
        
        x = np.arange(len(power_names))
        ax2.bar(x, outputs, color='#f1c40f', label='Output')
        ax2.bar(x, capacities, color='#95a5a6', alpha=0.3, label='Capacity')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('Power Station Output')
        ax2.set_xticks(x)
        ax2.set_xticklabels(power_names)
        ax2.legend()
        
        # Plot 3: Water Output
        ax3 = self.axes[0, 2]
        water_stations = metrics['water_statuses']
        water_names = [ws['name'].replace('WaterStation_', 'WS') for ws in water_stations]
        water_outputs = [ws['output'] for ws in water_stations]
        water_capacities = [ws['capacity'] for ws in water_stations]
        
        x = np.arange(len(water_names))
        ax3.bar(x, water_outputs, color='#3498db', label='Output')
        ax3.bar(x, water_capacities, color='#95a5a6', alpha=0.3, label='Capacity')
        ax3.set_ylabel('Water Units')
        ax3.set_title('Water Station Output')
        ax3.set_xticks(x)
        ax3.set_xticklabels(water_names)
        ax3.legend()
        
        # Plot 4: Damage Levels
        ax4 = self.axes[1, 0]
        all_infra = (
            [(h['name'], h['damage']) for h in hospitals] +
            [(ps['name'], ps['damage']) for ps in power_stations] +
            [(ws['name'], ws['damage']) for ws in water_stations]
        )
        infra_names = [i[0].split('_')[0][0] + i[0].split('_')[1] for i in all_infra]
        damages = [i[1] * 100 for i in all_infra]
        colors = ['#e74c3c' if d > 50 else '#f1c40f' if d > 25 else '#2ecc71' for d in damages]
        
        ax4.barh(infra_names, damages, color=colors)
        ax4.set_xlabel('Damage (%)')
        ax4.set_title('Infrastructure Damage')
        ax4.set_xlim(0, 100)
        
        # Plot 5: Timeline (Rewards)
        ax5 = self.axes[1, 1]
        if history['rewards']:
            ax5.plot(history['time_steps'], history['rewards'], color='#2ecc71', linewidth=1.5)
            ax5.fill_between(history['time_steps'], history['rewards'], alpha=0.3, color='#2ecc71')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Reward')
        ax5.set_title('Reward Timeline')
        
        # Plot 6: Cumulative Outcomes
        ax6 = self.axes[1, 2]
        if history['discharged']:
            cum_discharged = np.cumsum(history['discharged'])
            cum_deaths = np.cumsum(history['deaths'])
            ax6.plot(history['time_steps'], cum_discharged, 
                    color='#3498db', linewidth=2, label='Discharged')
            ax6.plot(history['time_steps'], cum_deaths, 
                    color='#e74c3c', linewidth=2, label='Deaths')
            ax6.legend()
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Count')
        ax6.set_title('Cumulative Outcomes')
        
        # Add disaster info text
        disaster = metrics.get('disaster', {})
        info_text = (
            f"Scenario: {disaster.get('name', 'Unknown')}\n"
            f"Time: {metrics.get('hours_elapsed', 0):.1f}h\n"
            f"Disaster Active: {'Yes' if disaster.get('active', False) else 'No'}"
        )
        self.fig.text(0.02, 0.02, info_text, fontsize=10, 
                     family='monospace', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def close(self):
        """Close the visualization"""
        if self.fig:
            plt.close(self.fig)
        plt.ioff()


def load_and_visualize(history_path: str = "models/training_history.json"):
    """
    Load training history and create visualizations
    
    Args:
        history_path: Path to training history JSON file
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    visualizer = TrainingVisualizer()
    visualizer.plot_training_history(history)
    
    return visualizer


if __name__ == "__main__":
    # Demo visualization with sample data
    sample_history = {
        'episode_rewards': list(np.random.randn(50).cumsum() + 100),
        'episode_discharged': list(np.random.randint(10, 50, 50)),
        'episode_deaths': list(np.random.randint(0, 10, 50)),
        'episode_lengths': list(np.random.randint(40, 52, 50)),
        'scenarios_used': ['Earthquake', 'Flood', 'Hurricane', 'Industrial_Accident'] * 12 + ['Earthquake', 'Flood'],
    }
    
    visualizer = TrainingVisualizer()
    visualizer.plot_training_history(sample_history)
