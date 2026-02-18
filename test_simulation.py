"""
Unit Tests for Disaster Management Simulation
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Import modules
from infrastructure import (
    Hospital, PowerStation, WaterStation, PublicVenue,
    DisasterEvent, DamageLevel, ResourceLevel
)
from environment import DisasterEnvironment
from agent import QLearningAgent, ManualPolicy, AdaptiveManualPolicy
from trainer import Trainer, Evaluator


class TestInfrastructure:
    """Tests for infrastructure models"""
    
    def test_hospital_creation(self):
        """Test hospital initialization"""
        hospital = Hospital(
            id=1,
            name="Test Hospital",
            bed_capacity=100,
            current_patients=50
        )
        assert hospital.bed_capacity == 100
        assert hospital.current_patients == 50
        assert hospital.is_operational == True
    
    def test_hospital_damage(self):
        """Test hospital damage mechanics"""
        hospital = Hospital(id=1, name="Test")
        hospital.apply_damage(0.5)
        assert hospital.damage_level == 0.5
        assert hospital.get_discrete_damage_level() == DamageLevel.MODERATE
        
        hospital.apply_damage(0.5)  # Total = 1.0
        assert hospital.is_operational == False
    
    def test_hospital_repair(self):
        """Test hospital repair"""
        hospital = Hospital(id=1, name="Test", repair_rate=0.1)
        hospital.apply_damage(0.3)
        hospital.repair()
        assert hospital.damage_level == 0.2
    
    def test_hospital_resource_satisfaction(self):
        """Test resource satisfaction calculation"""
        hospital = Hospital(
            id=1, name="Test",
            water_requirement=100,
            power_requirement=500
        )
        hospital.allocate_resources(water=80, power=400)
        satisfaction = hospital.get_resource_satisfaction()
        assert 0.75 <= satisfaction <= 0.85
    
    def test_power_station_generation(self):
        """Test power generation"""
        station = PowerStation(
            id=1, name="Test",
            total_capacity=1000,
            fuel_level=1.0
        )
        power = station.generate_power()
        assert power > 0
        assert power <= station.total_capacity
        assert station.fuel_level < 1.0  # Consumed some fuel
    
    def test_water_station_pumping(self):
        """Test water pumping"""
        station = WaterStation(
            id=1, name="Test",
            total_capacity=500,
            power_required=100
        )
        station.allocate_power(100)
        water = station.pump_water()
        assert water > 0
        assert water <= station.total_capacity


class TestEnvironment:
    """Tests for disaster environment"""
    
    def test_environment_initialization(self):
        """Test environment creation"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        assert len(env.hospitals) > 0
        assert len(env.power_stations) > 0
        assert len(env.water_stations) > 0
        assert env.n_actions == 25  # 5 * 5
    
    def test_get_state(self):
        """Test state representation"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        state = env.get_state()
        assert isinstance(state, np.ndarray)
        assert len(state) > 0
        assert all(0 <= s <= 4 for s in state)  # Discrete levels 0-4
    
    def test_step(self):
        """Test environment step"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        initial_state = env.get_state()
        
        action = 0
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert 'time_step' in info
        assert 'total_discharged' in info
    
    def test_reset(self):
        """Test environment reset"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        
        # Run some steps
        for _ in range(10):
            env.step(0)
        
        # Reset
        state = env.reset()
        assert env.time_step == 0
        assert env.total_discharged == 0
        assert env.total_deaths == 0
    
    def test_episode_completion(self):
        """Test episode runs to completion"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, done, _ = env.step(np.random.randint(env.n_actions))
            steps += 1
        
        assert done or steps == 100


class TestAgent:
    """Tests for RL agents"""
    
    def test_qlearning_creation(self):
        """Test Q-Learning agent initialization"""
        agent = QLearningAgent(n_actions=25)
        assert agent.n_actions == 25
        assert agent.alpha == 0.5
        assert agent.gamma == 0.7
    
    def test_qlearning_action_selection(self):
        """Test action selection"""
        agent = QLearningAgent(n_actions=25, epsilon=1.0)  # Full exploration
        state = (1, 2, 1, 2, 1, 2, 1, 2)
        
        action = agent.get_action(state, training=True)
        assert 0 <= action < 25
    
    def test_qlearning_update(self):
        """Test Q-value update"""
        agent = QLearningAgent(n_actions=25)
        state = (1, 2, 1, 2)
        next_state = (1, 1, 1, 1)
        
        # Initial Q-value should be 0
        assert agent.q_table[state][0] == 0
        
        # Update with reward
        agent.update(state, 0, 10.0, next_state, done=False)
        
        # Q-value should be updated
        assert agent.q_table[state][0] > 0
    
    def test_qlearning_save_load(self):
        """Test agent save and load"""
        agent = QLearningAgent(n_actions=25)
        state = (1, 2, 1, 2)
        agent.update(state, 0, 10.0, state, done=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_agent.pkl")
            agent.save(filepath)
            
            # Load into new agent
            new_agent = QLearningAgent(n_actions=25)
            new_agent.load(filepath)
            
            assert new_agent.q_table[state][0] == agent.q_table[state][0]
    
    def test_manual_policy(self):
        """Test manual policy"""
        policy = ManualPolicy(strategy="balanced")
        state = (1, 2, 1, 2)
        
        action = policy.get_action(state)
        assert 0 <= action < 25


class TestTraining:
    """Tests for training pipeline"""
    
    def test_quick_train(self):
        """Test quick training function"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        agent = QLearningAgent(n_actions=env.n_actions)
        trainer = Trainer(n_episodes=5, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_dir = Path(tmpdir)
            results = trainer.train(agent, env)
        
        assert 'total_episodes' in results
        assert results['total_episodes'] == 5
        assert len(trainer.training_history['episode_rewards']) == 5
    
    def test_evaluation(self):
        """Test agent evaluation"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        agent = QLearningAgent(n_actions=env.n_actions)
        
        evaluator = Evaluator(n_episodes=3)
        results = evaluator.evaluate(agent, env, "Test Agent")
        
        assert 'avg_reward' in results
        assert 'avg_discharged' in results
        assert len(results['rewards']) == 3


class TestIntegration:
    """Integration tests"""
    
    def test_full_training_cycle(self):
        """Test complete training and evaluation cycle"""
        # Create environment
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        
        # Create and train agent
        agent = QLearningAgent(n_actions=env.n_actions)
        
        # Quick training
        for episode in range(5):
            state = env.reset()
            state_tuple = tuple(state.tolist())
            done = False
            
            while not done:
                action = agent.get_action(state_tuple, training=True)
                next_state, reward, done, info = env.step(action)
                next_state_tuple = tuple(next_state.tolist())
                agent.update(state_tuple, action, reward, next_state_tuple, done)
                state_tuple = next_state_tuple
            
            agent.end_episode(env.current_episode_reward, env.time_step)
        
        # Verify learning occurred
        assert len(agent.q_table) > 0
        assert agent.current_episode >= 5
    
    def test_agent_vs_manual_comparison(self):
        """Test RL agent against manual policy"""
        env = DisasterEnvironment(scenario_name="Earthquake", seed=42)
        
        # Train agent briefly
        agent = QLearningAgent(n_actions=env.n_actions)
        for _ in range(3):
            state = env.reset()
            state_tuple = tuple(state.tolist())
            done = False
            while not done:
                action = agent.get_action(state_tuple, training=True)
                next_state, reward, done, _ = env.step(action)
                agent.update(state_tuple, action, reward, 
                           tuple(next_state.tolist()), done)
                state_tuple = tuple(next_state.tolist())
        
        # Evaluate both agents
        manual_policy = ManualPolicy("balanced")
        evaluator = Evaluator(n_episodes=2)
        
        rl_results = evaluator.evaluate(agent, env, "RL")
        manual_results = evaluator.evaluate(manual_policy, env, "Manual")
        
        # Both should produce valid results
        assert rl_results['avg_reward'] is not None
        assert manual_results['avg_reward'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
