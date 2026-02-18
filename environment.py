"""
Disaster Simulation Environment
OpenAI Gym-style environment for disaster management
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random

from infrastructure import (
    Hospital, PowerStation, WaterStation, PublicVenue,
    DisasterEvent, DamageLevel, ResourceLevel
)
from config import (
    SIMULATION_CONFIG, INFRASTRUCTURE_CONFIG, STATE_CONFIG,
    ACTION_CONFIG, REWARD_CONFIG, DISASTER_SCENARIOS
)


class DisasterEnvironment:
    """
    Disaster Management Simulation Environment
    
    State Space:
    - Physical damage levels of all infrastructure (discrete: 0-4)
    - Resource availability levels of all infrastructure (discrete: 0-4)
    
    Action Space:
    - Electricity distribution ratios (5 options)
    - Water distribution ratios (5 options)
    - Combined: 25 discrete actions
    
    Reward:
    - Positive for patients discharged
    - Negative for deaths, infrastructure failures
    """
    
    def __init__(self, scenario_name: str = "Earthquake", seed: Optional[int] = None):
        """Initialize the disaster environment"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.scenario_name = scenario_name
        self.scenario = self._get_scenario(scenario_name)
        
        # Time tracking
        self.time_step = 0
        self.max_time_steps = (SIMULATION_CONFIG["episode_duration_hours"] * 60) // SIMULATION_CONFIG["time_step_minutes"]
        self.hours_elapsed = 0
        
        # Infrastructure
        self.hospitals: List[Hospital] = []
        self.power_stations: List[PowerStation] = []
        self.water_stations: List[WaterStation] = []
        self.public_venues: List[PublicVenue] = []
        
        # Initialize infrastructure
        self._initialize_infrastructure()
        
        # Apply initial disaster damage
        self._apply_initial_damage()
        
        # Action space dimensions
        self.n_electricity_actions = len(ACTION_CONFIG["electricity_distribution_ratios"])
        self.n_water_actions = len(ACTION_CONFIG["water_distribution_ratios"])
        self.n_actions = self.n_electricity_actions * self.n_water_actions
        
        # State space dimensions
        self.n_infrastructure = (
            len(self.hospitals) + 
            len(self.power_stations) + 
            len(self.water_stations) + 
            len(self.public_venues)
        )
        self.n_damage_levels = STATE_CONFIG["physical_damage_levels"]
        self.n_resource_levels = STATE_CONFIG["resource_availability_levels"]
        
        # Tracking metrics
        self.total_discharged = 0
        self.total_deaths = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # History for visualization
        self.history = {
            "time_steps": [],
            "rewards": [],
            "discharged": [],
            "deaths": [],
            "power_output": [],
            "water_output": [],
            "hospital_resources": [],
            "actions": [],
        }
        
        # Disaster event
        self.disaster = DisasterEvent(
            name=scenario_name,
            severity=self.scenario.get("initial_damage_multiplier", 1.0),
            duration_hours=self.scenario.get("duration_hours", 24),
            aftershock_probability=self.scenario.get("aftershock_probability", 0),
            ongoing_damage_rate=self.scenario.get("ongoing_damage_rate", 0)
        )
    
    def _get_scenario(self, name: str) -> Dict:
        """Get scenario configuration by name"""
        for scenario in DISASTER_SCENARIOS:
            if scenario["name"] == name:
                return scenario
        return DISASTER_SCENARIOS[0]  # Default to first scenario
    
    def _initialize_infrastructure(self):
        """Initialize all infrastructure entities"""
        config = INFRASTRUCTURE_CONFIG
        
        # Initialize hospitals
        for i in range(config["hospitals"]["count"]):
            hospital = Hospital(
                id=i,
                name=f"Hospital_{i+1}",
                bed_capacity=config["hospitals"]["bed_capacity"][i],
                current_patients=config["hospitals"]["initial_patients"][i],
                water_requirement=config["hospitals"]["water_requirement"][i],
                power_requirement=config["hospitals"]["power_requirement"][i],
                discharge_rate_optimal=config["hospitals"]["discharge_rate_optimal"]
            )
            self.hospitals.append(hospital)
        
        # Initialize power stations
        for i in range(config["power_stations"]["count"]):
            station = PowerStation(
                id=i,
                name=f"PowerStation_{i+1}",
                total_capacity=config["power_stations"]["total_capacity"][i],
                damage_level=config["power_stations"]["initial_damage"][i],
                repair_rate=config["power_stations"]["repair_rate"]
            )
            self.power_stations.append(station)
        
        # Initialize water stations
        for i in range(config["water_stations"]["count"]):
            station = WaterStation(
                id=i,
                name=f"WaterStation_{i+1}",
                total_capacity=config["water_stations"]["total_capacity"][i],
                damage_level=config["water_stations"]["initial_damage"][i],
                repair_rate=config["water_stations"]["repair_rate"]
            )
            self.water_stations.append(station)
        
        # Initialize public venues
        for i in range(config["public_venues"]["count"]):
            venue = PublicVenue(
                id=i,
                name=f"PublicVenue_{i+1}",
                population_capacity=config["public_venues"]["population"][i],
                current_population=config["public_venues"]["population"][i] // 2,
                water_requirement=config["public_venues"]["water_requirement"][i],
                power_requirement=config["public_venues"]["power_requirement"][i]
            )
            self.public_venues.append(venue)
    
    def _apply_initial_damage(self):
        """Apply initial disaster damage to infrastructure"""
        multiplier = self.scenario.get("initial_damage_multiplier", 1.0)
        
        for hospital in self.hospitals:
            damage = np.random.uniform(0.1, 0.4) * multiplier
            hospital.apply_damage(damage)
        
        for station in self.power_stations:
            additional_damage = np.random.uniform(0, 0.2) * multiplier
            station.apply_damage(additional_damage)
        
        for station in self.water_stations:
            additional_damage = np.random.uniform(0, 0.2) * multiplier
            station.apply_damage(additional_damage)
            if self.scenario.get("water_contamination", False):
                station.contaminate(np.random.uniform(0.1, 0.3))
        
        for venue in self.public_venues:
            damage = np.random.uniform(0.1, 0.3) * multiplier
            venue.apply_damage(damage)
    
    def get_state(self) -> np.ndarray:
        """
        Get current environment state as a vector
        
        Returns:
            State vector containing damage and resource levels for all infrastructure
        """
        state = []
        
        # Hospital states (damage + resource level)
        for hospital in self.hospitals:
            state.append(hospital.get_discrete_damage_level())
            state.append(hospital.get_resource_level())
        
        # Power station states
        for station in self.power_stations:
            state.append(station.get_discrete_damage_level())
            state.append(station.get_resource_level())
        
        # Water station states
        for station in self.water_stations:
            state.append(station.get_discrete_damage_level())
            state.append(station.get_resource_level())
        
        # Public venue states
        for venue in self.public_venues:
            state.append(venue.get_discrete_damage_level())
            state.append(venue.get_resource_level())
        
        return np.array(state, dtype=np.int32)
    
    def get_state_tuple(self) -> Tuple:
        """Get state as tuple for Q-table indexing"""
        return tuple(self.get_state().tolist())
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode combined action into electricity and water actions"""
        electricity_action = action // self.n_water_actions
        water_action = action % self.n_water_actions
        return electricity_action, water_action
    
    def encode_action(self, electricity_action: int, water_action: int) -> int:
        """Encode electricity and water actions into combined action"""
        return electricity_action * self.n_water_actions + water_action
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step with the given action
        
        Args:
            action: Combined action index (0 to n_actions-1)
        
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        electricity_action, water_action = self.decode_action(action)
        
        # Get distribution ratios
        elec_ratios = ACTION_CONFIG["electricity_distribution_ratios"][electricity_action]
        water_ratios = ACTION_CONFIG["water_distribution_ratios"][water_action]
        
        # Generate resources from stations
        total_power = sum(station.generate_power() for station in self.power_stations)
        
        # Allocate power to water stations first (they need it to operate)
        water_station_power = total_power * 0.1  # 10% for water stations
        for station in self.water_stations:
            station.allocate_power(water_station_power / len(self.water_stations))
        
        total_water = sum(station.pump_water() for station in self.water_stations)
        
        # Distribute remaining power
        distributable_power = total_power * 0.9
        hospital_power = distributable_power * elec_ratios[0]
        venue_power = distributable_power * elec_ratios[1]
        
        # Distribute water
        hospital_water = total_water * water_ratios[0]
        venue_water = total_water * water_ratios[1]
        
        # Allocate to hospitals
        step_discharged = 0
        step_deaths = 0
        
        for i, hospital in enumerate(self.hospitals):
            power_share = hospital_power / len(self.hospitals)
            water_share = hospital_water / len(self.hospitals)
            hospital.allocate_resources(water_share, power_share)
            
            result = hospital.simulate_step()
            step_discharged += result["discharged"]
            step_deaths += result["deceased"]
        
        # Allocate to public venues
        for i, venue in enumerate(self.public_venues):
            power_share = venue_power / len(self.public_venues)
            water_share = venue_water / len(self.public_venues)
            venue.allocate_resources(water_share, power_share)
            
            result = venue.simulate_step()
            step_deaths += result["casualties"]
        
        # Repair infrastructure
        for hospital in self.hospitals:
            hospital.repair()
        for station in self.power_stations:
            station.repair()
            station.refuel(0.02)  # Slow refueling
        for station in self.water_stations:
            station.repair()
            station.replenish_reservoir(0.03)
            station.treat_water(0.02)
        for venue in self.public_venues:
            venue.repair()
        
        # Check for ongoing disaster effects
        if self.disaster.is_active:
            self.disaster.tick()
            
            # Apply ongoing damage
            ongoing_damage = self.disaster.get_ongoing_damage()
            if ongoing_damage > 0:
                for infra in self.hospitals + self.power_stations + self.water_stations + self.public_venues:
                    infra.apply_damage(ongoing_damage * np.random.random())
            
            # Check for aftershocks
            if self.disaster.check_aftershock():
                self._apply_aftershock()
        
        # Calculate reward
        reward = self._calculate_reward(step_discharged, step_deaths, total_power, total_water)
        
        # Update tracking
        self.total_discharged += step_discharged
        self.total_deaths += step_deaths
        self.current_episode_reward += reward
        self.time_step += 1
        self.hours_elapsed = (self.time_step * SIMULATION_CONFIG["time_step_minutes"]) / 60
        
        # Update history
        self._update_history(reward, step_discharged, step_deaths, total_power, total_water, action)
        
        # Check if done
        done = self.time_step >= self.max_time_steps
        
        # Info dictionary
        info = {
            "time_step": self.time_step,
            "hours_elapsed": self.hours_elapsed,
            "discharged_this_step": step_discharged,
            "deaths_this_step": step_deaths,
            "total_discharged": self.total_discharged,
            "total_deaths": self.total_deaths,
            "total_power": total_power,
            "total_water": total_water,
            "disaster_active": self.disaster.is_active,
        }
        
        return self.get_state(), reward, done, info
    
    def _apply_aftershock(self):
        """Apply aftershock damage"""
        aftershock_damage = self.scenario.get("aftershock_damage", 0.2)
        
        for infra in self.hospitals + self.power_stations + self.water_stations + self.public_venues:
            if np.random.random() < 0.5:  # 50% chance to affect each infrastructure
                infra.apply_damage(aftershock_damage * np.random.random())
    
    def _calculate_reward(self, discharged: int, deaths: int, power: float, water: float) -> float:
        """Calculate reward for this time step"""
        reward = 0
        
        # Positive reward for discharges
        reward += discharged * REWARD_CONFIG["patient_discharged"]
        
        # Negative reward for deaths
        reward += deaths * REWARD_CONFIG["patient_death"]
        
        # Check for infrastructure failures
        for infra in self.hospitals + self.power_stations + self.water_stations + self.public_venues:
            if not infra.is_operational:
                reward += REWARD_CONFIG["infrastructure_failure"]
        
        # Efficiency bonus
        total_hospital_satisfaction = sum(h.get_resource_satisfaction() for h in self.hospitals)
        avg_satisfaction = total_hospital_satisfaction / len(self.hospitals)
        if avg_satisfaction > 0.7:
            reward += REWARD_CONFIG["efficient_allocation"]
        
        return reward
    
    def _update_history(self, reward, discharged, deaths, power, water, action):
        """Update history for visualization"""
        self.history["time_steps"].append(self.time_step)
        self.history["rewards"].append(reward)
        self.history["discharged"].append(discharged)
        self.history["deaths"].append(deaths)
        self.history["power_output"].append(power)
        self.history["water_output"].append(water)
        self.history["actions"].append(action)
        
        # Hospital resource levels
        hospital_resources = [h.get_resource_satisfaction() for h in self.hospitals]
        self.history["hospital_resources"].append(np.mean(hospital_resources))
    
    def reset(self, scenario_name: Optional[str] = None) -> np.ndarray:
        """Reset the environment for a new episode"""
        if scenario_name:
            self.scenario_name = scenario_name
            self.scenario = self._get_scenario(scenario_name)
        
        # Reset time
        self.time_step = 0
        self.hours_elapsed = 0
        
        # Store episode reward
        if self.current_episode_reward != 0:
            self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        # Reset tracking
        self.total_discharged = 0
        self.total_deaths = 0
        
        # Clear infrastructure and reinitialize
        self.hospitals.clear()
        self.power_stations.clear()
        self.water_stations.clear()
        self.public_venues.clear()
        
        self._initialize_infrastructure()
        self._apply_initial_damage()
        
        # Reset disaster event
        self.disaster = DisasterEvent(
            name=self.scenario_name,
            severity=self.scenario.get("initial_damage_multiplier", 1.0),
            duration_hours=self.scenario.get("duration_hours", 24),
            aftershock_probability=self.scenario.get("aftershock_probability", 0),
            ongoing_damage_rate=self.scenario.get("ongoing_damage_rate", 0)
        )
        
        # Clear history
        self.history = {
            "time_steps": [],
            "rewards": [],
            "discharged": [],
            "deaths": [],
            "power_output": [],
            "water_output": [],
            "hospital_resources": [],
            "actions": [],
        }
        
        return self.get_state()
    
    def render(self) -> str:
        """Render current state as text"""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"DISASTER MANAGEMENT SIMULATION - {self.scenario_name}")
        output.append(f"Time: {self.hours_elapsed:.1f} hours | Step: {self.time_step}/{self.max_time_steps}")
        output.append(f"{'='*60}")
        
        output.append("\n📊 HOSPITALS:")
        for h in self.hospitals:
            status = "🟢" if h.is_operational else "🔴"
            output.append(f"  {status} {h.name}: {h.current_patients}/{h.bed_capacity} patients | "
                         f"Damage: {h.damage_level:.1%} | Resources: {h.get_resource_satisfaction():.1%}")
            output.append(f"      Discharged: {h.patients_discharged} | Deceased: {h.patients_deceased}")
        
        output.append("\n⚡ POWER STATIONS:")
        for ps in self.power_stations:
            status = "🟢" if ps.is_operational else "🔴"
            output.append(f"  {status} {ps.name}: Output {ps.current_output:.0f}/{ps.total_capacity:.0f} kW | "
                         f"Damage: {ps.damage_level:.1%} | Fuel: {ps.fuel_level:.1%}")
        
        output.append("\n💧 WATER STATIONS:")
        for ws in self.water_stations:
            status = "🟢" if ws.is_operational else "🔴"
            output.append(f"  {status} {ws.name}: Output {ws.current_output:.0f}/{ws.total_capacity:.0f} | "
                         f"Damage: {ws.damage_level:.1%} | Reservoir: {ws.reservoir_level:.1%}")
        
        output.append("\n🏛️ PUBLIC VENUES:")
        for pv in self.public_venues:
            status = "🟢" if pv.is_operational else "🔴"
            output.append(f"  {status} {pv.name}: {pv.current_population}/{pv.population_capacity} people | "
                         f"Damage: {pv.damage_level:.1%} | Casualties: {pv.casualties}")
        
        output.append(f"\n{'='*60}")
        output.append(f"TOTALS - Discharged: {self.total_discharged} | Deaths: {self.total_deaths} | "
                     f"Reward: {self.current_episode_reward:.1f}")
        output.append(f"{'='*60}\n")
        
        return "\n".join(output)
    
    def get_metrics(self) -> Dict:
        """Get current simulation metrics"""
        return {
            "time_step": self.time_step,
            "hours_elapsed": self.hours_elapsed,
            "total_discharged": self.total_discharged,
            "total_deaths": self.total_deaths,
            "current_reward": self.current_episode_reward,
            "hospital_statuses": [
                {
                    "name": h.name,
                    "patients": h.current_patients,
                    "capacity": h.bed_capacity,
                    "discharged": h.patients_discharged,
                    "deceased": h.patients_deceased,
                    "damage": h.damage_level,
                    "resource_satisfaction": h.get_resource_satisfaction(),
                    "operational": h.is_operational,
                }
                for h in self.hospitals
            ],
            "power_statuses": [
                {
                    "name": ps.name,
                    "output": ps.current_output,
                    "capacity": ps.total_capacity,
                    "damage": ps.damage_level,
                    "fuel": ps.fuel_level,
                    "operational": ps.is_operational,
                }
                for ps in self.power_stations
            ],
            "water_statuses": [
                {
                    "name": ws.name,
                    "output": ws.current_output,
                    "capacity": ws.total_capacity,
                    "damage": ws.damage_level,
                    "reservoir": ws.reservoir_level,
                    "contamination": ws.contamination_level,
                    "operational": ws.is_operational,
                }
                for ws in self.water_stations
            ],
            "venue_statuses": [
                {
                    "name": pv.name,
                    "population": pv.current_population,
                    "capacity": pv.population_capacity,
                    "casualties": pv.casualties,
                    "damage": pv.damage_level,
                    "resource_satisfaction": pv.get_resource_satisfaction(),
                    "operational": pv.is_operational,
                }
                for pv in self.public_venues
            ],
            "disaster": {
                "name": self.disaster.name,
                "active": self.disaster.is_active,
                "hour": self.disaster.current_hour,
                "duration": self.disaster.duration_hours,
            }
        }
