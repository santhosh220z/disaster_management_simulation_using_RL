"""
Infrastructure Models for Disaster Management Simulation
Defines Hospital, Power Station, Water Station, and Public Venue entities
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import IntEnum


class DamageLevel(IntEnum):
    """Physical damage levels for infrastructure"""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class ResourceLevel(IntEnum):
    """Resource availability levels"""
    NONE = 0
    CRITICAL = 1
    LOW = 2
    MEDIUM = 3
    FULL = 4


@dataclass
class Infrastructure:
    """Base class for all infrastructure entities"""
    id: int
    name: str
    damage_level: float = 0.0  # 0-1 continuous damage
    repair_rate: float = 0.05
    is_operational: bool = True
    
    def get_discrete_damage_level(self) -> DamageLevel:
        """Convert continuous damage to discrete level"""
        if self.damage_level < 0.1:
            return DamageLevel.NONE
        elif self.damage_level < 0.3:
            return DamageLevel.MINOR
        elif self.damage_level < 0.5:
            return DamageLevel.MODERATE
        elif self.damage_level < 0.75:
            return DamageLevel.SEVERE
        else:
            return DamageLevel.CRITICAL
    
    def apply_damage(self, damage: float):
        """Apply damage to infrastructure"""
        self.damage_level = min(1.0, self.damage_level + damage)
        if self.damage_level >= 0.9:
            self.is_operational = False
    
    def repair(self):
        """Repair infrastructure over time"""
        if self.damage_level > 0:
            self.damage_level = max(0, self.damage_level - self.repair_rate)
            if self.damage_level < 0.9:
                self.is_operational = True
    
    def get_efficiency(self) -> float:
        """Get operational efficiency based on damage"""
        return max(0, 1.0 - self.damage_level) if self.is_operational else 0


@dataclass
class Hospital(Infrastructure):
    """Hospital infrastructure with patient management"""
    bed_capacity: int = 100
    current_patients: int = 50
    water_requirement: float = 100.0  # units per hour
    power_requirement: float = 500.0  # kW
    water_received: float = 0.0
    power_received: float = 0.0
    discharge_rate_optimal: float = 5.0
    patients_discharged: int = 0
    patients_deceased: int = 0
    
    def get_resource_satisfaction(self) -> float:
        """Calculate how well resources are meeting requirements"""
        water_sat = min(1.0, self.water_received / self.water_requirement) if self.water_requirement > 0 else 1.0
        power_sat = min(1.0, self.power_received / self.power_requirement) if self.power_requirement > 0 else 1.0
        return (water_sat + power_sat) / 2
    
    def get_resource_level(self) -> ResourceLevel:
        """Get discrete resource availability level"""
        satisfaction = self.get_resource_satisfaction()
        if satisfaction < 0.1:
            return ResourceLevel.NONE
        elif satisfaction < 0.3:
            return ResourceLevel.CRITICAL
        elif satisfaction < 0.5:
            return ResourceLevel.LOW
        elif satisfaction < 0.8:
            return ResourceLevel.MEDIUM
        else:
            return ResourceLevel.FULL
    
    def allocate_resources(self, water: float, power: float):
        """Allocate resources to the hospital"""
        self.water_received = water
        self.power_received = power
    
    def simulate_step(self) -> Dict[str, int]:
        """Simulate one time step of hospital operation"""
        efficiency = self.get_efficiency()
        resource_level = self.get_resource_satisfaction()
        
        # Calculate discharge rate based on efficiency and resources
        effective_discharge_rate = self.discharge_rate_optimal * efficiency * resource_level
        
        # Discharge patients (with some randomness)
        discharged = int(np.random.poisson(effective_discharge_rate))
        discharged = min(discharged, self.current_patients)
        
        # Calculate deaths based on poor conditions
        death_probability = max(0, (0.5 - resource_level) * 0.1 * (1 + self.damage_level))
        deaths = int(np.random.binomial(self.current_patients, death_probability))
        deaths = min(deaths, self.current_patients - discharged)
        
        # Update patient count
        self.current_patients -= (discharged + deaths)
        self.patients_discharged += discharged
        self.patients_deceased += deaths
        
        # New patients arriving (disaster scenario)
        new_patients = int(np.random.poisson(2))
        self.current_patients = min(self.bed_capacity, self.current_patients + new_patients)
        
        return {"discharged": discharged, "deceased": deaths, "new_patients": new_patients}


@dataclass
class PowerStation(Infrastructure):
    """Electrical power generation station"""
    total_capacity: float = 2000.0  # kW
    current_output: float = 0.0
    fuel_level: float = 1.0  # 0-1
    fuel_consumption_rate: float = 0.01
    
    def get_available_power(self) -> float:
        """Get available power output"""
        return self.total_capacity * self.get_efficiency() * self.fuel_level
    
    def generate_power(self) -> float:
        """Generate power for this time step"""
        self.current_output = self.get_available_power()
        self.fuel_level = max(0, self.fuel_level - self.fuel_consumption_rate)
        return self.current_output
    
    def refuel(self, amount: float):
        """Refuel the power station"""
        self.fuel_level = min(1.0, self.fuel_level + amount)
    
    def get_resource_level(self) -> ResourceLevel:
        """Get discrete output level"""
        output_ratio = self.current_output / self.total_capacity if self.total_capacity > 0 else 0
        if output_ratio < 0.1:
            return ResourceLevel.NONE
        elif output_ratio < 0.3:
            return ResourceLevel.CRITICAL
        elif output_ratio < 0.5:
            return ResourceLevel.LOW
        elif output_ratio < 0.8:
            return ResourceLevel.MEDIUM
        else:
            return ResourceLevel.FULL


@dataclass
class WaterStation(Infrastructure):
    """Water pumping and distribution station"""
    total_capacity: float = 500.0  # water units
    current_output: float = 0.0
    reservoir_level: float = 1.0  # 0-1
    contamination_level: float = 0.0  # 0-1
    power_required: float = 100.0  # kW to operate
    power_received: float = 0.0
    
    def get_available_water(self) -> float:
        """Get available water output"""
        power_factor = min(1.0, self.power_received / self.power_required) if self.power_required > 0 else 0
        clean_factor = 1.0 - self.contamination_level
        return self.total_capacity * self.get_efficiency() * self.reservoir_level * power_factor * clean_factor
    
    def pump_water(self) -> float:
        """Pump water for this time step"""
        self.current_output = self.get_available_water()
        self.reservoir_level = max(0, self.reservoir_level - 0.02)  # Slow depletion
        return self.current_output
    
    def allocate_power(self, power: float):
        """Allocate power to the water station"""
        self.power_received = power
    
    def replenish_reservoir(self, amount: float):
        """Replenish water reservoir"""
        self.reservoir_level = min(1.0, self.reservoir_level + amount)
    
    def contaminate(self, level: float):
        """Add contamination to water supply"""
        self.contamination_level = min(1.0, self.contamination_level + level)
    
    def treat_water(self, treatment_level: float):
        """Treat contaminated water"""
        self.contamination_level = max(0, self.contamination_level - treatment_level)
    
    def get_resource_level(self) -> ResourceLevel:
        """Get discrete output level"""
        output_ratio = self.current_output / self.total_capacity if self.total_capacity > 0 else 0
        if output_ratio < 0.1:
            return ResourceLevel.NONE
        elif output_ratio < 0.3:
            return ResourceLevel.CRITICAL
        elif output_ratio < 0.5:
            return ResourceLevel.LOW
        elif output_ratio < 0.8:
            return ResourceLevel.MEDIUM
        else:
            return ResourceLevel.FULL


@dataclass 
class PublicVenue(Infrastructure):
    """Public venue (shelter, evacuation center)"""
    population_capacity: int = 200
    current_population: int = 0
    water_requirement: float = 50.0
    power_requirement: float = 200.0
    water_received: float = 0.0
    power_received: float = 0.0
    casualties: int = 0
    
    def get_resource_satisfaction(self) -> float:
        """Calculate resource satisfaction"""
        water_sat = min(1.0, self.water_received / self.water_requirement) if self.water_requirement > 0 else 1.0
        power_sat = min(1.0, self.power_received / self.power_requirement) if self.power_requirement > 0 else 1.0
        return (water_sat + power_sat) / 2
    
    def get_resource_level(self) -> ResourceLevel:
        """Get discrete resource level"""
        satisfaction = self.get_resource_satisfaction()
        if satisfaction < 0.1:
            return ResourceLevel.NONE
        elif satisfaction < 0.3:
            return ResourceLevel.CRITICAL
        elif satisfaction < 0.5:
            return ResourceLevel.LOW
        elif satisfaction < 0.8:
            return ResourceLevel.MEDIUM
        else:
            return ResourceLevel.FULL
    
    def allocate_resources(self, water: float, power: float):
        """Allocate resources"""
        self.water_received = water
        self.power_received = power
    
    def simulate_step(self) -> Dict[str, int]:
        """Simulate one time step"""
        resource_level = self.get_resource_satisfaction()
        
        # Casualties from poor conditions
        casualty_prob = max(0, (0.3 - resource_level) * 0.02 * (1 + self.damage_level))
        new_casualties = int(np.random.binomial(self.current_population, casualty_prob))
        self.current_population -= new_casualties
        self.casualties += new_casualties
        
        # New evacuees arriving
        new_arrivals = int(np.random.poisson(5))
        self.current_population = min(self.population_capacity, self.current_population + new_arrivals)
        
        return {"casualties": new_casualties, "new_arrivals": new_arrivals}


@dataclass
class DisasterEvent:
    """Represents a disaster event"""
    name: str
    severity: float = 1.0  # Multiplier for damage
    duration_hours: int = 24
    current_hour: int = 0
    is_active: bool = True
    aftershock_probability: float = 0.0
    ongoing_damage_rate: float = 0.0
    
    def tick(self) -> bool:
        """Advance disaster timeline, return True if still active"""
        self.current_hour += 1
        if self.current_hour >= self.duration_hours:
            self.is_active = False
        return self.is_active
    
    def check_aftershock(self) -> bool:
        """Check if aftershock occurs"""
        return np.random.random() < self.aftershock_probability
    
    def get_ongoing_damage(self) -> float:
        """Get ongoing damage for this time step"""
        return self.ongoing_damage_rate if self.is_active else 0
