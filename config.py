"""
Configuration settings for Disaster Management Simulation
"""

# Simulation Parameters
SIMULATION_CONFIG = {
    "episode_duration_hours": 26,
    "time_step_minutes": 30,
    "max_episodes": 200,
    "random_seed": 42
}

# Infrastructure Configuration
INFRASTRUCTURE_CONFIG = {
    "hospitals": {
        "count": 3,
        "initial_patients": [50, 75, 100],
        "bed_capacity": [100, 150, 200],
        "water_requirement": [100, 150, 200],  # units per hour
        "power_requirement": [500, 750, 1000],  # kW
        "discharge_rate_optimal": 5,  # patients per hour when resources optimal
    },
    "power_stations": {
        "count": 2,
        "total_capacity": [2000, 1500],  # kW
        "initial_damage": [0.2, 0.3],  # 0-1 damage level
        "repair_rate": 0.05,  # per time step
    },
    "water_stations": {
        "count": 2,
        "total_capacity": [500, 400],  # water units
        "initial_damage": [0.25, 0.15],
        "repair_rate": 0.04,
    },
    "public_venues": {
        "count": 2,
        "population": [200, 150],
        "water_requirement": [50, 40],
        "power_requirement": [200, 150],
    }
}

# Reinforcement Learning Parameters
RL_CONFIG = {
    "learning_rate_alpha": 0.1,
    "discount_factor_gamma": 0.95,
    "exploration_rate_epsilon": 0.3,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
}

# State Space Configuration
STATE_CONFIG = {
    "physical_damage_levels": 5,  # 0: None, 1: Minor, 2: Moderate, 3: Severe, 4: Critical
    "resource_availability_levels": 5,  # 0: None, 1: Critical, 2: Low, 3: Medium, 4: Full
}

# Action Space Configuration
ACTION_CONFIG = {
    "electricity_distribution_ratios": [
        [0.5, 0.3, 0.2],  # Prioritize hospitals
        [0.4, 0.4, 0.2],  # Balanced
        [0.3, 0.5, 0.2],  # Prioritize public venues
        [0.6, 0.2, 0.2],  # Emergency hospital mode
        [0.35, 0.35, 0.3],  # Even distribution
    ],
    "water_distribution_ratios": [
        [0.6, 0.25, 0.15],  # Prioritize hospitals
        [0.4, 0.4, 0.2],  # Balanced
        [0.3, 0.5, 0.2],  # Prioritize public venues
        [0.7, 0.2, 0.1],  # Emergency hospital mode
        [0.35, 0.35, 0.3],  # Even distribution
    ],
}

# Reward Configuration
REWARD_CONFIG = {
    "patient_discharged": 10,
    "patient_death": -50,
    "infrastructure_failure": -20,
    "resource_waste": -1,
    "efficient_allocation": 5,
}

# Visualization Configuration
VIS_CONFIG = {
    "update_interval_ms": 500,
    "plot_history_length": 100,
    "dashboard_port": 8050,
}

# World Generation Ranges (procedural infrastructure generation)
# Capacities scale linearly with city population (reference: 10,000 people)
GENERATION_CONFIG = {
    "hospital_beds_range": (100, 200),
    "hospital_occupancy_ratio": 0.5,
    "hospital_water_req_range": (100, 200),
    "hospital_power_req_range": (500, 1000),
    "hospital_discharge_rate": 5,
    "power_capacity_range": (1500, 2500),
    "power_initial_damage_range": (0.1, 0.3),
    "power_repair_rate": 0.05,
    "water_capacity_range": (400, 600),
    "water_initial_damage_range": (0.1, 0.3),
    "water_repair_rate": 0.04,
    "venue_capacity_range": (150, 250),
    "venue_water_req_range": (40, 60),
    "venue_power_req_range": (150, 250),
    "venue_initial_shelter_ratio": 0.05,  # fraction of population initially in shelters
}

# Operations dimension of the action space
OPS_CONFIG = {
    "operations": [
        "repair_power",
        "repair_water",
        "repair_hospitals",
        "evacuate",
        "send_aid",
        "none",
    ],
    "repair_boost": 0.05,       # extra repair per crew per step
    "evacuation_batch": 100,    # people moved per evacuate operation
    "aid_water_units": 300,     # water delivered per aid convoy
    "aid_medical_kits": 50,     # medical kits per convoy (1 kit treats 0.2 patients)
    "migration_rate": 0.005,    # voluntary per-step migration out of heavily damaged cities
    "migration_damage_threshold": 0.4,
    "home_casualty_factor": 0.0005,  # per-step at-home casualty probability per unit damage
}

# Disaster Scenarios
DISASTER_SCENARIOS = [
    {
        "name": "Earthquake",
        "initial_damage_multiplier": 1.5,
        "aftershock_probability": 0.1,
        "aftershock_damage": 0.2,
    },
    {
        "name": "Flood",
        "initial_damage_multiplier": 1.2,
        "ongoing_damage_rate": 0.02,
        "water_contamination": True,
    },
    {
        "name": "Hurricane",
        "initial_damage_multiplier": 1.8,
        "power_outage_probability": 0.3,
        "duration_hours": 12,
    },
    {
        "name": "Industrial_Accident",
        "initial_damage_multiplier": 1.0,
        "affected_area": "localized",
        "casualty_rate": 0.05,
    },
    {
        "name": "Tsunami",
        "initial_damage_multiplier": 2.0,
        "wave_surges": 3,
        "surge_damage": 0.25,
        "water_contamination": True,
        "coastal_flooding": True,
        "evacuation_required": True,
    },
]
