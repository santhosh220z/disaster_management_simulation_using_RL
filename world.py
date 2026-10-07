"""
World generation for the disaster management simulation.

A WorldConfig describes the world: how many cities, how much infrastructure
each city has, its population, and the disaster's epicenter and magnitude.
generate_world() deterministically builds the cities from the config seed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from infrastructure import Hospital, PowerStation, WaterStation, PublicVenue
from config import GENERATION_CONFIG, DISASTER_SCENARIOS, OPS_CONFIG


@dataclass
class WorldConfig:
    """Configuration for procedural world generation"""
    n_cities: int = 1
    hospitals_per_city: int = 3
    power_stations_per_city: int = 2
    water_stations_per_city: int = 2
    venues_per_city: int = 2
    city_population: int = 10_000
    epicenter_city: int = 0
    magnitude: float = 5.0  # 1-10
    scenario_type: str = "Earthquake"
    repair_crews: int = 2
    delivery_delay_steps: int = 2
    inter_city_transfer_delay: int = 6
    seed: int = 42


def clamp_magnitude(magnitude: float) -> float:
    return min(10.0, max(1.0, float(magnitude)))


def magnitude_to_damage_multiplier(magnitude: float) -> float:
    """Map 1-10 magnitude to a base damage multiplier (0.6 - 2.0)"""
    m = clamp_magnitude(magnitude)
    return 0.6 + (m - 1) * (1.4 / 9)


def magnitude_to_duration(magnitude: float) -> int:
    """Map 1-10 magnitude to disaster duration in hours (12 - 30)"""
    m = clamp_magnitude(magnitude)
    return int(12 + (m - 1) * 2)


def magnitude_to_aftershock_prob(magnitude: float) -> float:
    """Map 1-10 magnitude to aftershock probability (0.05 - 0.23)"""
    m = clamp_magnitude(magnitude)
    return 0.05 + (m - 1) * 0.02


def get_scenario_template(name: str) -> Dict:
    """Get scenario mechanics template by name"""
    for scenario in DISASTER_SCENARIOS:
        if scenario["name"] == name:
            return scenario
    return DISASTER_SCENARIOS[0]


@dataclass
class City:
    """A city with its own infrastructure and population"""
    id: int
    name: str
    population: int
    at_home_population: int
    hospitals: List[Hospital] = field(default_factory=list)
    power_stations: List[PowerStation] = field(default_factory=list)
    water_stations: List[WaterStation] = field(default_factory=list)
    public_venues: List[PublicVenue] = field(default_factory=list)
    distance_from_epicenter: int = 0

    def all_infrastructure(self) -> List:
        return (
            self.hospitals + self.power_stations
            + self.water_stations + self.public_venues
        )

    def avg_damage(self) -> float:
        infra = self.all_infrastructure()
        if not infra:
            return 0.0
        return float(np.mean([i.damage_level for i in infra]))

    def venue_free_capacity(self) -> int:
        return sum(
            v.population_capacity - v.current_population for v in self.public_venues
        )

    def venue_occupancy_ratio(self) -> float:
        capacity = sum(v.population_capacity for v in self.public_venues)
        if capacity <= 0:
            return 0.0
        return sum(v.current_population for v in self.public_venues) / capacity

    def hospital_load_ratio(self) -> float:
        beds = sum(h.bed_capacity for h in self.hospitals)
        if beds <= 0:
            return 0.0
        return sum(h.current_patients for h in self.hospitals) / beds


def generate_world(config: WorldConfig) -> List[City]:
    """
    Deterministically generate cities from the world config.

    Infrastructure capacities scale with each city's population.
    Uses a dedicated RNG seeded from the config so the same config
    always produces the same world layout.
    """
    rng = np.random.default_rng(config.seed)
    gen = GENERATION_CONFIG
    epicenter = min(config.epicenter_city, config.n_cities - 1)

    cities: List[City] = []
    for c in range(config.n_cities):
        population = int(config.city_population * rng.uniform(0.8, 1.2))
        scale = population / 10_000

        hospitals = []
        for i in range(config.hospitals_per_city):
            beds = int(rng.uniform(*gen["hospital_beds_range"]) * scale)
            beds = max(20, beds)
            hospitals.append(Hospital(
                id=i,
                name=f"C{c+1}_Hospital_{i+1}",
                bed_capacity=beds,
                current_patients=int(beds * gen["hospital_occupancy_ratio"]),
                water_requirement=float(rng.uniform(*gen["hospital_water_req_range"]) * scale),
                power_requirement=float(rng.uniform(*gen["hospital_power_req_range"]) * scale),
                discharge_rate_optimal=gen["hospital_discharge_rate"],
                medical_stock=OPS_CONFIG["initial_medical_stock"],
            ))

        power_stations = []
        for i in range(config.power_stations_per_city):
            power_stations.append(PowerStation(
                id=i,
                name=f"C{c+1}_PowerStation_{i+1}",
                total_capacity=float(rng.uniform(*gen["power_capacity_range"]) * scale),
                damage_level=float(rng.uniform(*gen["power_initial_damage_range"])),
                repair_rate=gen["power_repair_rate"],
            ))

        water_stations = []
        for i in range(config.water_stations_per_city):
            water_stations.append(WaterStation(
                id=i,
                name=f"C{c+1}_WaterStation_{i+1}",
                total_capacity=float(rng.uniform(*gen["water_capacity_range"]) * scale),
                damage_level=float(rng.uniform(*gen["water_initial_damage_range"])),
                repair_rate=gen["water_repair_rate"],
            ))

        venues = []
        for i in range(config.venues_per_city):
            capacity = int(rng.uniform(*gen["venue_capacity_range"]) * scale)
            capacity = max(50, capacity)
            venues.append(PublicVenue(
                id=i,
                name=f"C{c+1}_Venue_{i+1}",
                population_capacity=capacity,
                current_population=0,
                water_requirement=float(rng.uniform(*gen["venue_water_req_range"]) * scale),
                power_requirement=float(rng.uniform(*gen["venue_power_req_range"]) * scale),
            ))

        # Initially shelter a fraction of the population, spread across venues
        sheltered = min(
            int(population * gen["venue_initial_shelter_ratio"]),
            sum(v.population_capacity for v in venues),
        )
        remaining = sheltered
        for v in venues:
            take = min(remaining, v.population_capacity // 2)
            v.current_population = take
            remaining -= take

        cities.append(City(
            id=c,
            name=f"City_{c+1}",
            population=population,
            at_home_population=population - sheltered,
            hospitals=hospitals,
            power_stations=power_stations,
            water_stations=water_stations,
            public_venues=venues,
            distance_from_epicenter=abs(c - epicenter),
        ))

    return cities
