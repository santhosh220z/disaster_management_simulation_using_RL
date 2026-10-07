"""
Disaster Simulation Environment
OpenAI Gym-style environment for multi-city disaster management.

The world is described by a WorldConfig (see world.py): N cities, each with
its own hospitals, power stations, water stations, public venues and
population. The disaster strikes an epicenter city at a configurable
magnitude; neighbouring cities take splash damage decaying with distance.

State Space (aggregate per city, 6N + 2 dimensions, all 0-4):
- avg infrastructure damage, hospital resource satisfaction, hospital load,
  population-at-risk ratio, pending deliveries, venue occupancy
- global: disaster active flag, episode progress

Action Space (150 discrete actions):
- 5 electricity distribution ratios x 5 water distribution ratios
  (applied within each city) x 6 operations:
  repair_power, repair_water, repair_hospitals, evacuate, send_aid, none

Reward:
- Positive for patients discharged
- Negative for deaths (hospitals, venues, at-home population),
  infrastructure failures, and wasted (unallocated) resources
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random

from infrastructure import (
    Hospital, PowerStation, WaterStation, PublicVenue,
    DisasterEvent, DamageLevel, ResourceLevel
)
from world import (
    WorldConfig, City, generate_world, get_scenario_template,
    magnitude_to_damage_multiplier, magnitude_to_duration,
    magnitude_to_aftershock_prob,
)
from config import (
    SIMULATION_CONFIG, STATE_CONFIG,
    ACTION_CONFIG, REWARD_CONFIG, OPS_CONFIG,
)


def _bucket(ratio: float) -> int:
    """Bucket a 0-1 ratio into discrete levels 0-4"""
    if ratio < 0.1:
        return 0
    if ratio < 0.3:
        return 1
    if ratio < 0.5:
        return 2
    if ratio < 0.8:
        return 3
    return 4


class DisasterEnvironment:
    """Multi-city disaster management simulation environment"""

    def __init__(
        self,
        scenario_name: str = "Earthquake",
        seed: Optional[int] = None,
        world_config: Optional[WorldConfig] = None,
    ):
        """Initialize the disaster environment

        Args:
            scenario_name: Scenario mechanics template (legacy interface;
                used when world_config is not provided)
            seed: Random seed for stochastic dynamics
            world_config: Full world description (cities, population,
                magnitude, epicenter). Defaults to a single-city world.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if world_config is None:
            world_config = WorldConfig(
                scenario_type=scenario_name,
                seed=seed if seed is not None else SIMULATION_CONFIG["random_seed"],
            )
        self.world_config = world_config
        self.scenario_name = world_config.scenario_type
        self.scenario = get_scenario_template(self.scenario_name)

        # Time tracking
        self.time_step = 0
        self.max_time_steps = (SIMULATION_CONFIG["episode_duration_hours"] * 60) // SIMULATION_CONFIG["time_step_minutes"]
        self.hours_elapsed = 0

        # Build the world (deterministic layout from config seed)
        self.cities: List[City] = generate_world(self.world_config)

        # Apply initial disaster damage (stochastic per episode)
        self._apply_initial_damage()

        # Action space dimensions
        self.n_electricity_actions = len(ACTION_CONFIG["electricity_distribution_ratios"])
        self.n_water_actions = len(ACTION_CONFIG["water_distribution_ratios"])
        self.n_ops = len(OPS_CONFIG["operations"])
        self.n_actions = self.n_electricity_actions * self.n_water_actions * self.n_ops

        # State space metadata
        self.n_infrastructure = sum(len(c.all_infrastructure()) for c in self.cities)
        self.n_damage_levels = STATE_CONFIG["physical_damage_levels"]
        self.n_resource_levels = STATE_CONFIG["resource_availability_levels"]

        # Tracking metrics
        self.total_discharged = 0
        self.total_deaths = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.surges_applied = 0
        self.surge_interval_steps = self._compute_surge_interval()

        # Resource delivery pipeline and aid convoys
        self.delivery_queue: List[Dict] = []
        self.aid_convoys: List[Dict] = []
        self._aid_water_available: Dict[int, float] = {c.id: 0.0 for c in self.cities}

        # Pre-stock infrastructure with half a step of resources to cover
        # the initial delivery delay (represents pre-disaster reserves)
        self._prestock_resources()

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
        self.disaster = self._make_disaster_event()

    # ------------------------------------------------------------------
    # Backwards-compatible flattened infrastructure views
    # ------------------------------------------------------------------
    @property
    def hospitals(self) -> List[Hospital]:
        return [h for c in self.cities for h in c.hospitals]

    @property
    def power_stations(self) -> List[PowerStation]:
        return [s for c in self.cities for s in c.power_stations]

    @property
    def water_stations(self) -> List[WaterStation]:
        return [s for c in self.cities for s in c.water_stations]

    @property
    def public_venues(self) -> List[PublicVenue]:
        return [v for c in self.cities for v in c.public_venues]

    def _all_infrastructure(self) -> List:
        return [i for c in self.cities for i in c.all_infrastructure()]

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------
    def _make_disaster_event(self) -> DisasterEvent:
        """Create the disaster event from template + magnitude"""
        duration = min(
            self.scenario.get("duration_hours", 999),
            magnitude_to_duration(self.world_config.magnitude),
        )
        aftershock = max(
            self.scenario.get("aftershock_probability", 0),
            magnitude_to_aftershock_prob(self.world_config.magnitude),
        )
        return DisasterEvent(
            name=self.scenario_name,
            severity=magnitude_to_damage_multiplier(self.world_config.magnitude),
            duration_hours=duration,
            aftershock_probability=aftershock,
            ongoing_damage_rate=self.scenario.get("ongoing_damage_rate", 0),
        )

    def _compute_surge_interval(self) -> int:
        """Steps between tsunami wave surges (0 if scenario has no surges)"""
        wave_surges = self.scenario.get("wave_surges", 0)
        if wave_surges <= 0:
            return 0
        duration_hours = min(
            self.scenario.get("duration_hours", 999),
            magnitude_to_duration(self.world_config.magnitude),
        )
        duration_steps = (duration_hours * 60) // SIMULATION_CONFIG["time_step_minutes"]
        return max(1, duration_steps // (wave_surges + 1))

    def _city_damage_factor(self, city: City) -> float:
        """Splash damage decay: full at epicenter, halves per city of distance"""
        return 1.0 / (1 + city.distance_from_epicenter)

    def _apply_initial_damage(self):
        """Apply initial disaster damage, decaying with distance from epicenter"""
        multiplier = magnitude_to_damage_multiplier(self.world_config.magnitude)

        for city in self.cities:
            factor = self._city_damage_factor(city)
            for hospital in city.hospitals:
                hospital.apply_damage(np.random.uniform(0.1, 0.4) * multiplier * factor)
            for station in city.power_stations:
                station.apply_damage(np.random.uniform(0, 0.2) * multiplier * factor)
            for station in city.water_stations:
                station.apply_damage(np.random.uniform(0, 0.2) * multiplier * factor)
                if self.scenario.get("water_contamination", False) and factor >= 0.5:
                    station.contaminate(np.random.uniform(0.1, 0.3))
            for venue in city.public_venues:
                venue.apply_damage(np.random.uniform(0.1, 0.3) * multiplier * factor)

    def _prestock_resources(self):
        """Give hospitals/venues half their requirements as starting reserves"""
        for hospital in self.hospitals:
            hospital.water_received = hospital.water_requirement * 0.5
            hospital.power_received = hospital.power_requirement * 0.5
        for venue in self.public_venues:
            venue.water_received = venue.water_requirement * 0.5
            venue.power_received = venue.power_requirement * 0.5

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        """
        Aggregate state: 6 buckets per city plus 2 global indicators.
        All values are discrete levels 0-4.
        """
        state = []

        # System demand for normalizing pending deliveries
        total_demand = sum(
            h.water_requirement + h.power_requirement for h in self.hospitals
        ) + sum(
            v.water_requirement + v.power_requirement for v in self.public_venues
        )

        for city in self.cities:
            # 1. Average infrastructure damage
            state.append(_bucket(city.avg_damage()))

            # 2. Average hospital resource satisfaction
            if city.hospitals:
                satisfaction = np.mean([h.get_resource_satisfaction() for h in city.hospitals])
            else:
                satisfaction = 0.0
            state.append(_bucket(satisfaction))

            # 3. Hospital load (patients / beds)
            state.append(_bucket(city.hospital_load_ratio()))

            # 4. Population still at home (at risk)
            at_risk = city.at_home_population / city.population if city.population > 0 else 0.0
            state.append(_bucket(at_risk))

            # 5. Pending deliveries and aid relative to demand
            pending = sum(
                d["hospital_power"] + d["hospital_water"] + d["venue_power"] + d["venue_water"]
                for d in self.delivery_queue if d["city_id"] == city.id
            ) + sum(
                a["water"] for a in self.aid_convoys if a["target"] == city.id
            )
            state.append(_bucket(pending / total_demand if total_demand > 0 else 0.0))

            # 6. Venue occupancy
            state.append(_bucket(city.venue_occupancy_ratio()))

        # Global: disaster active flag and episode progress
        state.append(int(self.disaster.is_active))
        progress = self.time_step / self.max_time_steps if self.max_time_steps > 0 else 0
        state.append(min(4, int(progress * 5)))

        return np.array(state, dtype=np.int32)

    def get_state_tuple(self) -> Tuple:
        """Get state as tuple for Q-table indexing"""
        return tuple(self.get_state().tolist())

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decode combined action into (electricity, water, operation)"""
        op = action % self.n_ops
        rem = action // self.n_ops
        water_action = rem % self.n_water_actions
        electricity_action = rem // self.n_water_actions
        return electricity_action, water_action, op

    def encode_action(self, electricity_action: int, water_action: int, op: int = None) -> int:
        """Encode (electricity, water, operation) into a combined action"""
        if op is None:
            op = self.n_ops - 1  # default: "none"
        return (electricity_action * self.n_water_actions + water_action) * self.n_ops + op

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step with the given action"""
        electricity_action, water_action, op = self.decode_action(action)

        elec_ratios = ACTION_CONFIG["electricity_distribution_ratios"][electricity_action]
        water_ratios = ACTION_CONFIG["water_distribution_ratios"][water_action]

        step_discharged = 0
        step_deaths = 0
        total_power_all = 0.0
        total_water_all = 0.0
        total_reserve = 0.0

        for city in self.cities:
            factor = self._city_damage_factor(city)

            # Hurricane-style grid outages (splash-scaled)
            outage_prob = self.scenario.get("power_outage_probability", 0) * factor
            for station in city.power_stations:
                station.outage = bool(
                    self.disaster.is_active and outage_prob > 0
                    and np.random.random() < outage_prob
                )

            # Generate power locally
            city_power = sum(s.generate_power() for s in city.power_stations)

            # Power water stations by actual requirement
            total_water_req = sum(s.power_required for s in city.water_stations)
            water_station_power = min(city_power, total_water_req)
            for station in city.water_stations:
                share = station.power_required / total_water_req if total_water_req > 0 else 0
                station.allocate_power(water_station_power * share)

            # Pump water (plus any aid water delivered earlier)
            city_water = sum(s.pump_water() for s in city.water_stations)
            city_water += self._aid_water_available.get(city.id, 0.0)
            self._aid_water_available[city.id] = 0.0

            total_power_all += city_power
            total_water_all += city_water

            # Compute allocation amounts
            distributable_power = city_power - water_station_power
            allocation = {
                "due": self.time_step + self.world_config.delivery_delay_steps,
                "city_id": city.id,
                "hospital_power": distributable_power * elec_ratios[0],
                "venue_power": distributable_power * elec_ratios[1],
                "hospital_water": city_water * water_ratios[0],
                "venue_water": city_water * water_ratios[1],
            }
            reserve = distributable_power * elec_ratios[2] + city_water * water_ratios[2]
            total_reserve += reserve
            self.delivery_queue.append(allocation)

        # Deliver allocations that are due
        self._deliver_due_allocations()

        # Deliver due aid convoys (water joins next step's pool; medical heals now)
        step_discharged += self._deliver_due_aid()

        # Execute the chosen operation
        self._execute_operation(op)

        # Simulate hospitals and venues in each city.
        # casualty_rate is a scenario severity knob; it is scaled down to a
        # per-step venue probability (a flat 5%/step would be a massacre).
        casualty_bonus = self.scenario.get("casualty_rate", 0) * 0.1 if self.disaster.is_active else 0
        arrival_bonus = 10.0 if (
            self.scenario.get("evacuation_required", False) and self.disaster.is_active
        ) else 0.0

        for city in self.cities:
            pop_scale = city.population / 10_000

            for hospital in city.hospitals:
                result = hospital.simulate_step()
                # Scale patient inflow with city population
                extra = int(np.random.poisson(2 * pop_scale * (1 + city.avg_damage())))
                hospital.current_patients = min(hospital.bed_capacity, hospital.current_patients + extra)
                step_discharged += result["discharged"]
                step_deaths += result["deceased"]

            for venue in city.public_venues:
                result = venue.simulate_step(
                    casualty_bonus=casualty_bonus * self._city_damage_factor(city),
                    arrival_bonus=arrival_bonus,
                )
                step_deaths += result["casualties"]

        # Passive repair, refuel, replenish, treat
        for city in self.cities:
            for hospital in city.hospitals:
                hospital.repair()
            for station in city.power_stations:
                station.repair()
                station.refuel(0.02)
            for station in city.water_stations:
                station.repair()
                station.replenish_reservoir(0.03)
                station.treat_water(0.02)
            for venue in city.public_venues:
                venue.repair()

        # Ongoing disaster effects
        if self.disaster.is_active:
            self.disaster.tick(SIMULATION_CONFIG["time_step_minutes"] / 60)

            ongoing_damage = self.disaster.get_ongoing_damage()
            if ongoing_damage > 0:
                for city in self.cities:
                    factor = self._city_damage_factor(city)
                    for infra in city.all_infrastructure():
                        infra.apply_damage(ongoing_damage * factor * np.random.random())

            # Aftershocks (per city, splash-scaled)
            for city in self.cities:
                prob = self.disaster.aftershock_probability * self._city_damage_factor(city)
                if prob > 0 and np.random.random() < prob:
                    self._apply_aftershock(city)

            # Tsunami wave surges
            if (
                self.surge_interval_steps > 0
                and self.surges_applied < self.scenario.get("wave_surges", 0)
                and self.time_step >= (self.surges_applied + 1) * self.surge_interval_steps
            ):
                self._apply_surge()
                self.surges_applied += 1

        # At-home population casualties and migration
        step_deaths += self._apply_home_casualties()
        self._apply_migration()

        # Reward
        waste_fraction = (
            total_reserve / (total_power_all + total_water_all)
            if (total_power_all + total_water_all) > 0 else 0.0
        )
        reward = self._calculate_reward(step_discharged, step_deaths, waste_fraction)

        # Tracking
        self.total_discharged += step_discharged
        self.total_deaths += step_deaths
        self.current_episode_reward += reward
        self.time_step += 1
        self.hours_elapsed = (self.time_step * SIMULATION_CONFIG["time_step_minutes"]) / 60

        self._update_history(reward, step_discharged, step_deaths, total_power_all, total_water_all, action)

        done = self.time_step >= self.max_time_steps

        info = {
            "time_step": self.time_step,
            "hours_elapsed": self.hours_elapsed,
            "discharged_this_step": step_discharged,
            "deaths_this_step": step_deaths,
            "total_discharged": self.total_discharged,
            "total_deaths": self.total_deaths,
            "total_power": total_power_all,
            "total_water": total_water_all,
            "disaster_active": self.disaster.is_active,
        }

        return self.get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------
    def _deliver_due_allocations(self):
        """Deliver queued resource allocations whose delay has elapsed"""
        remaining = []
        for entry in self.delivery_queue:
            if entry["due"] <= self.time_step:
                city = self.cities[entry["city_id"]]
                if city.hospitals:
                    for h in city.hospitals:
                        h.allocate_resources(
                            entry["hospital_water"] / len(city.hospitals),
                            entry["hospital_power"] / len(city.hospitals),
                        )
                if city.public_venues:
                    for v in city.public_venues:
                        v.allocate_resources(
                            entry["venue_water"] / len(city.public_venues),
                            entry["venue_power"] / len(city.public_venues),
                        )
            else:
                remaining.append(entry)
        self.delivery_queue = remaining

    def _deliver_due_aid(self) -> int:
        """Deliver due aid convoys; returns patients healed by medical kits"""
        healed_total = 0
        remaining = []
        for convoy in self.aid_convoys:
            if convoy["due"] <= self.time_step:
                city = self.cities[convoy["target"]]
                self._aid_water_available[city.id] = (
                    self._aid_water_available.get(city.id, 0.0) + convoy["water"]
                )
                # Medical kits: each kit treats 0.2 patients (abstracted)
                if city.hospitals:
                    treatable = int(convoy["medical"] * 0.2)
                    per_hospital = max(1, treatable // len(city.hospitals))
                    for h in city.hospitals:
                        healed = min(per_hospital, h.current_patients)
                        h.current_patients -= healed
                        h.patients_discharged += healed
                        healed_total += healed
            else:
                remaining.append(convoy)
        self.aid_convoys = remaining
        return healed_total

    def _most_damaged_city(self) -> City:
        return max(self.cities, key=lambda c: c.avg_damage())

    def _safest_city(self) -> City:
        return min(self.cities, key=lambda c: c.avg_damage())

    def _execute_operation(self, op: int):
        """Execute the operations dimension of the action"""
        op_name = OPS_CONFIG["operations"][op]
        if op_name == "none":
            return

        target = self._most_damaged_city()
        boost = OPS_CONFIG["repair_boost"] * self.world_config.repair_crews

        if op_name == "repair_power":
            for station in target.power_stations:
                station.repair_amount(boost)
        elif op_name == "repair_water":
            for station in target.water_stations:
                station.repair_amount(boost)
        elif op_name == "repair_hospitals":
            for hospital in target.hospitals:
                hospital.repair_amount(boost)
        elif op_name == "evacuate":
            self._evacuate(target, OPS_CONFIG["evacuation_batch"])
        elif op_name == "send_aid":
            self.aid_convoys.append({
                "due": self.time_step + self.world_config.inter_city_transfer_delay,
                "target": target.id,
                "water": OPS_CONFIG["aid_water_units"],
                "medical": OPS_CONFIG["aid_medical_kits"],
            })

    def _evacuate(self, source: City, batch: int):
        """Move people from a damaged city to the safest city's venues"""
        batch = min(batch, source.at_home_population)
        if batch <= 0:
            return
        dest = self._safest_city()
        if dest.id == source.id:
            # Single city: move people into local venues instead
            dest = source
        accepted = min(batch, dest.venue_free_capacity())
        if accepted <= 0:
            return
        source.at_home_population -= accepted
        for venue in dest.public_venues:
            if accepted <= 0:
                break
            take = min(accepted, venue.population_capacity - venue.current_population)
            venue.current_population += take
            accepted -= take

    def _apply_aftershock(self, city: City):
        """Apply aftershock damage to one city"""
        aftershock_damage = self.scenario.get("aftershock_damage", 0.2)
        for infra in city.all_infrastructure():
            if np.random.random() < 0.5:
                infra.apply_damage(aftershock_damage * np.random.random())

    def _apply_surge(self):
        """Apply a tsunami surge: epicenter full damage, splash decay elsewhere"""
        surge_damage = self.scenario.get("surge_damage", 0.25)
        for city in self.cities:
            factor = self._city_damage_factor(city)
            for infra in city.all_infrastructure():
                infra.apply_damage(surge_damage * factor * np.random.random())
            if self.scenario.get("water_contamination", False) and factor >= 0.5:
                for station in city.water_stations:
                    station.contaminate(np.random.uniform(0.1, 0.3))

    def _apply_home_casualties(self) -> int:
        """Casualties among the at-home (non-sheltered) population.

        The scenario casualty_rate amplifies the base damage-driven rate
        (it was designed for small venue populations, so it is scaled
        into a per-step rate appropriate for a whole city).
        """
        deaths = 0
        for city in self.cities:
            if city.at_home_population <= 0:
                continue
            damage = city.avg_damage()
            if damage <= 0:
                continue
            amplification = 1 + 20 * self.scenario.get("casualty_rate", 0)
            active_scale = 1.0 if self.disaster.is_active else 0.2
            prob = (
                OPS_CONFIG["home_casualty_factor"]
                * damage * amplification * active_scale
            )
            city_deaths = int(np.random.binomial(city.at_home_population, min(1.0, prob)))
            city.at_home_population -= city_deaths
            city.population -= city_deaths
            deaths += city_deaths
        return deaths

    def _apply_migration(self):
        """Voluntary migration from heavily damaged cities to the safest city"""
        threshold = OPS_CONFIG["migration_damage_threshold"]
        rate = OPS_CONFIG["migration_rate"]
        dest = self._safest_city()
        for city in self.cities:
            if city.id == dest.id or city.avg_damage() < threshold:
                continue
            migrants = int(city.at_home_population * rate)
            if migrants <= 0:
                continue
            accepted = min(migrants, dest.venue_free_capacity())
            if accepted <= 0:
                continue
            city.at_home_population -= accepted
            for venue in dest.public_venues:
                if accepted <= 0:
                    break
                take = min(accepted, venue.population_capacity - venue.current_population)
                venue.current_population += take
                accepted -= take

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _calculate_reward(self, discharged: int, deaths: int, waste_fraction: float = 0.0) -> float:
        """Calculate reward for this time step"""
        reward = 0

        reward += discharged * REWARD_CONFIG["patient_discharged"]
        reward += deaths * REWARD_CONFIG["patient_death"]

        for infra in self._all_infrastructure():
            if not infra.is_operational:
                reward += REWARD_CONFIG["infrastructure_failure"]

        # Penalty for generated resources left unallocated (reserve share)
        reward += REWARD_CONFIG["resource_waste"] * waste_fraction * 10

        if self.hospitals:
            avg_satisfaction = np.mean([h.get_resource_satisfaction() for h in self.hospitals])
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
        hospital_resources = [h.get_resource_satisfaction() for h in self.hospitals]
        self.history["hospital_resources"].append(np.mean(hospital_resources) if hospital_resources else 0.0)

    # ------------------------------------------------------------------
    # Reset / render / metrics
    # ------------------------------------------------------------------
    def reset(self, scenario_name: Optional[str] = None) -> np.ndarray:
        """Reset the environment for a new episode"""
        if scenario_name:
            self.scenario_name = scenario_name
            self.scenario = get_scenario_template(scenario_name)
            self.world_config.scenario_type = scenario_name

        self.time_step = 0
        self.hours_elapsed = 0
        self.surges_applied = 0
        self.surge_interval_steps = self._compute_surge_interval()

        if self.current_episode_reward != 0:
            self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0

        self.total_discharged = 0
        self.total_deaths = 0

        self.delivery_queue = []
        self.aid_convoys = []
        self._aid_water_available = {}

        # Regenerate the world (identical layout from config seed) and
        # re-apply stochastic initial damage
        self.cities = generate_world(self.world_config)
        self._apply_initial_damage()
        self._aid_water_available = {c.id: 0.0 for c in self.cities}
        self._prestock_resources()

        self.disaster = self._make_disaster_event()

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

        for city in self.cities:
            epicenter = " [EPICENTER]" if city.distance_from_epicenter == 0 else ""
            output.append(f"\n🏙️ {city.name}{epicenter} | Population: {city.population:,} "
                          f"| At home: {city.at_home_population:,} | Avg damage: {city.avg_damage():.1%}")

            output.append("  📊 HOSPITALS:")
            for h in city.hospitals:
                status = "🟢" if h.is_operational else "🔴"
                output.append(f"    {status} {h.name}: {h.current_patients}/{h.bed_capacity} patients | "
                              f"Damage: {h.damage_level:.1%} | Resources: {h.get_resource_satisfaction():.1%}")
                output.append(f"        Discharged: {h.patients_discharged} | Deceased: {h.patients_deceased}")

            output.append("  ⚡ POWER STATIONS:")
            for ps in city.power_stations:
                status = "🟢" if ps.is_operational else "🔴"
                output.append(f"    {status} {ps.name}: Output {ps.current_output:.0f}/{ps.total_capacity:.0f} kW | "
                              f"Damage: {ps.damage_level:.1%} | Fuel: {ps.fuel_level:.1%}")

            output.append("  💧 WATER STATIONS:")
            for ws in city.water_stations:
                status = "🟢" if ws.is_operational else "🔴"
                output.append(f"    {status} {ws.name}: Output {ws.current_output:.0f}/{ws.total_capacity:.0f} | "
                              f"Damage: {ws.damage_level:.1%} | Reservoir: {ws.reservoir_level:.1%}")

            output.append("  🏛️ PUBLIC VENUES:")
            for pv in city.public_venues:
                status = "🟢" if pv.is_operational else "🔴"
                output.append(f"    {status} {pv.name}: {pv.current_population}/{pv.population_capacity} people | "
                              f"Damage: {pv.damage_level:.1%} | Casualties: {pv.casualties}")

        output.append(f"\n{'='*60}")
        output.append(f"TOTALS - Discharged: {self.total_discharged} | Deaths: {self.total_deaths} | "
                      f"Reward: {self.current_episode_reward:.1f}")
        output.append(f"{'='*60}\n")

        return "\n".join(output)

    def get_metrics(self) -> Dict:
        """Get current simulation metrics"""
        city_metrics = []
        for city in self.cities:
            city_metrics.append({
                "id": city.id,
                "name": city.name,
                "population": city.population,
                "at_home_population": city.at_home_population,
                "avg_damage": city.avg_damage(),
                "hospital_load": city.hospital_load_ratio(),
                "venue_occupancy": city.venue_occupancy_ratio(),
                "distance_from_epicenter": city.distance_from_epicenter,
                "aid_convoys_incoming": sum(1 for a in self.aid_convoys if a["target"] == city.id),
                "hospitals": [
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
                    for h in city.hospitals
                ],
                "power_stations": [
                    {
                        "name": ps.name,
                        "output": ps.current_output,
                        "capacity": ps.total_capacity,
                        "damage": ps.damage_level,
                        "fuel": ps.fuel_level,
                        "operational": ps.is_operational,
                    }
                    for ps in city.power_stations
                ],
                "water_stations": [
                    {
                        "name": ws.name,
                        "output": ws.current_output,
                        "capacity": ws.total_capacity,
                        "damage": ws.damage_level,
                        "reservoir": ws.reservoir_level,
                        "contamination": ws.contamination_level,
                        "operational": ws.is_operational,
                    }
                    for ws in city.water_stations
                ],
                "venues": [
                    {
                        "name": pv.name,
                        "population": pv.current_population,
                        "capacity": pv.population_capacity,
                        "casualties": pv.casualties,
                        "damage": pv.damage_level,
                        "resource_satisfaction": pv.get_resource_satisfaction(),
                        "operational": pv.is_operational,
                    }
                    for pv in city.public_venues
                ],
            })

        return {
            "time_step": self.time_step,
            "hours_elapsed": self.hours_elapsed,
            "total_discharged": self.total_discharged,
            "total_deaths": self.total_deaths,
            "current_reward": self.current_episode_reward,
            "n_cities": len(self.cities),
            "cities": city_metrics,
            # Legacy flattened views (all cities combined)
            "hospital_statuses": [h for c in city_metrics for h in c["hospitals"]],
            "power_statuses": [p for c in city_metrics for p in c["power_stations"]],
            "water_statuses": [w for c in city_metrics for w in c["water_stations"]],
            "venue_statuses": [v for c in city_metrics for v in c["venues"]],
            "disaster": {
                "name": self.disaster.name,
                "active": self.disaster.is_active,
                "hour": self.disaster.current_hour,
                "duration": self.disaster.duration_hours,
                "magnitude": self.world_config.magnitude,
                "epicenter_city": self.world_config.epicenter_city,
            },
        }
