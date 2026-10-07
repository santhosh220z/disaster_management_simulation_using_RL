"""
Tests for multi-city world generation and mechanics
"""

import pytest
import numpy as np

from world import (
    WorldConfig, City, generate_world, get_scenario_template,
    magnitude_to_damage_multiplier, magnitude_to_duration,
    magnitude_to_aftershock_prob,
)
from environment import DisasterEnvironment
from agent import QLearningAgent, ManualPolicy, AdaptiveManualPolicy
from config import OPS_CONFIG


class TestWorldGeneration:
    """Tests for procedural world generation"""

    def test_deterministic_generation(self):
        """Same config seed produces identical worlds"""
        config = WorldConfig(n_cities=3, seed=42)
        world_a = generate_world(config)
        world_b = generate_world(config)

        for ca, cb in zip(world_a, world_b):
            assert ca.population == cb.population
            assert [h.bed_capacity for h in ca.hospitals] == [h.bed_capacity for h in cb.hospitals]
            assert [s.total_capacity for s in ca.power_stations] == [s.total_capacity for s in cb.power_stations]

    def test_different_seeds_differ(self):
        """Different seeds produce different worlds"""
        world_a = generate_world(WorldConfig(n_cities=2, seed=1))
        world_b = generate_world(WorldConfig(n_cities=2, seed=999))
        assert world_a[0].population != world_b[0].population

    def test_city_structure(self):
        """Cities get the configured infrastructure counts"""
        config = WorldConfig(n_cities=3, hospitals_per_city=2, venues_per_city=3, seed=42)
        world = generate_world(config)

        assert len(world) == 3
        for city in world:
            assert len(city.hospitals) == 2
            assert len(city.public_venues) == 3
            assert city.population > 0
            assert city.at_home_population >= 0
            assert city.at_home_population <= city.population

    def test_epicenter_distances(self):
        """Distances computed from epicenter"""
        config = WorldConfig(n_cities=4, epicenter_city=1, seed=42)
        world = generate_world(config)
        distances = [c.distance_from_epicenter for c in world]
        assert distances == [1, 0, 1, 2]

    def test_epicenter_clamped(self):
        """Epicenter beyond city count is clamped"""
        config = WorldConfig(n_cities=2, epicenter_city=10, seed=42)
        world = generate_world(config)
        assert world[1].distance_from_epicenter == 0

    def test_population_scales_infrastructure(self):
        """Larger populations get larger infrastructure"""
        small = generate_world(WorldConfig(n_cities=1, city_population=5_000, seed=42))
        large = generate_world(WorldConfig(n_cities=1, city_population=40_000, seed=42))
        small_beds = sum(h.bed_capacity for h in small[0].hospitals)
        large_beds = sum(h.bed_capacity for h in large[0].hospitals)
        assert large_beds > small_beds


class TestMagnitudeMapping:
    """Tests for magnitude parameter mapping"""

    def test_damage_multiplier_bounds(self):
        assert magnitude_to_damage_multiplier(1) == pytest.approx(0.6)
        assert magnitude_to_damage_multiplier(10) == pytest.approx(2.0)
        assert magnitude_to_damage_multiplier(5) > magnitude_to_damage_multiplier(3)

    def test_magnitude_clamped(self):
        assert magnitude_to_damage_multiplier(0) == magnitude_to_damage_multiplier(1)
        assert magnitude_to_damage_multiplier(99) == magnitude_to_damage_multiplier(10)

    def test_duration_bounds(self):
        assert magnitude_to_duration(1) == 12
        assert magnitude_to_duration(10) == 30

    def test_aftershock_increases(self):
        assert magnitude_to_aftershock_prob(8) > magnitude_to_aftershock_prob(2)


class TestMultiCityEnvironment:
    """Tests for multi-city environment mechanics"""

    def test_state_dimensions(self):
        """State is 6 dims per city + 2 global"""
        for n in (1, 2, 3):
            env = DisasterEnvironment(world_config=WorldConfig(n_cities=n, seed=42), seed=42)
            state = env.get_state()
            assert len(state) == 6 * n + 2
            assert all(0 <= s <= 4 for s in state)

    def test_splash_damage_decays(self):
        """Cities farther from epicenter take less initial damage"""
        config = WorldConfig(n_cities=3, epicenter_city=0, magnitude=9, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        damages = [c.avg_damage() for c in env.cities]
        assert damages[0] > damages[2]

    def test_action_space(self):
        """150 combined actions, encode/decode roundtrip"""
        env = DisasterEnvironment(seed=42)
        assert env.n_actions == 150
        for elec in range(5):
            for water in range(5):
                for op in range(6):
                    action = env.encode_action(elec, water, op)
                    assert env.decode_action(action) == (elec, water, op)

    def test_default_op_is_none(self):
        """encode_action without op defaults to 'none'"""
        env = DisasterEnvironment(seed=42)
        _, _, op = env.decode_action(env.encode_action(0, 0))
        assert OPS_CONFIG["operations"][op] == "none"

    def test_delivery_delay(self):
        """With delay > 0, allocations arrive later, not immediately"""
        config = WorldConfig(n_cities=1, delivery_delay_steps=3, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        env.step(0)
        assert len(env.delivery_queue) > 0
        # All queued deliveries are due in the future
        assert all(d["due"] > env.time_step - 1 for d in env.delivery_queue)

    def test_zero_delay_delivers_immediately(self):
        """With delay = 0, allocations arrive the same step"""
        config = WorldConfig(n_cities=1, delivery_delay_steps=0, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        env.step(0)
        assert len(env.delivery_queue) == 0

    def test_repair_crew_operation(self):
        """repair_hospitals op reduces hospital damage in the worst city"""
        config = WorldConfig(n_cities=2, magnitude=9, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        target = env._most_damaged_city()
        before = [h.damage_level for h in target.hospitals]

        op = OPS_CONFIG["operations"].index("repair_hospitals")
        env._execute_operation(op)

        after = [h.damage_level for h in target.hospitals]
        assert all(a <= b for a, b in zip(after, before))
        assert any(a < b for a, b in zip(after, before))

    def test_evacuation_moves_population(self):
        """Evacuate op moves people from damaged city to safer venues"""
        config = WorldConfig(n_cities=2, epicenter_city=0, magnitude=9, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)

        source = env._most_damaged_city()
        dest = env._safest_city()
        assert source.id != dest.id

        source_before = source.at_home_population
        dest_before = sum(v.current_population for v in dest.public_venues)

        op = OPS_CONFIG["operations"].index("evacuate")
        env._execute_operation(op)

        dest_after = sum(v.current_population for v in dest.public_venues)
        assert source.at_home_population < source_before
        assert dest_after > dest_before

    def test_evacuation_respects_capacity(self):
        """Evacuation never overflows venue capacity"""
        config = WorldConfig(n_cities=1, venues_per_city=1, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        for _ in range(30):
            env._evacuate(env.cities[0], OPS_CONFIG["evacuation_batch"])
        for venue in env.public_venues:
            assert venue.current_population <= venue.population_capacity

    def test_aid_convoy_arrives_after_delay(self):
        """send_aid delivers water and medical treatment after transfer delay"""
        config = WorldConfig(n_cities=2, epicenter_city=0, magnitude=8,
                             inter_city_transfer_delay=3, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        target = env._most_damaged_city()

        op = OPS_CONFIG["operations"].index("send_aid")
        env._execute_operation(op)
        assert len(env.aid_convoys) == 1

        healed = 0
        for _ in range(4):  # convoy due at step 0 + 3; checks run at steps 0..3
            healed += env._deliver_due_aid()
            env.time_step += 1
        assert healed > 0
        assert env._aid_water_available[target.id] > 0

    def test_migration_from_damaged_cities(self):
        """Population migrates away from heavily damaged cities"""
        config = WorldConfig(n_cities=2, epicenter_city=0, magnitude=9, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)

        damaged = env._most_damaged_city()
        assert damaged.avg_damage() >= OPS_CONFIG["migration_damage_threshold"]

        before = damaged.at_home_population
        env._apply_migration()
        assert damaged.at_home_population < before

    def test_multicities_episode_completes(self):
        """A 3-city episode runs to completion"""
        config = WorldConfig(n_cities=3, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        agent = QLearningAgent(n_actions=env.n_actions)

        state = tuple(env.reset().tolist())
        done = False
        steps = 0
        while not done and steps < 60:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, tuple(next_state.tolist()), done)
            state = tuple(next_state.tolist())
            steps += 1

        assert done
        assert env.time_step == env.max_time_steps

    def test_home_casualties_scale_with_damage(self):
        """At-home casualties occur in damaged cities"""
        config = WorldConfig(n_cities=1, magnitude=10, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        total = sum(env._apply_home_casualties() for _ in range(10))
        assert total > 0

    def test_adaptive_policy_on_aggregate_state(self):
        """AdaptiveManualPolicy returns valid actions on aggregate states"""
        env = DisasterEnvironment(world_config=WorldConfig(n_cities=2, seed=42), seed=42)
        policy = AdaptiveManualPolicy()
        state = env.get_state_tuple()
        action = policy.get_action(state)
        assert 0 <= action < env.n_actions

    def test_metrics_include_cities(self):
        """get_metrics exposes per-city breakdown"""
        env = DisasterEnvironment(world_config=WorldConfig(n_cities=2, seed=42), seed=42)
        metrics = env.get_metrics()
        assert metrics["n_cities"] == 2
        assert len(metrics["cities"]) == 2
        assert metrics["disaster"]["magnitude"] == 5.0
        # Legacy flattened views still work
        assert len(metrics["hospital_statuses"]) == sum(
            len(c["hospitals"]) for c in metrics["cities"]
        )


class TestRealismMechanics:
    """Tests for realism mechanics: overcrowding, contamination,
    deprivation damage, fuel logistics, and report tracking"""

    def test_overcrowding_increases_deaths(self):
        """Hospitals above 90% occupancy have elevated mortality"""
        config = WorldConfig(n_cities=1, hospitals_per_city=1, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        hospital = env.hospitals[0]

        np.random.seed(42)
        hospital.current_patients = hospital.bed_capacity // 2
        hospital.allocate_resources(0, 0)  # starve so base death prob > 0
        deaths_normal = sum(hospital.simulate_step()["deceased"] for _ in range(50))

        np.random.seed(42)
        hospital.current_patients = hospital.bed_capacity  # full = overcrowded
        hospital.allocate_resources(0, 0)
        deaths_crowded = sum(hospital.simulate_step()["deceased"] for _ in range(50))

        assert deaths_crowded > deaths_normal

    def test_contamination_increases_deaths(self):
        """Contaminated water supply causes infections"""
        config = WorldConfig(n_cities=1, hospitals_per_city=1, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        hospital = env.hospitals[0]
        hospital.allocate_resources(0, 0)

        np.random.seed(42)
        patients0 = hospital.current_patients
        clean = sum(hospital.simulate_step(contamination=0.0)["deceased"] for _ in range(100))

        np.random.seed(42)
        hospital.current_patients = patients0
        dirty = sum(hospital.simulate_step(contamination=0.9)["deceased"] for _ in range(100))

        assert dirty > clean

    def test_deprivation_damage(self):
        """Resource-starved facilities take damage over time"""
        config = WorldConfig(n_cities=1, hospitals_per_city=1, seed=42,
                             delivery_delay_steps=6)
        env = DisasterEnvironment(world_config=config, seed=42)
        hospital = env.hospitals[0]
        hospital.water_received = 0
        hospital.power_received = 0
        hospital.repair_rate = 0  # isolate deprivation from passive repair
        before = hospital.damage_level

        env.step(0)

        assert hospital.damage_level > before

    def test_deprivation_slower_than_repair(self):
        """Deprivation only slows recovery; passive repair still dominates"""
        config = WorldConfig(n_cities=1, hospitals_per_city=1, seed=42,
                             delivery_delay_steps=6)
        env = DisasterEnvironment(world_config=config, seed=42)
        hospital = env.hospitals[0]
        hospital.water_received = 0
        hospital.power_received = 0
        before = hospital.damage_level

        env.step(0)

        # Net damage still decreases because repair (0.05) > deprivation (0.005)
        assert hospital.damage_level < before

    def test_fuel_depletes_without_aid(self):
        """Passive refuel is below consumption, so fuel drains slowly"""
        config = WorldConfig(n_cities=1, seed=42)
        env = DisasterEnvironment(world_config=config, seed=42)
        station = env.power_stations[0]
        before = station.fuel_level

        for _ in range(5):
            env.step(0)

        assert station.fuel_level < before

    def test_aid_convoy_delivers_fuel(self):
        """Aid convoys refuel power stations on arrival"""
        config = WorldConfig(n_cities=2, magnitude=8, seed=42,
                             inter_city_transfer_delay=2)
        env = DisasterEnvironment(world_config=config, seed=42)
        target = env._most_damaged_city()
        for s in target.power_stations:
            s.fuel_level = 0.3

        env.aid_convoys.append({
            "due": env.time_step, "target": target.id,
            "water": 100, "medical": 10, "fuel": OPS_CONFIG["aid_fuel_units"],
        })
        env._deliver_due_aid()

        for s in target.power_stations:
            assert s.fuel_level == pytest.approx(0.3 + OPS_CONFIG["aid_fuel_units"])
        assert env.aid_delivered["convoys"] == 1
        assert env.aid_delivered["fuel"] > 0

    def test_deaths_by_cause_sums_to_total(self):
        """Death accounting is consistent"""
        env = DisasterEnvironment(world_config=WorldConfig(n_cities=2, magnitude=7, seed=42), seed=42)
        for _ in range(20):
            env.step(0)
        assert sum(env.deaths_by_cause.values()) == env.total_deaths
        assert sum(env.deaths_by_city.values()) == env.total_deaths

    def test_episode_report_structure(self):
        """After-action report has all sections"""
        env = DisasterEnvironment(world_config=WorldConfig(n_cities=2, seed=42), seed=42)
        for _ in range(5):
            env.step(env.encode_action(0, 0, OPS_CONFIG["operations"].index("send_aid")))

        report = env.get_episode_report()
        for key in ["scenario", "magnitude", "n_cities", "total_deaths",
                    "deaths_by_cause", "per_city", "total_evacuated",
                    "aid_delivered", "resources", "history"]:
            assert key in report
        assert len(report["per_city"]) == 2
        assert set(report["deaths_by_cause"].keys()) == {"hospital", "venue", "at_home"}
        assert 0.0 <= report["resources"]["utilization"] <= 1.0
        assert report["resources"]["power_generated"] > 0

    def test_metrics_expose_report(self):
        """get_metrics includes the report section"""
        env = DisasterEnvironment(seed=42)
        env.step(0)
        assert "report" in env.get_metrics()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
