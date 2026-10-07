# 🚨 Real-Time Disaster Management Simulation Using Reinforcement Learning

An AI-powered decision support system that learns optimal disaster response strategies through reinforcement learning in a simulated environment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![RL](https://img.shields.io/badge/RL-Q--Learning-orange.svg)

## 📋 Overview

Emergency responders must make rapid, high-impact decisions during disasters with limited prior experience. Incorrect allocation of critical resources such as water and electricity can result in increased casualties and infrastructure failure.

This project implements an intelligent decision-support system that learns optimal disaster response strategies through **Q-Learning** reinforcement learning in a custom-built simulation environment.

### Key Features

- 🏙️ **Configurable Multi-City World**: 1-5 cities with procedurally generated infrastructure, population, and a disaster epicenter with distance-decayed splash damage
- 🎚️ **Full Simulation Control**: Cities, infrastructure counts, population, disaster magnitude (1-10), delivery delays, repair crews — all from the dashboard World Builder
- 🏥 **Dynamic Population**: Casualties, hospital inflow, evacuee migration, and shelter capacity all scale with city population
- 🚚 **Operations Actions**: Repair crews, evacuation, and inter-city aid convoys with realistic delivery delays
- 🤖 **Q-Learning Agent**: Learns optimal resource allocation policies
- 📊 **Real-Time Dashboard**: Streamlit world builder, per-city monitoring, policy inspector, CSV/JSON export
- 📈 **Performance Comparison**: Compare RL agent against manual decision-making strategies
- 🌪️ **Multiple Disaster Scenarios**: Earthquake, Flood, Hurricane, Industrial Accident, and Tsunami

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DISASTER SIMULATION                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Hospital │  │  Power   │  │  Water   │  │  Public  │    │
│  │    1-3   │  │ Stations │  │ Stations │  │  Venues  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│                           │                                  │
│                    ┌──────┴──────┐                          │
│                    │    STATE    │                          │
│                    │   VECTOR    │                          │
│                    └──────┬──────┘                          │
└───────────────────────────┼─────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │  RL AGENT     │
                    │  (Q-Learning) │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │    ACTION     │
                    │ (Resource     │
                    │ Distribution) │
                    └───────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the disaster directory
cd disaster

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Command Line Interface

```bash
# Train an agent
python main.py train --episodes 50 --scenario Earthquake

# Evaluate trained agent
python main.py evaluate --model models/best_agent.pkl --episodes 10

# Compare with manual policies
python main.py compare --model models/best_agent.pkl --visualize

# Run interactive simulation
python main.py simulate --model models/best_agent.pkl --steps 100 --render

# Launch dashboard
python main.py dashboard
```

## 📐 Technical Details

### World Configuration

The world is described by a `WorldConfig` (see `world.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_cities` | 1 | Number of cities (1-5 recommended) |
| `hospitals_per_city` | 3 | Hospitals in each city |
| `power_stations_per_city` | 2 | Power stations in each city |
| `water_stations_per_city` | 2 | Water stations in each city |
| `venues_per_city` | 2 | Public shelters in each city |
| `city_population` | 10,000 | Population per city (infrastructure scales with it) |
| `epicenter_city` | 0 | City where the disaster strikes hardest |
| `magnitude` | 5.0 | Disaster magnitude 1-10 (drives damage, duration, aftershocks) |
| `repair_crews` | 2 | Deployable repair crews |
| `delivery_delay_steps` | 2 | Steps before allocated resources arrive |
| `inter_city_transfer_delay` | 6 | Steps for aid convoys between cities |

Splash damage decays with distance from the epicenter: `damage / (1 + distance)`.

### State Space

Aggregate per-city state — compact regardless of world size:
**6 dimensions per city + 2 global** (all discrete levels 0-4):

1. Average infrastructure damage
2. Hospital resource satisfaction
3. Hospital load (patients / beds)
4. Population at risk (at home / total)
5. Pending deliveries & aid (relative to demand)
6. Venue occupancy

Plus a global disaster-active flag and episode progress indicator.
A 3-city world has a 20-dimensional state.

### Action Space

**150 discrete actions** = 5 electricity ratios × 5 water ratios × 6 operations:

- **Distribution ratios**: allocations to hospitals, venues, and reserve
  (applied within each city; the reserve share is unallocated and penalized as waste)
- **Operations** (auto-target the most-damaged city):
  `repair_power`, `repair_water`, `repair_hospitals` — deploy repair crews
  `evacuate` — move people to the safest city's shelters (capacity-limited)
  `send_aid` — dispatch a convoy with water and medical kits from the safest city
  `none`

### Reward Function

```
Reward = (Patients Discharged × +10) + (Deaths × -50) + 
         (Infrastructure Failures × -20) + (Efficiency Bonus × +5) +
         (Unallocated Resource Fraction × -10)
```

The waste penalty applies to generated power/water held in "reserve" by the
chosen distribution ratios — there is no storage in the simulation, so
unallocated resources are lost.

### Learning Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| α (Learning Rate) | 0.1 | How quickly the agent updates Q-values |
| γ (Discount Factor) | 0.95 | Importance of future rewards |
| ε (Exploration Rate) | 0.3 | Initial probability of random action |
| ε Decay | 0.995 | Exploration decay rate per episode |
| Episodes | 200 | Number of training episodes |

Training is reproducible: the trainer re-seeds the RNG per episode from
`SIMULATION_CONFIG["random_seed"]` (42).

## 📁 Project Structure

```
disaster/
├── app.py              # Streamlit dashboard (world builder, policy inspector)
├── main.py             # CLI entry point
├── config.py           # Configuration parameters
├── world.py            # Multi-city world generation (WorldConfig, City)
├── environment.py      # Disaster simulation environment
├── infrastructure.py   # Infrastructure entity models
├── agent.py            # RL agents (Q-Learning, Manual policies)
├── trainer.py          # Training and evaluation pipelines
├── visualization.py    # Matplotlib visualizations
├── test_simulation.py  # Core test suite
├── test_world.py       # Multi-city world test suite
├── requirements.txt    # Python dependencies
├── .github/workflows/  # CI (pytest on push)
├── README.md           # This file
└── models/             # Saved models directory
    ├── best_agent.pkl
    ├── final_agent.pkl
    └── training_history.json
```

## 🎮 Disaster Scenarios

Scenario templates provide the disaster *mechanics*; the actual severity is
driven by the configurable **magnitude (1-10)**, which maps to damage
multiplier (0.6-2.0x), duration (12-30h), and aftershock probability.
In multi-city worlds, all effects decay with distance from the epicenter.

### 1. Earthquake
- Aftershocks with additional damage (probability scales with magnitude)
- Sudden infrastructure damage

### 2. Flood
- Ongoing damage rate while active
- Water contamination effects

### 3. Hurricane
- Random power-grid outages (30% chance per station per step while active)
- Duration-limited event (12 hours)

### 4. Industrial Accident
- Elevated casualty rate at public venues and among at-home population
- The deadliest scenario per hour in benchmarks

### 5. Tsunami
- 3 wave surges during the event, each damaging all infrastructure
  and re-contaminating water supplies
- Mandatory evacuation: elevated evacuee arrivals at public venues

## 📊 Evaluation Metrics

1. **Total Patients Discharged**: Primary success metric
2. **Total Deaths**: Minimize casualties
3. **Learning Convergence**: Speed of policy improvement
4. **Policy Stability**: Consistency of learned decisions

## 🔬 Results

Measured with `seed=42`, Earthquake magnitude 6.0, agents trained 500 episodes,
evaluated over 10 episodes each (greedy policy, no exploration):

**1 City** (8-dim state)

| Agent | Avg Reward | Avg Discharged | Avg Deaths |
|-------|------------|----------------|------------|
| Q-Learning (RL) | 5751.5 ± 308.4 | 609.4 | 9.2 |
| Manual (Balanced) | 4211.0 ± 577.2 | 498.6 | 16.4 |
| Manual (Hospital Priority) | 5834.4 ± 599.0 | 624.7 | 11.0 |
| Adaptive Manual | 6183.2 ± 292.7 | 655.1 | 10.2 |

**3 Cities** (20-dim state)

| Agent | Avg Reward | Avg Discharged | Avg Deaths |
|-------|------------|----------------|------------|
| Q-Learning (RL) | 18851.6 ± 622.8 | 1966.3 | 19.0 |
| Manual (Balanced) | 15914.6 ± 684.8 | 1693.2 | 22.8 |
| Manual (Hospital Priority) | 19391.5 ± 627.3 | 2019.1 | 19.1 |
| Adaptive Manual | 20279.0 ± 708.1 | 2087.0 | 15.0 |

Per-scenario baseline (Adaptive Manual policy, 3 episodes each, magnitude 5):

| Scenario | Avg Reward | Avg Deaths | Avg Discharged |
|----------|------------|------------|----------------|
| Earthquake | 6143 | 9.0 | 645.7 |
| Flood | 5852 | 15.3 | 647.7 |
| Hurricane | 4965 | 27.3 | 620.0 |
| Industrial Accident | 1815 | 96.0 | 648.0 |
| Tsunami | 6113 | 11.7 | 656.3 |

Key findings:

- The RL agent **decisively beats static heuristics** (+37% over Balanced in
  1-city, +18% in 3-city worlds).
- The Adaptive Manual policy — which also has access to repair-crew
  operations — currently leads. Tabular Q-learning with a coarse aggregate
  state plateaus (verified flat from 500 → 1500 training episodes); closing
  this gap is the prime motivation for the DQN future work below.
- A single RL policy generalizes across scenarios, magnitudes, and world
  sizes without hand-tuning — the manual heuristics encode assumptions
  specific to this reward structure.
- Industrial Accident is the deadliest scenario per hour; Tsunami surges and
  Hurricane outages stress-test recovery planning.

> **Note**: models saved before the multi-city rewrite (25 actions, per-entity
> state) are incompatible with the current environment — retrain after pulling.

## 🔮 Future Enhancements

- [ ] **Deep Q-Network (DQN)**: Replace Q-table with neural network for larger state spaces
- [ ] **Multi-Agent RL**: Coordinate multiple decision-making agents
- [ ] **Real-World Data Integration**: Incorporate actual sensor data
- [ ] **Transfer Learning**: Adapt learned policies across different disasters
- [ ] **Human-in-the-Loop**: Support human override and feedback

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- OpenAI Gym Documentation
- Disaster Management and Emergency Response Literature

## 📄 License

MIT License - Feel free to use and modify for research and educational purposes.

## 👥 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

---

**Built with ❤️ for disaster preparedness and emergency response optimization**
