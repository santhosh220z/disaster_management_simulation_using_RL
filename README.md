# 🚨 Real-Time Disaster Management Simulation Using Reinforcement Learning

An AI-powered decision support system that learns optimal disaster response strategies through reinforcement learning in a simulated environment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![RL](https://img.shields.io/badge/RL-Q--Learning-orange.svg)

## 📋 Overview

Emergency responders must make rapid, high-impact decisions during disasters with limited prior experience. Incorrect allocation of critical resources such as water and electricity can result in increased casualties and infrastructure failure.

This project implements an intelligent decision-support system that learns optimal disaster response strategies through **Q-Learning** reinforcement learning in a custom-built simulation environment.

### Key Features

- 🏥 **Multi-Infrastructure Simulation**: Hospitals, power stations, water pumping stations, and public venues
- 🤖 **Q-Learning Agent**: Learns optimal resource allocation policies
- 📊 **Real-Time Dashboard**: Streamlit-based visualization for monitoring and control
- 📈 **Performance Comparison**: Compare RL agent against manual decision-making strategies
- 🌪️ **Multiple Disaster Scenarios**: Earthquake, Flood, Hurricane, and Industrial Accidents

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

### State Space

The environment state is represented as a vector containing:
- **Physical Damage Levels** (0-4): None, Minor, Moderate, Severe, Critical
- **Resource Availability Levels** (0-4): None, Critical, Low, Medium, Full

For each infrastructure entity (hospitals, power stations, water stations, public venues).

### Action Space

Combined action space with 25 discrete actions:
- **5 Electricity Distribution Ratios**: Different allocations to hospitals, public venues, and reserves
- **5 Water Distribution Ratios**: Different allocations to hospitals, public venues, and reserves

### Reward Function

```
Reward = (Patients Discharged × +10) + (Deaths × -50) + 
         (Infrastructure Failures × -20) + (Efficiency Bonus × +5)
```

### Learning Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| α (Learning Rate) | 0.5 | How quickly the agent updates Q-values |
| γ (Discount Factor) | 0.7 | Importance of future rewards |
| ε (Exploration Rate) | 0.3 | Initial probability of random action |
| ε Decay | 0.995 | Exploration decay rate per episode |
| Episodes | 50 | Number of training episodes |

## 📁 Project Structure

```
disaster/
├── app.py              # Streamlit dashboard application
├── main.py             # CLI entry point
├── config.py           # Configuration parameters
├── environment.py      # Disaster simulation environment
├── infrastructure.py   # Infrastructure entity models
├── agent.py            # RL agents (Q-Learning, Manual policies)
├── trainer.py          # Training and evaluation pipelines
├── visualization.py    # Matplotlib visualizations
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── models/             # Saved models directory
    ├── best_agent.pkl
    ├── final_agent.pkl
    └── training_history.json
```

## 🎮 Disaster Scenarios

### 1. Earthquake
- High initial damage multiplier (1.5x)
- Aftershock probability (10%)
- Sudden infrastructure damage

### 2. Flood
- Moderate damage multiplier (1.2x)
- Ongoing damage rate
- Water contamination effects

### 3. Hurricane
- Highest damage multiplier (1.8x)
- Power outage probability (30%)
- Duration-limited event (12 hours)

### 4. Industrial Accident
- Standard damage multiplier (1.0x)
- Localized effects
- Higher casualty rate (5%)

## 📊 Evaluation Metrics

1. **Total Patients Discharged**: Primary success metric
2. **Total Deaths**: Minimize casualties
3. **Learning Convergence**: Speed of policy improvement
4. **Policy Stability**: Consistency of learned decisions

## 🔬 Results

The RL agent typically outperforms manual resource allocation strategies:

| Agent | Avg Reward | Avg Discharged | Avg Deaths |
|-------|------------|----------------|------------|
| Q-Learning (RL) | ~150-200 | ~40-50 | ~5-10 |
| Manual (Balanced) | ~100-150 | ~30-40 | ~10-15 |
| Manual (Hospital Priority) | ~120-160 | ~35-45 | ~8-12 |

*Results vary based on disaster scenario and training duration*

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
