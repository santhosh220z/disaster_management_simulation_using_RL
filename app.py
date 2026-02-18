"""
Streamlit Dashboard for Real-Time Disaster Management Simulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import json

from environment import DisasterEnvironment
from agent import QLearningAgent, ManualPolicy, AdaptiveManualPolicy
from trainer import Trainer, Evaluator, quick_train
from config import DISASTER_SCENARIOS, ACTION_CONFIG


# Page configuration
st.set_page_config(
    page_title="Disaster Management RL Simulation",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    div[data-testid="stMetric"] > div {
        color: #000000 !important;
    }
    div[data-testid="stMetric"] label {
        color: #000000 !important;
    }
    .status-operational {
        color: #28a745;
        font-weight: bold;
    }
    .status-damaged {
        color: #ffc107;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_status_color(damage_level: float) -> str:
    """Get status color based on damage level"""
    if damage_level < 0.25:
        return "🟢"
    elif damage_level < 0.5:
        return "🟡"
    elif damage_level < 0.75:
        return "🟠"
    else:
        return "🔴"


def create_gauge_chart(value: float, title: str, max_val: float = 100) -> go.Figure:
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, max_val * 0.25], 'color': "#e74c3c"},
                {'range': [max_val * 0.25, max_val * 0.5], 'color': "#f1c40f"},
                {'range': [max_val * 0.5, max_val * 0.75], 'color': "#2ecc71"},
                {'range': [max_val * 0.75, max_val], 'color': "#27ae60"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    st.title("🚨 Real-Time Disaster Management Simulation")
    st.markdown("### AI-Powered Decision Support System using Reinforcement Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Scenario selection
        scenario = st.selectbox(
            "Disaster Scenario",
            [s["name"] for s in DISASTER_SCENARIOS],
            index=0
        )
        
        # Agent selection
        agent_type = st.selectbox(
            "Agent Type",
            ["Q-Learning (RL)", "Manual (Balanced)", "Manual (Hospital Priority)", "Adaptive Manual"],
            index=0
        )
        
        st.divider()
        
        # Training parameters
        st.subheader("📊 Training Parameters")
        n_episodes = st.slider("Training Episodes", 10, 100, 50)
        learning_rate = st.slider("Learning Rate (α)", 0.1, 1.0, 0.5)
        discount_factor = st.slider("Discount Factor (γ)", 0.1, 1.0, 0.7)
        epsilon = st.slider("Initial Exploration (ε)", 0.1, 1.0, 0.3)
        
        st.divider()
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["🎮 Interactive Simulation", "🏋️ Train Agent", "📈 Evaluate & Compare"]
        )
    
    # Initialize session state
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    # Main content based on mode
    if mode == "🎮 Interactive Simulation":
        run_interactive_simulation(scenario, agent_type)
    
    elif mode == "🏋️ Train Agent":
        run_training(scenario, n_episodes, learning_rate, discount_factor, epsilon)
    
    elif mode == "📈 Evaluate & Compare":
        run_evaluation(scenario)


def run_interactive_simulation(scenario: str, agent_type: str):
    """Run interactive simulation mode"""
    st.header("🎮 Interactive Simulation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Reset Simulation", type="primary"):
            st.session_state.env = DisasterEnvironment(scenario_name=scenario, seed=42)
            if agent_type == "Q-Learning (RL)":
                st.session_state.agent = QLearningAgent(n_actions=st.session_state.env.n_actions)
            elif agent_type == "Manual (Balanced)":
                st.session_state.agent = ManualPolicy("balanced")
            elif agent_type == "Manual (Hospital Priority)":
                st.session_state.agent = ManualPolicy("hospital_priority")
            else:
                st.session_state.agent = AdaptiveManualPolicy()
            st.session_state.history = []
            st.success("Simulation reset!")
    
    with col2:
        run_steps = st.number_input("Steps to Run", 1, 52, 10)
    
    with col3:
        if st.button("▶️ Run Steps"):
            if st.session_state.env is None:
                st.warning("Please reset the simulation first!")
            else:
                run_simulation_steps(run_steps)
    
    # Display current state
    if st.session_state.env is not None:
        display_simulation_state()


def run_simulation_steps(n_steps: int):
    """Run simulation for n steps"""
    env = st.session_state.env
    agent = st.session_state.agent
    
    progress = st.progress(0)
    
    for i in range(n_steps):
        state = env.get_state_tuple()
        action = agent.get_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        
        # Update agent if it's RL
        if hasattr(agent, 'update'):
            next_state_tuple = tuple(next_state.tolist())
            agent.update(state, action, reward, next_state_tuple, done)
        
        # Record history
        st.session_state.history.append({
            'step': info['time_step'],
            'reward': reward,
            'discharged': info['discharged_this_step'],
            'deaths': info['deaths_this_step'],
            'total_discharged': info['total_discharged'],
            'total_deaths': info['total_deaths'],
            'power': info['total_power'],
            'water': info['total_water'],
        })
        
        progress.progress((i + 1) / n_steps)
        
        if done:
            st.info("Episode completed!")
            break
    
    progress.empty()


def display_simulation_state():
    """Display current simulation state"""
    env = st.session_state.env
    metrics = env.get_metrics()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("⏱️ Time Elapsed", f"{metrics['hours_elapsed']:.1f}h")
    with col2:
        st.metric("✅ Total Discharged", metrics['total_discharged'])
    with col3:
        st.metric("❌ Total Deaths", metrics['total_deaths'])
    with col4:
        st.metric("💰 Current Reward", f"{metrics['current_reward']:.1f}")
    with col5:
        disaster_status = "🔴 Active" if metrics['disaster']['active'] else "🟢 Ended"
        st.metric("🌪️ Disaster", disaster_status)
    
    st.divider()
    
    # Infrastructure Status
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Hospitals", "⚡ Power", "💧 Water", "🏛️ Venues"])
    
    with tab1:
        hospital_data = []
        for h in metrics['hospital_statuses']:
            hospital_data.append({
                "Name": h['name'],
                "Status": get_status_color(h['damage']),
                "Patients": f"{h['patients']}/{h['capacity']}",
                "Discharged": h['discharged'],
                "Deceased": h['deceased'],
                "Damage": f"{h['damage']*100:.1f}%",
                "Resources": f"{h['resource_satisfaction']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(hospital_data), use_container_width=True)
    
    with tab2:
        power_data = []
        for ps in metrics['power_statuses']:
            power_data.append({
                "Name": ps['name'],
                "Status": get_status_color(ps['damage']),
                "Output": f"{ps['output']:.0f}/{ps['capacity']:.0f} kW",
                "Damage": f"{ps['damage']*100:.1f}%",
                "Fuel": f"{ps['fuel']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(power_data), use_container_width=True)
    
    with tab3:
        water_data = []
        for ws in metrics['water_statuses']:
            water_data.append({
                "Name": ws['name'],
                "Status": get_status_color(ws['damage']),
                "Output": f"{ws['output']:.0f}/{ws['capacity']:.0f}",
                "Damage": f"{ws['damage']*100:.1f}%",
                "Reservoir": f"{ws['reservoir']*100:.1f}%",
                "Contamination": f"{ws['contamination']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(water_data), use_container_width=True)
    
    with tab4:
        venue_data = []
        for pv in metrics['venue_statuses']:
            venue_data.append({
                "Name": pv['name'],
                "Status": get_status_color(pv['damage']),
                "Population": f"{pv['population']}/{pv['capacity']}",
                "Casualties": pv['casualties'],
                "Damage": f"{pv['damage']*100:.1f}%",
                "Resources": f"{pv['resource_satisfaction']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(venue_data), use_container_width=True)
    
    # Charts
    if st.session_state.history:
        st.divider()
        st.subheader("📊 Performance Charts")
        
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df, x='step', y='reward', title='Reward over Time')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(df, x='step', y=['total_discharged', 'total_deaths'], 
                         title='Cumulative Outcomes')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def run_training(scenario: str, n_episodes: int, alpha: float, gamma: float, epsilon: float):
    """Run training mode"""
    st.header("🏋️ Agent Training")
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            # Initialize
            env = DisasterEnvironment(scenario_name=scenario, seed=42)
            agent = QLearningAgent(
                n_actions=env.n_actions,
                learning_rate=alpha,
                discount_factor=gamma,
                epsilon=epsilon
            )
            
            # Training loop with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            rewards = []
            discharged = []
            
            for episode in range(n_episodes):
                state = env.reset(scenario_name=scenario)
                state_tuple = tuple(state.tolist())
                episode_reward = 0
                done = False
                
                while not done:
                    action = agent.get_action(state_tuple, training=True)
                    next_state, reward, done, info = env.step(action)
                    next_state_tuple = tuple(next_state.tolist())
                    agent.update(state_tuple, action, reward, next_state_tuple, done)
                    state_tuple = next_state_tuple
                    episode_reward += reward
                
                agent.end_episode(episode_reward, env.time_step)
                rewards.append(episode_reward)
                discharged.append(env.total_discharged)
                
                # Update progress
                progress_bar.progress((episode + 1) / n_episodes)
                status_text.text(f"Episode {episode + 1}/{n_episodes} | Reward: {episode_reward:.1f} | Discharged: {env.total_discharged}")
                
                # Update chart every 5 episodes
                if (episode + 1) % 5 == 0:
                    df = pd.DataFrame({
                        'Episode': range(1, len(rewards) + 1),
                        'Reward': rewards,
                        'Discharged': discharged
                    })
                    fig = make_subplots(rows=1, cols=2, subplot_titles=['Rewards', 'Patients Discharged'])
                    fig.add_trace(go.Scatter(x=df['Episode'], y=df['Reward'], mode='lines', name='Reward'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['Episode'], y=df['Discharged'], mode='lines', name='Discharged'), row=1, col=2)
                    fig.update_layout(height=300, showlegend=False)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Save agent
            st.session_state.trained_agent = agent
            
            # Final results
            st.success(f"Training complete! Best reward: {max(rewards):.1f}")
            
            # Display final statistics
            stats = agent.get_statistics()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Reward", f"{stats.get('avg_reward_last_10', 0):.1f}")
            with col2:
                st.metric("Q-Table Size", stats['q_table_size'])
            with col3:
                st.metric("Final Epsilon", f"{stats['current_epsilon']:.4f}")


def run_evaluation(scenario: str):
    """Run evaluation mode"""
    st.header("📈 Agent Evaluation & Comparison")
    
    if 'trained_agent' not in st.session_state:
        st.warning("Please train an agent first in the 'Train Agent' mode.")
        return
    
    if st.button("🔍 Run Evaluation", type="primary"):
        with st.spinner("Evaluating agents..."):
            env = DisasterEnvironment(scenario_name=scenario, seed=42)
            
            # Create agents to compare
            agents = [
                ("RL Agent", st.session_state.trained_agent),
                ("Manual (Balanced)", ManualPolicy("balanced")),
                ("Manual (Hospital)", ManualPolicy("hospital_priority")),
                ("Adaptive Manual", AdaptiveManualPolicy()),
            ]
            
            # Evaluate
            evaluator = Evaluator(n_episodes=10, scenarios=[scenario])
            results = {}
            
            for name, agent in agents:
                result = evaluator.evaluate(agent, env, name)
                results[name] = result
            
            # Display results
            st.subheader("📊 Comparison Results")
            
            # Create comparison dataframe
            comparison_data = {
                'Agent': [],
                'Avg Reward': [],
                'Std Reward': [],
                'Avg Discharged': [],
                'Avg Deaths': []
            }
            
            for name, result in results.items():
                comparison_data['Agent'].append(name)
                comparison_data['Avg Reward'].append(result['avg_reward'])
                comparison_data['Std Reward'].append(result['std_reward'])
                comparison_data['Avg Discharged'].append(result['avg_discharged'])
                comparison_data['Avg Deaths'].append(result['avg_deaths'])
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Agent', y='Avg Reward', title='Average Reward Comparison',
                           color='Agent', error_y='Std Reward')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Agent', y=['Avg Discharged', 'Avg Deaths'], 
                           title='Healthcare Outcomes', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            # Winner
            winner = df.loc[df['Avg Reward'].idxmax(), 'Agent']
            st.success(f"🏆 Best Performing Agent: **{winner}**")


if __name__ == "__main__":
    main()
