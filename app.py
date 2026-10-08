"""
Streamlit Dashboard for Real-Time Disaster Management Simulation
Multi-city world builder, training, evaluation, policy inspector
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_dark"
import json
from pathlib import Path

from world import WorldConfig
from environment import DisasterEnvironment
from agent import QLearningAgent, ManualPolicy, AdaptiveManualPolicy
from trainer import Evaluator
from config import DISASTER_SCENARIOS, ACTION_CONFIG, OPS_CONFIG, RL_CONFIG


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
    .stApp { background-color: #0d1117; }
    .stMetric {
        background-color: #161b26;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid #262f42;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }
    .stMetric label, .stMetric [data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricLabel"], .stMetric [data-testid="stMetricDelta"] {
        color: #e8edf5 !important;
    }
    .city-card {
        background-color: #161b26;
        border: 1px solid #262f42;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .city-card h4 { margin: 0 0 4px 0; }
    .city-card .muted { color: #9aa7bd; font-size: 0.85rem; }
    .report-header {
        background: linear-gradient(90deg, #1d2b45, #2f4d7d);
        border: 1px solid #31435f;
        color: #e8edf5;
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .report-header h3 { margin: 0; color: #e8edf5; }
    .report-header p { margin: 2px 0 0 0; color: #b9c8de; font-size: 0.9rem; }
    div[data-testid="stSidebar"] { background-color: #11151f; }
    hr { border-color: #262f42; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")
OP_NAMES = OPS_CONFIG["operations"]
RATIO_NAMES = ["Hospitals", "Venues", "Reserve"]


def get_status_color(damage_level: float) -> str:
    if damage_level < 0.25:
        return "🟢"
    elif damage_level < 0.5:
        return "🟡"
    elif damage_level < 0.75:
        return "🟠"
    return "🔴"


def decode_action_labels(action: int, env: DisasterEnvironment) -> str:
    """Human-readable label for a combined action"""
    elec, water, op = env.decode_action(action)
    return f"E{elec}/W{water}/{OP_NAMES[op]}"


def list_saved_models() -> list:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.pkl"))


def build_agent(agent_type: str, env: DisasterEnvironment, model_name: str = None):
    """Create an agent from the sidebar selection"""
    if agent_type == "Q-Learning (RL)":
        return QLearningAgent(n_actions=env.n_actions)
    if agent_type == "Load trained model":
        if not model_name:
            return None
        agent = QLearningAgent(n_actions=env.n_actions)
        agent.load(str(MODELS_DIR / model_name))
        return agent
    if agent_type == "Manual (Balanced)":
        return ManualPolicy("balanced")
    if agent_type == "Manual (Hospital Priority)":
        return ManualPolicy("hospital_priority")
    return AdaptiveManualPolicy()


def world_config_from_sidebar() -> WorldConfig:
    """World builder controls"""
    with st.sidebar.expander("🌍 World Builder", expanded=True):
        n_cities = st.slider("Number of Cities", 1, 5, 1)
        city_population = st.number_input(
            "Population per City", min_value=2_000, max_value=100_000,
            value=10_000, step=1_000
        )
        hospitals = st.slider("Hospitals per City", 1, 6, 3)
        power = st.slider("Power Stations per City", 1, 4, 2)
        water = st.slider("Water Stations per City", 1, 4, 2)
        venues = st.slider("Public Venues per City", 1, 4, 2)

    with st.sidebar.expander("🌪️ Disaster", expanded=True):
        scenario = st.selectbox(
            "Disaster Type", [s["name"] for s in DISASTER_SCENARIOS], index=0
        )
        magnitude = st.slider("Magnitude (1-10)", 1.0, 10.0, 5.0, 0.5)
        epicenter = st.selectbox(
            "Epicenter", list(range(n_cities)),
            format_func=lambda i: f"City_{i+1}"
        )

    with st.sidebar.expander("🔧 Realism", expanded=False):
        delivery_delay = st.slider("Resource Delivery Delay (steps)", 0, 6, 2)
        repair_crews = st.slider("Repair Crews", 0, 5, 2)
        transfer_delay = st.slider("Inter-city Aid Delay (steps)", 1, 12, 6)

    return WorldConfig(
        n_cities=n_cities,
        hospitals_per_city=hospitals,
        power_stations_per_city=power,
        water_stations_per_city=water,
        venues_per_city=venues,
        city_population=int(city_population),
        epicenter_city=epicenter,
        magnitude=magnitude,
        scenario_type=scenario,
        repair_crews=repair_crews,
        delivery_delay_steps=delivery_delay,
        inter_city_transfer_delay=transfer_delay,
    )


def main():
    st.title("🚨 Real-Time Disaster Management Simulation")
    st.markdown("### AI-Powered Decision Support System using Reinforcement Learning")

    config = world_config_from_sidebar()

    with st.sidebar:
        st.divider()
        agent_type = st.selectbox(
            "Agent Type",
            ["Q-Learning (RL)", "Load trained model", "Manual (Balanced)",
             "Manual (Hospital Priority)", "Adaptive Manual"],
            index=0
        )
        model_name = None
        if agent_type == "Load trained model":
            saved = list_saved_models()
            if saved:
                model_name = st.selectbox("Model", saved)
            else:
                st.warning("No saved models in models/")

        st.divider()
        st.subheader("📊 Training Parameters")
        n_episodes = st.slider("Training Episodes", 10, 500, 200)
        learning_rate = st.slider(
            "Learning Rate (α)", 0.01, 1.0,
            float(RL_CONFIG["learning_rate_alpha"])
        )
        discount_factor = st.slider(
            "Discount Factor (γ)", 0.1, 1.0,
            float(RL_CONFIG["discount_factor_gamma"])
        )
        epsilon = st.slider(
            "Initial Exploration (ε)", 0.1, 1.0,
            float(RL_CONFIG["exploration_rate_epsilon"])
        )

    # Initialize session state
    for key, default in [("env", None), ("agent", None), ("history", []),
                         ("trained_agent", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    tab_sim, tab_train, tab_eval, tab_policy = st.tabs([
        "🎮 Interactive Simulation", "🏋️ Train Agent",
        "📈 Evaluate & Compare", "🔍 Policy Inspector",
    ])
    with tab_sim:
        run_interactive_simulation(config, agent_type, model_name)
    with tab_train:
        run_training(config, n_episodes, learning_rate, discount_factor, epsilon)
    with tab_eval:
        run_evaluation(config, agent_type, model_name)
    with tab_policy:
        run_policy_inspector(config, model_name)


# ---------------------------------------------------------------------------
# Interactive simulation
# ---------------------------------------------------------------------------
def run_interactive_simulation(config: WorldConfig, agent_type: str, model_name: str):
    st.header("🎮 Interactive Simulation")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔄 Reset Simulation", type="primary"):
            st.session_state.env = DisasterEnvironment(world_config=config, seed=42)
            st.session_state.agent = build_agent(agent_type, st.session_state.env, model_name)
            st.session_state.history = []
            if st.session_state.agent is None:
                st.warning("Select a model to load first.")
            else:
                st.success("Simulation reset!")

    with col2:
        run_steps = st.number_input("Steps to Run", 1, 52, 10)

    with col3:
        if st.button("▶️ Run Steps"):
            if st.session_state.env is None or st.session_state.agent is None:
                st.warning("Please reset the simulation first!")
            else:
                run_simulation_steps(run_steps)

    if st.session_state.env is not None:
        display_simulation_state()


def run_simulation_steps(n_steps: int):
    env = st.session_state.env
    agent = st.session_state.agent
    is_learning_agent = isinstance(agent, QLearningAgent) and not agent.total_steps

    progress = st.progress(0)

    for i in range(n_steps):
        state = env.get_state_tuple()
        # Fresh Q-Learning agents learn online; loaded/manual agents act greedily
        action = agent.get_action(state, training=bool(is_learning_agent))
        next_state, reward, done, info = env.step(action)

        if is_learning_agent and hasattr(agent, "update"):
            agent.update(state, action, reward, tuple(next_state.tolist()), done)

        st.session_state.history.append({
            "step": info["time_step"],
            "reward": reward,
            "discharged": info["discharged_this_step"],
            "deaths": info["deaths_this_step"],
            "total_discharged": info["total_discharged"],
            "total_deaths": info["total_deaths"],
            "power": info["total_power"],
            "water": info["total_water"],
            "action": decode_action_labels(action, env),
        })

        progress.progress((i + 1) / n_steps)

        if done:
            st.info("Episode completed!")
            break

    progress.empty()


def display_simulation_state():
    env = st.session_state.env
    metrics = env.get_metrics()

    pop = metrics["report"]["population"]
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("⏱️ Time Elapsed", f"{metrics['hours_elapsed']:.1f}h")
    with col2:
        st.metric("👥 Total Population", f"{pop['current']:,}",
                  delta=f"-{pop['initial'] - pop['current']:,}" if pop['current'] < pop['initial'] else None,
                  delta_color="inverse")
    with col3:
        st.metric("✅ Total Discharged", metrics["total_discharged"])
    with col4:
        st.metric("❌ Total Deaths", metrics["total_deaths"])
    with col5:
        st.metric("💰 Current Reward", f"{metrics['current_reward']:.1f}")
    with col6:
        status = "🔴 Active" if metrics["disaster"]["active"] else "🟢 Ended"
        st.metric("🌪️ Disaster", status)

    # Export current run
    if st.session_state.history:
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download run history (CSV)",
                pd.DataFrame(st.session_state.history).to_csv(index=False),
                file_name="run_history.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                "⬇️ Download metrics snapshot (JSON)",
                json.dumps(metrics, default=str, indent=2),
                file_name="metrics.json",
                mime="application/json",
            )

    st.divider()

    # City overview map (bubble: size = population, color = damage)
    if metrics["n_cities"] > 1:
        st.subheader("🗺️ City Overview")
        map_df = pd.DataFrame([{
            "City": c["name"],
            "Avg Damage (%)": c["avg_damage"] * 100,
            "Population": c["population"],
            "Deaths": next((r["deaths"] for r in metrics["report"]["per_city"]
                            if r["name"] == c["name"]), 0),
        } for c in metrics["cities"]])
        fig = px.scatter(
            map_df, x="City", y="Avg Damage (%)", size="Population",
            color="Avg Damage (%)", color_continuous_scale="Reds",
            hover_data=["Deaths"], size_max=60,
        )
        fig.update_layout(height=280, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", key="city_overview_map")

    # Per-city views
    city_tabs = st.tabs([f"🏙️ {c['name']}" for c in metrics["cities"]])
    for tab, city in zip(city_tabs, metrics["cities"]):
        with tab:
            display_city(city)

    # System-wide run charts (rendered once — not per city)
    if st.session_state.history:
        st.divider()
        st.subheader("📊 Performance Charts")
        df = pd.DataFrame(st.session_state.history)
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.line(df, x="step", y="reward", title="Reward over Time")
            fig.update_layout(height=280)
            st.plotly_chart(fig, width="stretch", key="run_reward_chart")
        with col2:
            fig = px.line(df, x="step", y=["total_discharged", "total_deaths"],
                          title="Cumulative Outcomes")
            fig.update_layout(height=280)
            st.plotly_chart(fig, width="stretch", key="run_outcomes_chart")
        with col3:
            pop_df = pd.DataFrame({
                "step": env.history["time_steps"],
                "Total": env.history["total_population"],
                "Sheltered": env.history["sheltered_population"],
            })
            fig = px.line(pop_df, x="step", y=["Total", "Sheltered"],
                          title="Population")
            fig.update_layout(height=280)
            st.plotly_chart(fig, width="stretch", key="run_population_chart")

        display_decision_log(env)
        display_after_action_report(env)


def display_decision_log(env: DisasterEnvironment):
    """Step-by-step record of what the agent did and why"""
    if not env.action_log:
        return

    st.divider()
    st.subheader("🧠 Decision Log")

    agent = st.session_state.agent
    is_rl = isinstance(agent, QLearningAgent)

    df = pd.DataFrame([{
        "Step": e["step"],
        "Action taken": e["description"],
        "Reward": round(e["reward"], 1),
        "Discharged": e["discharged"],
        "Deaths": e["deaths"],
    } for e in env.action_log])
    st.dataframe(df, width="stretch", hide_index=True, height=300)

    # Q-value inspector: why did the RL agent choose that action?
    if is_rl and agent.q_table:
        with st.expander("🔎 Why did the RL agent choose an action? (Q-value inspector)"):
            step_choice = st.selectbox(
                "Inspect step", [e["step"] for e in env.action_log],
                index=len(env.action_log) - 1,
            )
            entry = next(e for e in env.action_log if e["step"] == step_choice)
            q_values = agent.q_table.get(entry["state"])

            if q_values is None or not np.any(q_values != 0):
                st.info("This state was never visited during training — "
                        "all Q-values are zero, so the agent is guessing.")
            else:
                top = np.argsort(q_values)[::-1][:3]
                rows = []
                for rank, a in enumerate(top, 1):
                    rows.append({
                        "Rank": rank,
                        "Action": env.describe_action(int(a)),
                        "Q-value": round(float(q_values[a]), 2),
                        "Chosen": "✅" if int(a) == entry["action"] else "",
                    })
                st.markdown(f"**State:** `{entry['state']}`")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Download decision log (CSV)",
        df.to_csv(index=False),
        file_name="decision_log.csv",
        mime="text/csv",
    )


def display_after_action_report(env: DisasterEnvironment):
    """Detailed after-action report: causes, per-city outcomes, logistics"""
    report = env.get_episode_report()

    st.divider()
    st.markdown(
        f"""<div class="report-header">
            <h3>📋 After-Action Report</h3>
            <p>{report['scenario']} · magnitude {report['magnitude']} ·
            {report['n_cities']} cit{'ies' if report['n_cities'] > 1 else 'y'} ·
            {report['hours_elapsed']:.1f} hours simulated</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Headline KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("❌ Deaths", report["total_deaths"])
    with c2:
        st.metric("✅ Discharged", report["total_discharged"])
    with c3:
        st.metric("🚶 Evacuated", report["total_evacuated"])
    with c4:
        st.metric("🚚 Aid Convoys", report["aid_delivered"]["convoys"])
    with c5:
        st.metric("⚡ Resource Utilization",
                  f"{report['resources']['utilization']*100:.1f}%")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Deaths by cause
        causes = report["deaths_by_cause"]
        if sum(causes.values()) > 0:
            fig = px.pie(
                names=["Hospitals", "Public Venues", "At Home"],
                values=[causes["hospital"], causes["venue"], causes["at_home"]],
                title="Deaths by Cause",
                color_discrete_sequence=["#c0392b", "#e67e22", "#7f8c8d"],
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch", key="report_deaths_pie")

        # Aid delivered breakdown
        st.markdown("**🚚 Aid Delivered**")
        aid = report["aid_delivered"]
        st.caption(
            f"Convoys: {aid['convoys']} | Water: {aid['water']:.0f} units | "
            f"Medical kits: {aid['medical_kits']} | Fuel: {aid['fuel']*100:.0f}%"
        )
        st.markdown("**🏚️ Infrastructure Failures**")
        st.caption(f"{report['infrastructure_failures']} facilities non-operational")

    with col2:
        # Per-city outcomes table
        st.markdown("**🏙️ Outcomes by City**")
        city_df = pd.DataFrame([{
            "City": c["name"],
            "Epicenter": "⚠️" if c["is_epicenter"] else "",
            "Population": f"{c['population']:,}",
            "Deaths": c["deaths"],
            "Discharged": c["discharged"],
            "Avg Damage": f"{c['avg_damage']*100:.1f}%",
            "Hospital Load": f"{c['hospital_load']*100:.0f}%",
            "Fuel Left": f"{c['fuel_remaining']*100:.0f}%",
            "Contamination": f"{c['contamination']*100:.0f}%",
        } for c in report["per_city"]])
        st.dataframe(city_df, width="stretch", hide_index=True)

        # Resource generation vs waste
        res = report["resources"]
        fig = go.Figure(go.Bar(
            x=["Power", "Water"],
            y=[res["power_generated"], res["water_generated"]],
            name="Generated", marker_color="#35618e",
        ))
        fig.add_trace(go.Bar(
            x=["Power", "Water"],
            y=[res["power_wasted"], res["water_wasted"]],
            name="Wasted (reserve)", marker_color="#c0392b",
        ))
        fig.update_layout(height=260, barmode="overlay",
                          title="Resources Generated vs Wasted",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", key="report_resources_bar")

    # Downloads
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "⬇️ Download full report (JSON)",
            json.dumps(report, default=str, indent=2),
            file_name="after_action_report.json",
            mime="application/json",
        )
    with col_b:
        st.download_button(
            "⬇️ Download per-city outcomes (CSV)",
            pd.DataFrame(report["per_city"]).to_csv(index=False),
            file_name="city_outcomes.csv",
            mime="text/csv",
        )


def display_city(city: dict):
    epicenter = " ⚠️ EPICENTER" if city["distance_from_epicenter"] == 0 else ""
    st.caption(
        f"Population: {city['population']:,} | At home: {city['at_home_population']:,} | "
        f"Avg damage: {city['avg_damage']*100:.1f}% | "
        f"Aid convoys incoming: {city['aid_convoys_incoming']}{epicenter}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏥 Hospitals**")
        st.dataframe(pd.DataFrame([{
            "Name": h["name"], "Status": get_status_color(h["damage"]),
            "Patients": f"{h['patients']}/{h['capacity']}",
            "Discharged": h["discharged"], "Deceased": h["deceased"],
            "Damage": f"{h['damage']*100:.1f}%",
            "Resources": f"{h['resource_satisfaction']*100:.1f}%",
        } for h in city["hospitals"]]), width="stretch")

        st.markdown("**🏛️ Public Venues**")
        st.dataframe(pd.DataFrame([{
            "Name": v["name"], "Status": get_status_color(v["damage"]),
            "Population": f"{v['population']}/{v['capacity']}",
            "Casualties": v["casualties"],
            "Damage": f"{v['damage']*100:.1f}%",
            "Resources": f"{v['resource_satisfaction']*100:.1f}%",
        } for v in city["venues"]]), width="stretch")

    with col2:
        st.markdown("**⚡ Power Stations**")
        st.dataframe(pd.DataFrame([{
            "Name": p["name"], "Status": get_status_color(p["damage"]),
            "Output": f"{p['output']:.0f}/{p['capacity']:.0f} kW",
            "Damage": f"{p['damage']*100:.1f}%",
            "Fuel": f"{p['fuel']*100:.1f}%",
        } for p in city["power_stations"]]), width="stretch")

        st.markdown("**💧 Water Stations**")
        st.dataframe(pd.DataFrame([{
            "Name": w["name"], "Status": get_status_color(w["damage"]),
            "Output": f"{w['output']:.0f}/{w['capacity']:.0f}",
            "Damage": f"{w['damage']*100:.1f}%",
            "Reservoir": f"{w['reservoir']*100:.1f}%",
            "Contamination": f"{w['contamination']*100:.1f}%",
        } for w in city["water_stations"]]), width="stretch")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(config: WorldConfig, n_episodes: int, alpha: float, gamma: float, epsilon: float):
    st.header("🏋️ Agent Training")
    st.caption(
        f"World: {config.n_cities} cit{'ies' if config.n_cities > 1 else 'y'}, "
        f"~{config.city_population:,} people each, {config.scenario_type} "
        f"magnitude {config.magnitude}"
    )

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            env = DisasterEnvironment(world_config=config, seed=42)
            agent = QLearningAgent(
                n_actions=env.n_actions,
                learning_rate=alpha,
                discount_factor=gamma,
                epsilon=epsilon,
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()

            rewards, discharged = [], []

            for episode in range(n_episodes):
                state = env.reset(scenario_name=config.scenario_type)
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

                progress_bar.progress((episode + 1) / n_episodes)
                status_text.text(
                    f"Episode {episode + 1}/{n_episodes} | Reward: {episode_reward:.1f} "
                    f"| Discharged: {env.total_discharged} | Deaths: {env.total_deaths}"
                )

                if (episode + 1) % 5 == 0:
                    df = pd.DataFrame({
                        "Episode": range(1, len(rewards) + 1),
                        "Reward": rewards, "Discharged": discharged,
                    })
                    fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=["Rewards", "Patients Discharged"])
                    fig.add_trace(go.Scatter(x=df["Episode"], y=df["Reward"],
                                             mode="lines", name="Reward"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df["Episode"], y=df["Discharged"],
                                             mode="lines", name="Discharged"), row=1, col=2)
                    fig.update_layout(height=300, showlegend=False)
                    chart_placeholder.plotly_chart(fig, width="stretch", key="train_progress_chart")

            st.session_state.trained_agent = agent
            st.session_state.training_rewards = rewards

            # Save + export
            MODELS_DIR.mkdir(exist_ok=True)
            save_path = MODELS_DIR / "dashboard_agent.pkl"
            agent.save(str(save_path))
            st.success(f"Training complete! Best reward: {max(rewards):.1f} — saved to {save_path}")

            stats = agent.get_statistics()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Reward (last 10)", f"{stats.get('avg_reward_last_10', 0):.1f}")
            with col2:
                st.metric("Q-Table Size", stats["q_table_size"])
            with col3:
                st.metric("Final Epsilon", f"{stats['current_epsilon']:.4f}")

            st.download_button(
                "⬇️ Download training history (CSV)",
                pd.DataFrame({"episode": range(1, len(rewards) + 1),
                              "reward": rewards, "discharged": discharged}).to_csv(index=False),
                file_name="training_history.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(config: WorldConfig, agent_type: str, model_name: str):
    st.header("📈 Agent Evaluation & Comparison")

    rl_agent = st.session_state.trained_agent
    if rl_agent is None and agent_type == "Load trained model" and model_name:
        env_for_load = DisasterEnvironment(world_config=config, seed=42)
        rl_agent = build_agent("Load trained model", env_for_load, model_name)
    if rl_agent is None:
        saved = list_saved_models()
        if saved:
            st.info("No in-session trained agent found — pick a saved model to evaluate.")
            choice = st.selectbox("Saved model", saved, key="eval_saved_model")
            if choice:
                env_for_load = DisasterEnvironment(world_config=config, seed=42)
                rl_agent = build_agent("Load trained model", env_for_load, choice)
    if rl_agent is None:
        st.warning("Train an agent first, or save one to the models/ directory.")
        return

    if st.button("🔍 Run Evaluation", type="primary"):
        with st.spinner("Evaluating agents..."):
            env = DisasterEnvironment(world_config=config, seed=42)
            agents = [
                ("RL Agent", rl_agent),
                ("Manual (Balanced)", ManualPolicy("balanced")),
                ("Manual (Hospital)", ManualPolicy("hospital_priority")),
                ("Adaptive Manual", AdaptiveManualPolicy()),
            ]

            evaluator = Evaluator(n_episodes=10, scenarios=[config.scenario_type], seed=42)
            results = {name: evaluator.evaluate(agent, env, name) for name, agent in agents}

            df = pd.DataFrame([
                {
                    "Agent": name,
                    "Avg Reward": r["avg_reward"],
                    "Std Reward": r["std_reward"],
                    "Avg Discharged": r["avg_discharged"],
                    "Avg Deaths": r["avg_deaths"],
                }
                for name, r in results.items()
            ])
            st.dataframe(df, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df, x="Agent", y="Avg Reward",
                             title="Average Reward Comparison",
                             color="Agent", error_y="Std Reward")
                st.plotly_chart(fig, width="stretch", key="eval_reward_bar")
            with col2:
                fig = px.bar(df, x="Agent", y=["Avg Discharged", "Avg Deaths"],
                             title="Healthcare Outcomes", barmode="group")
                st.plotly_chart(fig, width="stretch", key="eval_outcomes_bar")

            winner = df.loc[df["Avg Reward"].idxmax(), "Agent"]
            st.success(f"🏆 Best Performing Agent: **{winner}**")

            st.download_button(
                "⬇️ Download comparison (CSV)",
                df.to_csv(index=False),
                file_name="comparison.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Policy inspector
# ---------------------------------------------------------------------------
def run_policy_inspector(config: WorldConfig, model_name: str):
    st.header("🔍 Policy Inspector")

    agent = st.session_state.trained_agent
    if agent is None and model_name:
        env_for_load = DisasterEnvironment(world_config=config, seed=42)
        agent = build_agent("Load trained model", env_for_load, model_name)
    if agent is None:
        saved = list_saved_models()
        if saved:
            choice = st.selectbox("Saved model", saved, key="policy_saved_model")
            if choice:
                env_for_load = DisasterEnvironment(world_config=config, seed=42)
                agent = build_agent("Load trained model", env_for_load, choice)
    if agent is None:
        st.warning("Train an agent or load a saved model to inspect its policy.")
        return

    env = DisasterEnvironment(world_config=config, seed=42)
    q_table = agent.q_table
    st.caption(f"Q-table: {len(q_table)} states | {agent.n_actions} actions")

    # Action distribution from training
    st.subheader("Action Usage Distribution")
    action_dist = agent.training_stats.get("action_distribution", {})
    if action_dist:
        labels = [decode_action_labels(int(a), env) for a in action_dist.keys()]
        fig = px.bar(x=labels, y=list(action_dist.values()),
                     title="Actions taken during training",
                     labels={"x": "Action (Elec/Water/Op)", "y": "Count"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch", key="insp_action_dist")
    else:
        st.info("No action distribution recorded (agent was loaded from disk).")

    # Operation preference heatmap over (city damage x hospital satisfaction)
    st.subheader("Preferred Operation by City Condition")
    damage_labels = ["None", "Minor", "Moderate", "Severe", "Critical"]
    satisfaction_labels = ["None", "Critical", "Low", "Medium", "Full"]

    n_cities = len(env.cities)
    best_ops = np.full((5, 5), np.nan)
    best_ops_text = [["" for _ in range(5)] for _ in range(5)]

    for damage in range(5):
        for satisfaction in range(5):
            # Synthetic state: every city at (damage, satisfaction), neutral elsewhere
            state = []
            for _ in range(n_cities):
                state += [damage, satisfaction, 2, 2, 0, 2]
            state += [1, 2]
            q = q_table.get(tuple(state))
            if q is not None and np.any(q != 0):
                best = int(np.argmax(q))
                _, _, op = env.decode_action(best)
                best_ops[damage][satisfaction] = op
                best_ops_text[damage][satisfaction] = OP_NAMES[op].replace("_", " ")

    if not np.isnan(best_ops).all():
        fig = go.Figure(go.Heatmap(
            z=best_ops, x=satisfaction_labels, y=damage_labels,
            text=best_ops_text, texttemplate="%{text}",
            colorscale="Viridis", showscale=False,
        ))
        fig.update_layout(
            title="Best operation per (city damage, hospital resources) — visited states only",
            xaxis_title="Hospital resource satisfaction",
            yaxis_title="Average city damage",
            height=400,
        )
        st.plotly_chart(fig, width="stretch", key="insp_heatmap")
    else:
        st.info("No matching visited states for the heatmap grid. Train longer to populate it.")

    # Top states by max Q-value
    st.subheader("Highest-Value States")
    if q_table:
        rows = []
        for state, q in q_table.items():
            best = int(np.argmax(q))
            rows.append({
                "state": str(state),
                "best_action": decode_action_labels(best, env),
                "max_q": float(np.max(q)),
            })
        top = pd.DataFrame(rows).sort_values("max_q", ascending=False).head(20)
        st.dataframe(top, width="stretch")

        st.download_button(
            "⬇️ Download policy table (CSV)",
            pd.DataFrame(rows).to_csv(index=False),
            file_name="policy.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
