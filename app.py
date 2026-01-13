import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- 1. CONFIG & SYSTEM THEME ---
st.set_page_config(page_title="Elite Scouting Engine", layout="wide")

st.markdown(f"""
    <style>
    /* Main Background & Sidebar: Fixed to Midnight/Slate */
    .stApp {{ background-color: #0e1117; color: #e0e0e0; }}
    [data-testid="stSidebar"] {{ 
        background-color: #acb3bd; 
        border-right: 1px solid #30363d; 
    }}  
            
    /* ELIMINATE THE WHITE TOP BAR */
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    .block-container {{
        padding-top: 2rem;
    }}

    /* Metric Card Styling */
    [data-testid="stMetricLabel"] {{ color: #8b949e !important; font-size: 0.9rem !important; }}
    [data-testid="stMetricValue"] {{ color: #ffffff !important; font-weight: 700 !important; }}
    div[data-testid="stMetric"] {{
        background: #1c2128;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 12px;
    }}

    /* Candidate Cards: Professional Slate Gradient */
    .candidate-card {{
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 12px;
        transition: transform 0.2s ease;
    }}
    .candidate-card:hover {{ border-color: #00d4ff; transform: scale(1.01); }}
    .candidate-name {{ font-size: 1.15rem; font-weight: 700; color: #ffffff; }}
    .candidate-subtitle {{ color: #8b949e; font-size: 0.85rem; text-transform: uppercase; }}
    .similarity-score {{ font-size: 1.8rem; font-weight: 800; color: #00d4ff; text-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA UTILITIES ---
#def clean_names(text):
    #if not isinstance(text, str): return text
    #return text.replace(" Football Club", "").replace(" FC", "").replace("-", " ").title()

def format_pretty(n, currency=False):
    """Fixed: Ensures numbers like 40,000,000 show as €40.0M."""
    try:
        if currency:
            if n >= 1_000_000: return f"€{n/1_000_000:.1f}M"
            return f"€{n:,.0f}"
        return f"{n:,.0f}"
    except: return str(n)

@st.cache_data
def load_and_clean_data():
    players = pd.read_csv('data/players.csv')
    valuations = pd.read_csv('data/player_valuations.csv')
    appearances = pd.read_csv('data/appearances.csv')
    clubs = pd.read_csv('data/clubs.csv')
    competitions = pd.read_csv('data/competitions.csv')
    
    club_info = clubs[['club_id', 'name', 'domestic_competition_id']].rename(columns={'name': 'club_name'})
    league_info = club_info.merge(competitions[['competition_id', 'name']], left_on='domestic_competition_id', right_on='competition_id', how='left').rename(columns={'name': 'league'})
    players = players.merge(league_info[['club_id', 'club_name', 'league']], left_on='current_club_id', right_on='club_id', how='left')

    #players['club_name'] = players['club_name'].apply(clean_names)
    #players['league'] = players['league'].apply(clean_names)

    perf = appearances.groupby('player_id').agg({'goals': 'sum', 'assists': 'sum', 'minutes_played': 'sum'}).reset_index()
    perf['goals_per_90'] = (perf['goals'] / (perf['minutes_played'] + 1)) * 90
    perf['assists_per_90'] = (perf['assists'] / (perf['minutes_played'] + 1)) * 90

    val_col = next((c for c in valuations.columns if 'market_value' in c), valuations.columns[-1])
    latest_val = valuations.sort_values('date').groupby('player_id')[val_col].last().reset_index()
    latest_val.rename(columns={val_col: 'price'}, inplace=True)

    df = players.merge(latest_val, on='player_id', how='inner').merge(perf, on='player_id', how='inner')
    
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        df['age'] = (datetime.now() - df['date_of_birth']).dt.days // 365
    df['age'] = df['age'].fillna(25).astype(int)
    df['height_in_cm'] = pd.to_numeric(df['height_in_cm'], errors='coerce').round(1)

    return df[df['minutes_played'] >= 450].fillna(0)

df = load_and_clean_data()
factor_map = {'goals_per_90': 'Goals per 90', 'assists_per_90': 'Assists per 90', 'minutes_played': 'Minutes Played', 'height_in_cm': 'Height (cm)'}
reverse_map = {v: k for k, v in factor_map.items()}

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🔍 Scouting Filters")
    target_name = st.selectbox("Select Target Player:", sorted(df['name'].unique()))
    target_row = df[df['name'] == target_name].iloc[0]
    
    st.subheader("Tactical Profile Builder")
    options = ['Height (cm)', 'Minutes Played'] if "Goalkeeper" in str(target_row['position']) else list(factor_map.values())
    
    # Selection Order dictates weight
    selected_labels = st.multiselect("Select and Order Factors:", options=options, default=options)
    selected_factors = [reverse_map[name] for name in selected_labels]

    budget = st.slider("Max Budget (€)", 0, int(df['price'].max()), 200_000_000, step=1_000_000)
    st.sidebar.markdown(f"{format_pretty(budget, currency=True)}")
    age_col1, age_col2 = st.columns(2)
    min_age = age_col1.number_input("Min Age", 15, 50, 15)
    max_age = age_col2.number_input("Max Age", 15, 50, 50)
    
    run_engine = st.button("🚀 Run Scouting Engine", use_container_width=True)

# --- 4. ENGINE LOGIC (RANKED MAGNITUDE) ---
if run_engine:
    mask = (df['price'] <= budget) & (df['position'] == target_row['position']) & \
           (df['name'] != target_name) & (df['age'] >= min_age) & (df['age'] <= max_age)
    candidates = df[mask].copy()
    
    if not candidates.empty and selected_factors:
        # Step 1: Assign 3.0x Anchor weight to the FIRST selected factor
        num = len(selected_factors)
        weights = [1.0 + (i * 0.5) for i in range(num)][::-1]
        weights[0] = 3.0 # The "Goals/90" priority boost
        
        scaler = StandardScaler()
        combined = pd.concat([candidates[selected_factors], pd.DataFrame([target_row[selected_factors]])])
        scaled = scaler.fit_transform(combined)
        
        # Step 2: Scale vectors by rank-based weights
        w_vec = np.array(weights)
        scaled_weighted = scaled * w_vec
        
        # Step 3: Euclidean Magnitude Similarity
        target_vec = scaled_weighted[-1]
        distances = np.linalg.norm(scaled_weighted[:-1] - target_vec, axis=1)
        candidates['similarity'] = 1 / (1 + distances)
        
        # Step 4: Scale to 100%
        max_sim = candidates['similarity'].max()
        candidates['similarity_pct'] = (candidates['similarity'] / max_sim) * 100
        
        st.session_state.results = candidates.sort_values('similarity_pct', ascending=False).head(5)
        st.session_state.last_target = target_name

# --- 5. UI RENDERING ---
if st.session_state.get('results') is None:
    # --- LANDING PAGE RESTORED ---
    st.title("🛰️ Elite Scouting Engine")
    st.subheader("System Status: Ready for Deployment")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### Tactical Instructions:
        1. **Target Benchmark:** Select a player (e.g., Cristiano Ronaldo) from the sidebar.
        2. **Order of Importance:** Click your primary factor (e.g., Goals/90) **first**. The engine will apply a **3.0x anchor weight** to it.
        3. **Recruitment Range:** Set your budget and age parameters.
        4. **Deploy:** Click 'Run' to find candidates who actually match the output magnitude.
        """)
    with col_b:
        st.info("💡 **Scout's Note:** Using Magnitude-Sensitive similarity ensures we find players who match the volume of output, not just the statistical 'shape'.")
else:
    target_data = df[df['name'] == st.session_state.last_target].iloc[0]
    st.title(f"Scouting Report: {target_data['name']}")
    
    # Metric Cards
    m_cols = st.columns(len(selected_factors) + 1)
    m_cols[0].metric("Market Value", format_pretty(target_data['price'], currency=True))
    for i, f in enumerate(selected_factors):
        val = target_data[f]
        fmt = f"{val:.2f}" if "per_90" in f else f"{val:.1f}" if "height" in f else format_pretty(val)
        m_cols[i+1].metric(factor_map[f], fmt)

    st.divider()
    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.subheader("Top Match Candidates")
        for _, row in st.session_state.results.iterrows():
            st.markdown(f"""
                <div class="candidate-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div class="candidate-name">{row['name']} ({int(row['age'])})</div>
                            <div class="candidate-subtitle">{row['club_name']} • {row['league']}</div>
                            <div style="color: #58a6ff; font-weight: bold; margin-top: 4px;">{format_pretty(row['price'], currency=True)}</div>
                        </div>
                        <div class="similarity-score">{int(row['similarity_pct'])}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with c2:
        st.subheader("Statistical Profile Overlay")
        if len(selected_factors) >= 3:
            comp_player = st.selectbox("Compare against:", st.session_state.results['name'].tolist())
            alt_data = df[df['name'] == comp_player].iloc[0]
            
            fig = go.Figure()
            # Hover Tooltips Restored
            for p in [{'n': target_data['name'], 'd': target_data, 'c': 'cyan'}, {'n': alt_data['name'], 'd': alt_data, 'c': '#FF0066'}]:
                p_data = p['d']
                r_shape = [p_data[m] / (df[m].max() if df[m].max() != 0 else 1) for m in selected_factors]
                r_actual = [f"{p_data[m]:.2f}" if "per_90" in m else f"{p_data[m]:.1f}" if "height" in m else format_pretty(p_data[m]) for m in selected_factors]
                
                fig.add_trace(go.Scatterpolar(
                    r=r_shape, theta=selected_labels, fill='toself', name=p['n'],
                    line=dict(color=p['c'], width=2),
                    customdata=r_actual,
                    hovertemplate="<b>%{fullData.name}</b><br>%{theta}: <b>%{customdata}</b><extra></extra>"
                ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False),
                           angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#ffffff", size=11))),
                legend=dict(font=dict(color="white"), orientation="h", y=1.2, x=0.5, xanchor="center"),
                height=450, margin=dict(l=50, r=50, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Performance Matrix with formatting
    st.divider()
    st.subheader("📊 Detailed Stats Breakdown")
    final_df = pd.concat([pd.DataFrame([target_data]), st.session_state.results]).reset_index(drop=True)
    st.dataframe(
        final_df[['name', 'club_name', 'price', 'age', 'height_in_cm', 'minutes_played'] + [f for f in selected_factors if f not in ['height_in_cm', 'minutes_played']]].style.format({
            'price': lambda x: format_pretty(x, currency=True), 'height_in_cm': '{:.1f}', 
            'goals_per_90': '{:.2f}', 'assists_per_90': '{:.2f}', 'minutes_played': lambda x: format_pretty(x)
        }), use_container_width=True, hide_index=True
    )