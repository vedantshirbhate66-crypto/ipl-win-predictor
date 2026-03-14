import streamlit as st
import pandas as pd
import pickle

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏏 IPL Win Probability Predictor",
    page_icon="🏏",
    layout="centered",
)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("pipe.pkl", "rb") as f:
        return pickle.load(f)

pipe = load_model()

# ── Constants ──────────────────────────────────────────────────────────────────
TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

CITIES = [
    "Abu Dhabi", "Ahmedabad", "Bangalore", "Bengaluru", "Bloemfontein",
    "Cape Town", "Centurion", "Chandigarh", "Chennai", "Cuttack",
    "Delhi", "Dharamsala", "Dubai", "East London", "Hyderabad",
    "Indore", "Jaipur", "Johannesburg", "Kanpur", "Kimberley",
    "Kolkata", "Lucknow", "Mumbai", "Navi Mumbai", "Pune",
    "Raipur", "Rajkot", "Ranchi", "Sharjah", "Visakhapatnam",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏏 IPL Win Probability Predictor")
st.markdown("Predict the **live win probability** for the chasing team in a T20 IPL match.")
st.divider()

# Team & Venue
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team (Chasing)", TEAMS, index=0)
with col2:
    bowling_options = [t for t in TEAMS if t != batting_team]
    bowling_team = st.selectbox("🎯 Bowling Team (Defending)", bowling_options, index=0)

city = st.selectbox("📍 Match City", sorted(CITIES))

st.divider()

# Match Situation
st.subheader("📊 Current Match Situation")
col3, col4, col5 = st.columns(3)
with col3:
    runs_target = st.number_input("🎯 Target Score", min_value=1, max_value=300, value=180)
with col4:
    runs_left = st.number_input("🏃 Runs Left", min_value=0, max_value=300, value=60)
with col5:
    wickets_remaining = st.number_input("❌ Wickets Remaining", min_value=0, max_value=10, value=6)

col6, col7 = st.columns(2)
with col6:
    balls_left = st.number_input("🎳 Balls Left", min_value=1, max_value=120, value=36)
with col7:
    # Auto-calculate run rates
    overs_bowled = (120 - balls_left) / 6
    overs_left = balls_left / 6
    runs_scored = runs_target - runs_left

    crr = round(runs_scored / overs_bowled, 2) if overs_bowled > 0 else 0.0
    rrr = round(runs_left / overs_left, 2) if overs_left > 0 else 0.0

    st.metric("📈 Current Run Rate", crr)

st.metric("⚡ Required Run Rate", rrr)

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Win Probability", use_container_width=True, type="primary"):

    # Basic validation
    if batting_team == bowling_team:
        st.error("Batting team and Bowling team cannot be the same!")
    elif runs_left < 0 or runs_left > runs_target:
        st.error("Runs Left must be between 0 and the Target Score.")
    else:
        input_df = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [city],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets_remaining": [wickets_remaining],
            "runs_target": [runs_target],
            "crr": [crr],
            "rrr": [rrr],
        })

        try:
            proba = pipe.predict_proba(input_df)[0]
            batting_win = round(proba[1] * 100, 1)
            bowling_win = round(proba[0] * 100, 1)

            st.subheader("🎯 Prediction Results")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label=f"🏏 {batting_team}",
                    value=f"{batting_win}%",
                    help="Win probability for the chasing team",
                )
                st.progress(int(batting_win))

            with col_b:
                st.metric(
                    label=f"🎯 {bowling_team}",
                    value=f"{bowling_win}%",
                    help="Win probability for the defending team",
                )
                st.progress(int(bowling_win))

            # Summary verdict
            st.divider()
            if batting_win >= 70:
                st.success(f"✅ **{batting_team}** is in a strong position to win!")
            elif batting_win >= 50:
                st.info(f"⚖️ It's a close game! **{batting_team}** has a slight edge.")
            elif batting_win >= 30:
                st.warning(f"⚠️ **{bowling_team}** is defending well. It'll be a tough chase!")
            else:
                st.error(f"❌ **{batting_team}** is under severe pressure. **{bowling_team}** looks dominant!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "Built with ❤️ using Streamlit · IPL Data 2008–2025 · Logistic Regression Model"
    "</div>",
    unsafe_allow_html=True,
)
