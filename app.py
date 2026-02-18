import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
.big-font {
    font-size:40px !important;
    font-weight:700;
}
.stButton>button {
    background: linear-gradient(90deg,#ff4b2b,#ff416c);
    color:white;
    border:none;
    padding:10px 20px;
    border-radius:8px;
}
.metric-card {
    background: #1c1f26;
    padding:20px;
    border-radius:15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
pipe = pickle.load(open('ipl_model.pkl','rb'))

# ---------------- TITLE ----------------
st.markdown('<p class="big-font">🏏 IPL Win Probability Predictor</p>', unsafe_allow_html=True)
st.write("Predict live match win probability using Machine Learning")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab',
         'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Mumbai', 'Bangalore', 'Kolkata',
          'Delhi', 'Chennai', 'Jaipur', 'Pune']

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)
    city = st.selectbox("City", cities)

with col2:
    target = st.number_input("Target", min_value=1)
    score = st.number_input("Current Score", min_value=0)
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Predict Probability 🚀"):

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets

    crr = score / overs if overs != 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_left':[wickets_left],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🏆 Batting Team Win %", f"{round(win*100)} %")
        st.progress(int(win*100))
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🔥 Bowling Team Win %", f"{round(loss*100)} %")
        st.progress(int(loss*100))
        st.markdown("</div>", unsafe_allow_html=True)

    st.balloons()

