import streamlit as st
import pickle
import pandas as pd

# Page Config
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        text-align: center;
        color: #00FFAA;
    }
    .stButton>button {
        background-color: #00FFAA;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
pipe = pickle.load(open('ipl_model.pkl','rb'))

st.title("🏏 IPL Win Probability Predictor")

teams = ['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
         'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
         'Rajasthan Royals','Delhi Capitals']

cities = ['Hyderabad','Bangalore','Mumbai','Chennai','Delhi','Kolkata','Jaipur']

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)
    city = st.selectbox("City", cities)

with col2:
    runs_left = st.number_input("Runs Left")
    balls_left = st.number_input("Balls Left")
    wickets_left = st.number_input("Wickets Left")
    total_runs = st.number_input("Target")

if st.button("Predict Win Probability 🚀"):

    if balls_left == 0:
        st.error("Balls left cannot be zero!")
    else:
        cr = (total_runs - runs_left) / (120 - balls_left)
        rrr = runs_left / (balls_left / 6)

        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets_left':[wickets_left],
            'crr':[cr],
            'rrr':[rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.markdown("## 🎯 Match Prediction")

        st.progress(int(win*100))

        st.success(f"🔥 Winning Probability: {round(win*100,2)}%")
        st.error(f"💀 Losing Probability: {round(loss*100,2)}%")
