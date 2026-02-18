import streamlit as st
import pickle
import pandas as pd

pipe = pickle.load(open('ipl_model.pkl','rb'))

st.title("IPL Win Probability Predictor")

teams = [
'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
'Rajasthan Royals','Delhi Capitals'
]

cities = ['Hyderabad','Mumbai','Bangalore','Kolkata','Delhi','Chennai','Jaipur','Mohali']

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
city = st.selectbox("City", cities)

runs_left = st.number_input("Runs Left")
balls_left = st.number_input("Balls Left")
wickets_left = st.number_input("Wickets Left")
crr = st.number_input("Current Run Rate")
rrr = st.number_input("Required Run Rate")

if st.button("Predict Probability"):
    input_df = pd.DataFrame([[batting_team,bowling_team,city,
                              runs_left,balls_left,wickets_left,crr,rrr]],
        columns=['batting_team','bowling_team','city',
                 'runs_left','balls_left','wickets_left','crr','rrr'])

    result = pipe.predict_proba(input_df)

    st.success(f"{batting_team} Win Probability: {round(result[0][1]*100)}%")
    st.error(f"{bowling_team} Win Probability: {round(result[0][0]*100)}%")
