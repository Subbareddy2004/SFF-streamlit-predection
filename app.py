import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open('rf_model_updated.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

rf_model, encoders = load_model_and_encoders()

# Load your dataset to get unique values for dropdowns
@st.cache_data
def load_data():
    return pd.read_csv('updated_match_details_with_impact_scores.csv')

df = load_data()

def predict_outcome(batting_team, bowling_team, venue, target, current_score, wickets_left, balls_left):
    runs_left = target - current_score
    balls_consumed = 120 - balls_left
    crr = current_score / (balls_consumed / 6) if balls_consumed > 0 else 0
    rrr = runs_left / (balls_left / 6) if balls_left > 0 else float('inf')

    batting_impact = df[df['batting_team'] == batting_team]['Batting_Team_Impact_Score'].mean()
    bowling_impact = df[df['bowling_team'] == bowling_team]['Bowling_Team_Impact_Score'].mean()

    input_data = pd.DataFrame({
        'batting_team': [encoders['batting_team'].transform([batting_team])[0]],
        'bowling_team': [encoders['bowling_team'].transform([bowling_team])[0]],
        'city': [encoders['city'].transform([venue])[0]],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_remaining': [wickets_left],
        'total_run_x': [target],
        'crr': [crr],
        'rrr': [rrr],
        'Batting_Team_Impact_Score': [batting_impact],
        'Bowling_Team_Impact_Score': [bowling_impact]
    })

    # Predict probabilities
    probability = rf_model.predict_proba(input_data)[0]
    return probability[1], probability[0]  # Assuming 1 is for batting team win, 0 for bowling team win

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PredictionInput(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    target: int
    score: int
    balls_left: int
    wickets: int

@app.post("/predict")
async def predict(input: PredictionInput):
    try:
        batting_prob, bowling_prob = predict_outcome(
            input.batting_team, input.bowling_team, input.city,
            input.target, input.score, input.wickets, input.balls_left
        )
        return {
            "batting_team": input.batting_team,
            "bowling_team": input.bowling_team,
            "batting_prob": float(batting_prob),
            "bowling_prob": float(bowling_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit app
def streamlit_app():
    st.title('Cricket Match Outcome Predictor')

    batting_team = st.selectbox('Select Batting Team', df['batting_team'].unique())
    bowling_team = st.selectbox('Select Bowling Team', df['bowling_team'].unique())
    venue = st.selectbox('Select Venue', df['city'].unique())
    target = st.number_input('Target Score', min_value=1, value=150)
    current_score = st.number_input('Current Score', min_value=0, max_value=target-1, value=0)
    wickets_left = st.number_input('Wickets Left', min_value=0, max_value=10, value=10)
    balls_left = st.number_input('Balls Left', min_value=0, max_value=120, value=120)

    if st.button('Predict Outcome'):
        batting_prob, bowling_prob = predict_outcome(
            batting_team, bowling_team, venue, target, current_score, wickets_left, balls_left
        )
        
        st.write(f"{batting_team} winning probability: {batting_prob:.2%}")
        st.write(f"{bowling_team} winning probability: {bowling_prob:.2%}")

        # Visualization with transparent background
        fig = go.Figure(data=[go.Pie(labels=[batting_team, bowling_team], values=[batting_prob, bowling_prob])])
        fig.update_layout(
            title='Win Probability',
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
        )
        st.plotly_chart(fig)

        # Display impact scores
        batting_impact = df[df['batting_team'] == batting_team]['Batting_Team_Impact_Score'].mean()
        bowling_impact = df[df['bowling_team'] == bowling_team]['Bowling_Team_Impact_Score'].mean()
        st.write(f"{batting_team} Impact Score: {batting_impact:.2f}")
        st.write(f"{bowling_team} Impact Score: {bowling_impact:.2f}")

# Combine Streamlit and FastAPI
server = Starlette(debug=True, routes=[
    Route("/", streamlit_app),
    Mount("/api", app)
])

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8888)