import streamlit as st
import pydeck as pdk
import pandas as pd

st.title("Minimal PyDeck Test")

df = pd.DataFrame({
    'lat': [12.9716, 12.9720, 12.9710],
    'lon': [77.5946, 77.5950, 77.5940],
    'name': ['Point A', 'Point B', 'Point C']
})

view_state = pdk.ViewState(
    latitude=12.9716,
    longitude=77.5946,
    zoom=14
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_color="[255, 0, 0, 200]",
    get_radius=50,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state
))
st.write("If you see a map above, WebGL and PyDeck are working.")
