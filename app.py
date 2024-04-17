import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import app_util as app_util
import pickle

# load data
df = pd.read_csv('https://objects-us-east-1.dream.io/cdn-dreamhost-e/msd_track_metadata_1M_fromEC2_v2_cleaned.csv')

# split data into decades
decades_dfs = app_util.get_decades_data(df)

# load gamma model(s)
x_attributes = ['tempo','loudness','song_key','song_mode','time_signature']
all_models = pickle.load(open('./models/gamma_all-decades_v1.pkl','rb'))

# select model trained from the "recent" tracks
model = all_models[-1]

# get top tracks from "recent" tracks
recent_top_tracks = app_util.get_top_tracks(model, decades_dfs['recent']['data'], 100)

# select a decade
decade_selection = [dkey for dkey in decades_dfs]
selected_decade = st.selectbox('Select a decade', decade_selection[:-1])

# get top songs from the selected decade (according to the selected model)
decade_top_tracks = app_util.get_top_tracks(model, decades_dfs[selected_decade]['data'], 100)

# display song selector
song_selections = recent_top_tracks['title'] + ' - ' + recent_top_tracks['artist_name']
song_selections = [''] + song_selections.to_numpy().tolist()
selected_song = st.selectbox('Select a top "recent" song', song_selections)

# combine recent data (data used to create model), with selected decade data
pca_data = pd.concat([recent_top_tracks, decade_top_tracks])

# get the PCAs (adds the 2 columns to the the pca dataframe)
pca = PCA(n_components=2).fit_transform(pca_data[x_attributes])
pca_data['pca1'] = pca[:,0]
pca_data['pca2'] = pca[:,1]
pca_data.head(3)

# plot the pcas
fig = px.scatter(pca_data, x='pca1', y='pca2', color='decade', hover_data=['title','artist_name','year'])

# highlight selected song
if (selected_song != ''):
    song_index = song_selections.index(selected_song)
    fig.add_traces(px.scatter(pca_data.iloc[song_index-1:song_index], x="pca1", y="pca2").update_traces(marker_size=20, marker_color="yellow").data)

st.plotly_chart(fig)

# list top predicted songs from the selected decade
st.write('Top predicted songs from the selected decade:')
st.dataframe(decade_top_tracks[['year','title','artist_name','release']].style.format(thousands=""), hide_index=True)

