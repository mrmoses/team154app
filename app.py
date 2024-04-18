import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import app_util as app_util
import pickle
import streamlit as st

st.set_page_config(page_title='Team 154: Songs that Resonate Across the Decades', layout="wide", menu_items=None)

# load data
df = pd.read_csv('https://objects-us-east-1.dream.io/cdn-dreamhost-e/msd_track_metadata_1M_fromEC2_v2_cleaned.csv')

# split data into decades
decades_dfs = app_util.get_decades_data(df)

# load gamma model(s)
all_models = pickle.load(open('./models/gamma_all-decades_v1.pkl','rb'))

# select model trained from the "recent" tracks
model = all_models[-1]

# get top tracks from "recent" tracks
recent_top_tracks = app_util.get_top_tracks(model, decades_dfs['recent']['data'], 100)

# 2 cols for the inputs
col1, col2 = st.columns(2)

# display decade selector
decade_selection = [dkey for dkey in decades_dfs][:-1]
decade_selection.sort(reverse=True)
selected_decade = col1.selectbox('Select a decade', decade_selection)

# get top songs from the selected decade (according to the selected model)
decade_top_tracks = app_util.get_top_tracks(model, decades_dfs[selected_decade]['data'], 100)

# display song selector
song_selections = recent_top_tracks['title'] + ' - ' + recent_top_tracks['artist_name']
song_selections = [''] + song_selections.to_numpy().tolist()
selected_song = col2.selectbox('Select a top "recent" song', song_selections)

# combine recent data (data used to create model), with selected decade data
pca_data = pd.concat([recent_top_tracks, decade_top_tracks])

# get the PCAs (adds the 2 columns to the the pca dataframe)
pca = PCA(n_components=2).fit_transform(pca_data[app_util.x_attributes])
pca_data['pca1'] = pca[:,0]
pca_data['pca2'] = pca[:,1]
pca_data.head(3)

# plot the pcas
fig = px.scatter(pca_data, x='pca1', y='pca2', color='decade'
                 , hover_data=['title','artist_name','year']
                 , color_discrete_map=app_util.color_map
                 , height=600)

# highlight selected song
if (selected_song != ''):
    song_index = song_selections.index(selected_song)
    fig.add_traces(px.scatter(pca_data.iloc[song_index-1:song_index], x="pca1", y="pca2").update_traces(marker_size=25, marker_color="#FFCD00", opacity=0.75).data)

st.plotly_chart(fig, use_container_width=True)

# list top predicted songs from the selected decade
st.write('Top predicted songs from the selected decade:')
st.dataframe(decade_top_tracks[['year','title','artist_name','release']].style.format(thousands=""), hide_index=True, use_container_width=True)

