import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import pickle
import app_util as app_util
import streamlit as st

st.set_page_config(page_title='Team 154: Songs that Resonate Across the Decades', layout="wide", menu_items=None)

st.title('Songs that Resonate Across the Decades')
st.markdown('''
    Songs from the [Million Song Dataset](http://millionsongdataset.com/) are used here to discover "hits" that were ahead of their time.
    The most recent songs from the dataset were used in a Gamma GLM regression to determine what makes a song popular "today".
    Below, you can select a decade to discover songs in that decade that are popular according to the popularity trends of recent songs.
    You can also select a song from the "recent" list to have it highlighted in the graph and be able to identify older or newer songs that are close or similar to the selected song.
    ''')


# load data
df = pd.read_csv('https://objects-us-east-1.dream.io/cdn-dreamhost-e/msd_track_metadata_1M_fromEC2_v2_cleaned.csv')

# split data into decades
decades_dfs = app_util.get_decades_data(df)

# load gamma model(s)
all_models = pickle.load(open('./models/gamma_all-decades_v1.pkl','rb'))

# select model trained from the "recent" tracks
model = all_models[-1]

# 2 cols for the inputs
col1, col2, col3 = st.columns(3)

# display decade selector
decade_selection = [dkey for dkey in decades_dfs][:-1]
decade_selection.sort(reverse=True)
selected_decade = col1.selectbox('Select a decade', decade_selection)

# display top track count selector
number_of_top_tracks = col2.selectbox('Number of top tracks', [10, 100, 500, 1000, 5000, 10000, 50000, 100000], 4)

# get top tracks
recent_top_tracks = app_util.get_top_tracks(model, decades_dfs['recent']['data'], number_of_top_tracks)
decade_top_tracks = app_util.get_top_tracks(model, decades_dfs[selected_decade]['data'], number_of_top_tracks)

# display song selector
song_selections = recent_top_tracks['title'] + ' - ' + recent_top_tracks['artist_name']
song_selections = [''] + song_selections.to_numpy().tolist()
selected_song = col3.selectbox('Select a top "recent" song', song_selections)

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

if (selected_song != ''):
    selected_song_index = song_selections.index(selected_song)
    selected_song_data = pca_data.iloc[selected_song_index-1:selected_song_index]

    # highlight selected song
    fig.add_traces(px.scatter(selected_song_data, x="pca1", y="pca2").update_traces(marker_size=25, marker_color="#FFCD00", opacity=0.75).data)

    # get 5 closest tracks
    closest_tracks = app_util.get_closest_tracks(pca_data[pca_data['decade'] == selected_decade].copy(), selected_song_data, k=5)

    # add lines to closest tracks
    for index, row in closest_tracks.iterrows():
        # Extract x and y coordinates of selected and closest points
        #x1, y1 = pca_data.iloc[selected_song_index][['pca1', 'pca2']]
        x1, y1 = pca_data.iloc[selected_song_index-1][['pca1', 'pca2']]
        x2, y2 = row[['pca1', 'pca2']]

        # Create line trace with desired color and opacity
        line = go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(color='red')  # Customize line properties
            , hoverinfo='skip'
        )
        fig.add_trace(line)

st.plotly_chart(fig, use_container_width=True)

# list top predicted songs
if (selected_song != ''):
    st.write(f'Top predicted songs from the selected decade ({selected_decade}) that are similar to the selected track ({selected_song}):')
    top_tracks = closest_tracks.sort_values('pred', ascending=False)
    st.dataframe(top_tracks[['year','title','artist_name','release']].style.format(thousands=""), hide_index=True, use_container_width=True)
else:
    st.write(f'Top predicted songs from the selected decade ({selected_decade}):')
    st.dataframe(decade_top_tracks[['year','title','artist_name','release']].style.format(thousands=""), hide_index=True, use_container_width=True)
