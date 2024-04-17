import numpy as np

x_attributes = ['tempo','loudness','song_key','song_mode','time_signature']

def get_year_selections(df):
    years = df['year'].unique()
    years.sort()
    years = years.tolist()
    years.insert(0, 'All')
    return years

def get_decades_data(df):
    decades = {
        1940: { 'name': '40s', 'start': 1940, 'end': 1950, 'data': [] },
        1950: { 'name': '50s', 'start': 1950, 'end': 1960, 'data': [] },
        1960: { 'name': '60s', 'start': 1960, 'end': 1970, 'data': [] },
        1970: { 'name': '70s', 'start': 1970, 'end': 1980, 'data': [] },
        1980: { 'name': '80s', 'start': 1980, 'end': 1990, 'data': [] },
        1990: { 'name': '90s', 'start': 1990, 'end': 2000, 'data': [] },
        2000: { 'name': '2000s', 'start': 2000, 'end': 2006, 'data': [] },
        'recent': { 'name': 'recent', 'start': 2006, 'end': 2011, 'data': [] }
    }

    for decade_key in decades:
        decade = decades[decade_key]
        data = df[(df['year'] >= decade['start']) & (df['year'] < decade['end'])].copy()
        data.loc[:, 'decade'] = decade_key
        decade['data'] = data

    return decades

def get_top_tracks(model, data, n = 10):
    result = data.copy()
    result.loc[:,'pred'] = model.predict(data[x_attributes])
    top_indices = np.argsort(result['pred'], axis=0)[-n:][::-1]
    return result.iloc[top_indices]
