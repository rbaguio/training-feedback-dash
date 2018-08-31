import dash
import dash_core_components as dcc
from dash.dependencies import *
import dash_html_components as html
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from textblob import TextBlob
from util.tf_idf import tfidf
from nltk.corpus import stopwords
import squarify
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')
import seaborn as sns



# url = 'https://docs.google.com/spreadsheets/d/e/' +\
#       '2PACX-1vSOCS4BDmbBhGzQcslQzx2iBnFuAjsmCsa' +\
#       '2i0fE3mmliWLidL1hRVFH_g2g61ku_5kHXd5HKDKU' +\
#       'lHd9/pub?gid=1988319498&single=true&output=csv'

# url = 'https://raw.githubusercontent.com/rbaguio/training-feedback-dash/master/sample-data.csv'

# key = '1V2cw0l7l9y-kKpC9d8t8qVOvtUs2PaphPpIE3eRxLgY'
# form_id = '1101984459'
# url = f'https://docs.google.com/spreadsheets/d/{key}/export?gid={form_id}&format=csv'
# df = pd.read_csv(url)
# df.drop(df.columns[-2:], inplace=True, axis=1)

url = 'https://docs.google.com/spreadsheets/d/1onA4xqBUa_uXDQB6qcS5p-dCetGj9Kpgo3D-VKl1XLw/export?gid=1988319498&format=csv'
seed_number = 420

col_names = [
    "participant_name",
    "role",
    "course",
    "instructor-name",
    "instructor-clarity",
    "instructor-brevity",
    "instructor-quality",
    "instructor-enthusiasm",
    "course-content",
    "course-organization",
    "course-amount-learned",
    "course-relevance",
    "comment-most-like",
    "comment-least-like",
    "comment-improvement",
    "net-promoter-score"
]

trainers = [
    'Isaac Reyes',
    'Jay Manahan',
    'Rey Baguio',
]

courses = [
    'Data Storytelling for Business',
    # 'Excel Analytics Ninja',
    'Analytics Talk',
    'Advanced Visualization and Dashboard Design',
    # 'Intro to R for Business Intelligence',
    # 'Intro to Data Science',
]

# Set Graph Elements

base_graph_layout = {
    'xaxis': {
        # 'title': 'X Axis',
        'showticklabels': False,
        'zerolinewidth': 3,
        'zeroline': True,
        'showline': False,
        'showgrid': False,
        'domain': [0.025, 1],
    },
    'yaxis': {
        # 'title': 'Y Axis',
        # 'showticklabels': False,
        'showline': False,
        'showgrid': False,
    },
    'barmode': 'relative',
    # 'showlegend': False,
    'title': 'Hello World',
    'font': {
        'family': 'Karla',
        'size': 14,
        'color': 'white',
    },
    'titlefont': {
        'font-weight': '600',
        'size': 30,
    },
    'plot_bgcolor': '#077cad',
    'paper_bgcolor': '#077cad',
}

graph_styles = {
    'width': '100%',
    'flex': '1 0 100%',
    # 'padding': '30px',
}

table_styles = {
    # 'margin-top': '50px',
    # 'flex': '1 0',
}

treemap_styles = {
}

instructor_graph_layout = dict(base_graph_layout)
instructor_graph_layout['title'] = 'Trainer'

content_graph_layout = dict(base_graph_layout)
content_graph_layout['title'] = 'Content'
content_graph_layout['showlegend'] = False


colors = {
    5: '#023445',
    4: '#416774',
    2: '#ee4a4a',
    1: '#e80d0d',
}

labels_dict = {
    1: 'Very Poor',
    2: 'Poor',
    4: 'Good',
    5: 'Excellent',
}

app = dash.Dash(__name__)
server = app.server

# Create App Layout

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css?family=Karla" rel="stylesheet"> 
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    # html.Div(
    #     html.Img(src='/assets/ds-logo.png', id='logo'),
    #     id='logo-container'
    # ),
    dcc.Interval(id='data-stream', interval=10000, n_intervals=0),
    html.Div([
        html.Div(
            html.Div([
                html.Div([
                    html.Div([
                        html.H2("Net Promoter Score (Ave)"),
                        html.Div(id='net-promoter-score', className='impact-metric'),
                    ], className='impact-metric-container'),
                    html.Div([
                        html.H2('Total Respondents'),
                        html.Div(id='count-respondents', className='impact-metric'),
                    ], className='impact-metric-container'),
                    html.Div(id='forest-data', style={'display': 'none'}),
                ], id="impact-metrics"),
                html.Div([
                    html.H4('Trainer/s'),
                    dcc.Dropdown(
                        id='trainer-dropdown',
                        className='dropdown',
                        options=[{
                            'label': trainer,
                            'value': trainer
                        } for trainer in trainers],
                        value=trainers,
                        multi=True
                    ),
                    html.H4('Course/s'),
                    dcc.Dropdown(
                        id='course-dropdown',
                        className='dropdown',
                        options=[{
                            'label': course,
                            'value': course
                        } for course in courses],
                        value=courses,
                        multi=True
                    ),
                ], id='data-filters'),
            ], id='settings'),
            id="settings-container"
        ),
        html.Div([
            html.H2('TF-IDF', className="feedback-header"),
            html.P('What did you like most about the training?'),
            dcc.Graph(id="treemap", style=treemap_styles)
        ], id="treemap-container"),
    ], id='left-side-container'),
    html.Div([
        html.H2('Feedback', className="feedback-header"),
        html.Div([
            html.Div(id="instructor-betas", style=table_styles),
            dcc.Graph(
                id="instructor-bar-chart",
                style=graph_styles,
            ),
        ], id='instructor-chart-container'),
        html.Div([
            html.Div(id="content-betas", style=table_styles),
            dcc.Graph(
                id="content-bar-chart",
                style=graph_styles,
            )
        ], id="content-chart-container")
    ], id='chart-container', style={
        'display': 'inline-flex',
        'width': '100vw',
        'max-width': '50%',
        'flex-direction': 'column',
        'margin': '30px',
    }),
], style={'display': 'flex'})

filters = [
    Input('trainer-dropdown', 'value'),
    Input('course-dropdown', 'value'),
    Input('data-stream', 'n_intervals'),
]

# Run App


@app.callback(
    Output('net-promoter-score', 'children'),
    filters
)
def render_nps(trainer, course, n_intervals):
    df = pd.read_csv(url)
    df.drop(df.columns[-2:], inplace=True, axis=1)
    print(df)
    df.columns = col_names

    query = (
        df['instructor-name'].isin(trainer)) & (
        df['course'].isin(course)
    )

    filtered_df = df[query]

    avg = filtered_df['net-promoter-score'].mean()
    return html.H2(f'{avg:.1f}')


@app.callback(
    Output('count-respondents', 'children'),
    filters
)
def render_count(trainer, course, n_intervals):
    df = pd.read_csv(url)
    df.drop(df.columns[-2:], inplace=True, axis=1)

    df.columns = col_names

    query = (
        df['instructor-name'].isin(trainer)) & (
        df['course'].isin(course)
    )

    filtered_df = df[query]
    count = len(filtered_df)

    return html.H2(f'{count}')


@app.callback(
    Output('instructor-bar-chart', 'figure'),
    filters
)
def update_instructor_bar_chart(trainer, course, n_intervals):
    df = pd.read_csv(url)
    df.drop(df.columns[-2:], inplace=True, axis=1)

    df.columns = col_names

    query = (
        df['instructor-name'].isin(trainer)) & (
        df['course'].isin(course)
    )

    filtered_df = df[query]

    instructor_bar_columns = [
        'clarity', 'brevity', 'quality', 'enthusiasm',
    ]

    idx = list(range(1, 11))

    instructor_df = pd.DataFrame()

    for col in instructor_bar_columns:
        col_name = f'instructor-{col}'

        df_hist = filtered_df[col_name].value_counts().reindex(
            idx, fill_value=0.00001
        ).reset_index()

        df_hist.rename(
            {f'{col_name}': f'{col_name}-annotation'},
            axis=1,
            inplace=True
        )

        df_hist[f'{col_name}-data'] = df_hist.apply(
            lambda row:
                row[f'{col_name}-annotation']
                if row['index'] > 3 else -row[f'{col_name}-annotation'],
            axis=1
        )

        df_hist.set_index('index', inplace=True)
        df_hist = df_hist.loc[df_hist.index != 3, :]
        instructor_df = pd.concat([instructor_df, df_hist], axis=1)

    annotations = instructor_df.columns[
        instructor_df.columns.str.contains('annotation')
    ]

    instructor_df[annotations] = instructor_df[annotations].replace({
        0.00001: 0
    })

    instructor_traces = []

    for idx, col in zip([4, 5, 2, 1], instructor_bar_columns):
        trace_dict = {}
        trace_dict['y'] = instructor_bar_columns
        trace_dict['x'] = instructor_df.loc[idx, instructor_df.columns.str.contains('data')]
        trace_dict['type'] = 'bar'
        trace_dict['orientation'] = 'h'
        trace_dict['text'] = instructor_df.loc[idx, instructor_df.columns.str.contains('annotation')]
        trace_dict['name'] = labels_dict[idx]
        trace_dict['marker'] = {
            'color': colors[idx],
            # 'line': {
            #     'color': '#FFF',
            #     'width': 2,
            # }
        }
        trace_dict['hoverinfo'] = 'text'

        instructor_traces.append(trace_dict)

    return {
        'data': instructor_traces,
        'layout': instructor_graph_layout,
    }


@app.callback(
    Output('content-bar-chart', 'figure'),
    filters
)
def update_content_bar_chart(trainer, course, n_intervals):
    df = pd.read_csv(url)
    df.drop(df.columns[-2:], inplace=True, axis=1)

    df.columns = col_names

    query = (
        df['instructor-name'].isin(trainer)) & (
        df['course'].isin(course)
    )

    filtered_df = df[query]

    content_bar_columns = [
        'content', 'organization', 'amount-learned', 'relevance'
    ]

    content_bar_columns_annotation = [
        'content', 'organization', 'amount <br> learned', 'relevance'
    ]

    idx = list(range(1, 6))
    content_df = pd.DataFrame()

    for col in content_bar_columns:
        col_name = f'course-{col}'

        df_hist = filtered_df[col_name].value_counts().reindex(
            idx, fill_value=0.00001
        ).reset_index()

        df_hist.rename(
            {f'{col_name}': f'{col_name}-annotation'},
            axis=1,
            inplace=True
        )

        df_hist[f'{col_name}-data'] = df_hist.apply(
            lambda row:
                row[f'{col_name}-annotation']
                if row['index'] > 3 else -row[f'{col_name}-annotation'],
            axis=1
        )

        df_hist.set_index('index', inplace=True)
        df_hist = df_hist.loc[df_hist.index != 3, :]
        content_df = pd.concat([content_df, df_hist], axis=1)

    annotations = content_df.columns[
        content_df.columns.str.contains('annotation')
    ]

    content_df[annotations] = content_df[annotations].replace({0.00001: 0})

    content_traces = []

    for idx, col in zip([4, 5, 2, 1], content_bar_columns):
        trace_dict = {}
        trace_dict['y'] = content_bar_columns_annotation
        trace_dict['x'] = content_df.loc[idx, content_df.columns.str.contains('data')]
        trace_dict['type'] = 'bar'
        trace_dict['orientation'] = 'h'
        trace_dict['text'] = content_df.loc[idx, content_df.columns.str.contains('annotation')]
        trace_dict['name'] = labels_dict[idx]
        trace_dict['marker'] = {
            'color': colors[idx],
            # 'line': {
            #     'color': '#FFF',
            #     'width': 2,
            # }
        }
        trace_dict['hoverinfo'] = 'text'

        content_traces.append(trace_dict)

    return {
        'data': content_traces,
        'layout': content_graph_layout,
    }


@app.callback(
    Output('forest-data', 'children'),
    filters
)
def run_regression(trainer, course, n_intervals):
    df = pd.read_csv(url)
    df.drop(df.columns[-2:], inplace=True, axis=1)

    df.columns = col_names

    query = (
        df['instructor-name'].isin(trainer)) & (
        df['course'].isin(course)
    )

    filtered_df = df[query]

    X = filtered_df[df.columns[4:12]]
    y = filtered_df['net-promoter-score']

    forest = ExtraTreesClassifier(
        n_estimators=10,
        max_depth=8,
        max_leaf_nodes=128,
        random_state=seed_number,
    )
    if len(df) > 0:
        forest.fit(X, y)

        features = X.columns

        std = np.std(
            [tree.feature_importances_ for tree in forest.estimators_],
            axis=0
        )

        # Reverse sort the indices

        series = pd.Series({
            feature: score for feature, score in zip(
                features, std
            )
        })

        return series.to_json(date_format='iso', orient='split')
    else:
        return pd.Series().to_json(date_format='iso', orient='split')


@app.callback(
    Output('instructor-betas', 'children'),
    [Input('forest-data', 'children')],
)
def update_beta_instructor(json):
    json_read_df = pd.read_json(json)

    if len(json_read_df) > 0:
        series = json_read_df['data']
        series.index = json_read_df['index']

        filtered_series = series[series.index.str.contains('instructor')]

        # return [html.Tr([html.Th('beta', style=th_styles)])] +\
        #     [html.Tr(html.Td(f'{d:.3f}', style=td_styles)) for d in filtered_series[::-1]]

        # return [html.Tr([html.Th('name'), html.Th('beta')])] +\
        #     [html.Tr([
        #         html.Td(f'{k}'),
        #         html.Td(f'{v:.2f}')]) for k, v in filtered_series[::-1].iteritems()]
        return [html.H5('beta', className='header')] + [html.H5(f'{d:.3f}') for d in filtered_series[::-1]]
    else:
        return html.H5('')


@app.callback(
    Output('content-betas', 'children'),
    [Input('forest-data', 'children')],
)
def update_beta_content(json):
    json_read_df = pd.read_json(json)
    series = json_read_df['data']
    series.index = json_read_df['index']
    if len(json_read_df) > 0:
        filtered_series = series[series.index.str.contains('course')]

        # return [html.Tr([html.Th('beta', style=th_styles)])] +\
        #     [html.Tr(html.Td(f'{d:.3f}', style=td_styles)) for d in filtered_series[::-1]]

        # return [html.Tr([html.Th('name'), html.Th('beta')])] +\
        #     [html.Tr([
        #         html.Td(f'{k}'),
        #         html.Td(f'{v:.2f}')]) for k, v in filtered_series[::-1].iteritems()]

        return [html.H5(f'{d:.3f}') for d in filtered_series[::-1]]
    else:
        return html.H5('')


@app.callback(
    Output('treemap', 'figure'),
    filters
)
def update_treemap(trainer, course, n_intervals):
    df = pd.read_csv(url)
    df.drop(df.columns[-2:], inplace=True, axis=1)

    df.columns = col_names

    query = (
        df['instructor-name'].isin(trainer)) & (
        df['course'].isin(course)
    )

    filtered_df = df[query]

    positive_comments = filtered_df[filtered_df.columns[-4]].tolist()

    blob_list = [
        TextBlob(str(comment)) for comment in positive_comments
    ]

    stop_words = stopwords.words('english')

    keys = {}
    for blob in blob_list:
        for word in blob.words:
            lower_word = word.lower()
            keys[lower_word] = tfidf(word, blob, blob_list)
    if 'nan' in keys.keys():
        del keys['nan']

    if 'none' in keys.keys():
        del keys['none']

    keywords = {}
    for key in keys:
        lower_keyword = key.lower()
        if lower_keyword not in stop_words:
            keywords[key] = keys[key]
        else:
            continue

    x = 0
    y = 0
    width = 100
    height = 100
    srs = pd.Series(keywords)
    sorted_srs = srs.sort_values(ascending=False)
    kw = sorted_srs.index.tolist()
    values = sorted_srs.tolist()

    # values = [500, 433, 78, 25, 25, 7]

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)
    color_tuples = sns.color_palette(None, len(values))

    color_hex = [
        'rgb({},{},{})'.format(*tuple(
            map(lambda val: int(val * 255), tup)
        )) for tup in color_tuples
    ]

    color_brewer = color_hex

    # color_brewer = color_temp[:5] + ['#a9a9a9' for i in range(len(values) - 5)]

    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append(dict(
            type='rect',
            x0=r['x'],
            y0=r['y'],
            x1=r['x'] + r['dx'],
            y1=r['y'] + r['dy'],
            line=dict(width=2),
            fillcolor=color_brewer[counter]
        ))

        annotations.append(dict(
            x=r['x'] + (r['dx'] / 2),
            y=r['y'] + (r['dy'] / 2),
            text=f'{values[counter]:.2f}<br>({kw[counter]})',
            showarrow=False
        ))

        counter += 1
        if counter >= len(color_brewer):
            counter = 0

    treemap_trace = go.Scatter(
        x=[r['x'] + (r['dx'] / 2) for r in rects],
        y=[r['y'] + (r['dy'] / 2) for r in rects],
        text=[str(v) for v in values],
        mode='text',
    )

    treemap_layout = {
        'height': 700,
        'width': 700,
        'xaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False
        },
        'yaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False
        },
        'shapes': shapes,
        'annotations': annotations,
        'hovermode': 'text',
        'plot_bgcolor': '#077cad',
        'paper_bgcolor': '#077cad',
    }

    return {
        'data': [treemap_trace],
        'layout': treemap_layout,
    }


if __name__ == '__main__':
    app.run_server(debug=True)
