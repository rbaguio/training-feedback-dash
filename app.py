import dash
import dash_core_components as dcc
from dash.dependencies import *
import dash_html_components as html
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
# from util import random_forest

# url = 'https://docs.google.com/spreadsheets/d/e/' +\
#       '2PACX-1vSOCS4BDmbBhGzQcslQzx2iBnFuAjsmCsa' +\
#       '2i0fE3mmliWLidL1hRVFH_g2g61ku_5kHXd5HKDKU' +\
#       'lHd9/pub?gid=1988319498&single=true&output=csv'

# url = 'https://raw.githubusercontent.com/rbaguio/training-feedback-dash/master/sample-data.csv'

url = 'https://docs.google.com/spreadsheets/d/1onA4xqBUa_uXDQB6qcS5p-dCetGj9Kpgo3D-VKl1XLw/export?gid=1988319498&format=csv'
# df = pd.read_csv(url)
# df.drop(df.columns[-2:], inplace=True, axis=1)

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
    'Analytics Talk',
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
    },
    'yaxis': {
        # 'title': 'Y Axis',
        # 'showticklabels': False,
        'showline': False,
        'showgrid': False,
    },
    'barmode': 'relative',
    'showlegend': False,
    'title': 'Hello World',
    'font': {
        'family': 'Raleway',
        'size': 14,
    }
}

graph_styles = {
    'width': '40%',
    'flex': '1 0 50%',
}


instructor_graph_layout = dict(base_graph_layout)
instructor_graph_layout['title'] = 'Trainer Feedback'

content_graph_layout = dict(base_graph_layout)
content_graph_layout['title'] = 'Content Feedback'


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

app.layout = html.Div([
    html.H1(
        children='DataSeer Feedback Dashboard',
        style={
            'text-align': 'center',
        }
    ),
    dcc.Interval(id='data-stream', interval=1000, n_intervals=0),
    html.Div([
        html.Div([
            html.H2(
                children='Net Promoter Score (Ave)'
            ),
            html.Div(id='net-promoter-score'),
        ]),
        html.Div([
            html.Div(
                html.H2(
                    children='Total Respondents'
                ),
            ),
            html.Div(id='count-respondents'),
        ]),
    ],
        id="impact-metrics"),
    html.Div([
        html.H4('Trainer/s'),
        dcc.Dropdown(
            id='trainer-dropdown',
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
            options=[{
                'label': course,
                'value': course
            } for course in courses],
            value=courses,
            multi=True
        ),
    ]),
    html.Div(
        children=[
            html.Table(id="instructor-betas"),
            dcc.Graph(
                id="instructor-bar-chart",
                style=graph_styles,
            ),
            dcc.Graph(
                id="content-bar-chart",
                style=graph_styles,
            ),
        ],
        id='chart-container',
        style={
            'width': '100%',
            'display': 'flex',
        }
    )
],
    style={
        'font-family': 'Raleway'
})

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
    Output('instructor-betas', 'children'),
    filters
)
def generate_table_rows(trainer, course, n_intervals):
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

    instructor_series = series[series.index.str.contains('instructor')]

    return [html.Tr([html.Th('beta')])] + [
        html.Tr(f'{d:.2f}') for d in instructor_series
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
