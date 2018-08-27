import dash
import dash_core_components as dcc
from dash.dependencies import Output, Input
import dash_html_components as html
import pandas as pd

# df = pd.read_csv(
#     'https://docs.google.com/spreadsheets/d/e/'
#     '2PACX-1vSOCS4BDmbBhGzQcslQzx2iBnFuAjsmCsa'
#     '2i0fE3mmliWLidL1hRVFH_g2g61ku_5kHXd5HKDKU'
#     'lHd9/pub?gid=1988319498&single=true&output=csv'
# )

# df.drop(df.columns[-2:], inplace=True, axis=1)

url = 'https://raw.githubusercontent.com/rbaguio/training-feedback-dash/master/sample-data.csv'
df = pd.read_csv(url)

df.columns = [
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


# Instructor DataFrame

instructor_bar_columns = [
    'clarity', 'brevity', 'quality', 'enthusiasm',
]

idx = list(range(1, 6))

instructor_df = pd.DataFrame()

for col in instructor_bar_columns:
    col_name = f'instructor-{col}'

    df_hist = df[col_name].value_counts().reindex(
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


instructor_df[instructor_df.columns[instructor_df.columns.str.contains(
    'annotation'
)]] = instructor_df[instructor_df.columns[instructor_df.columns.str.contains(
    'annotation'
)]].replace({0.00001: 0})


# Content DataFrame

content_bar_columns = [
    'content', 'organization', 'amount-learned', 'relevance'
]

content_bar_columns_annotation = [
    'content', 'organization', 'amount <br> learned', 'relevance'
]

content_df = pd.DataFrame()

for col in content_bar_columns:
    col_name = f'course-{col}'

    df_hist = df[col_name].value_counts().reindex(
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


content_df[content_df.columns[content_df.columns.str.contains(
    'annotation'
)]] = content_df[content_df.columns[content_df.columns.str.contains(
    'annotation'
)]].replace({0.00001: 0})


nps = df['net-promoter-score'].mean()

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
    # 'showlegend': False,
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

# Create App Traces

app = dash.Dash()


instructor_traces = []
content_traces = []


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


# Create App Layout

app.layout = html.Div([
    html.H1(
        children='DataSeer Feedback Dashboard',
        style={
            'text-align': 'center',
        }
    ),
    html.Div(
        html.H2(
            children='Net Promoter Score (Ave)'
        ),
    ),
    html.Div(id='net-promoter-score'),
    html.Div([
        html.H4('Trainer/s'),
        dcc.Dropdown(
            id='trainer-dropdown',
            options=[{
                'label': trainer,
                'value': trainer
            } for trainer in df['instructor-name'].unique()],
            value=df['instructor-name'].unique().tolist(),
            multi=True
        ),
        html.H4('Course/s'),
        dcc.Dropdown(
            id='course-dropdown',
            options=[{
                'label': course,
                'value': course
            } for course in df['course'].unique()],
            value=df['course'].unique().tolist(),
            multi=True
        ),
    ]),
    html.Div(
        children=[
            dcc.Graph(
                id="instructor-bar-chart",
                figure={
                    'data': instructor_traces,
                    'layout': instructor_graph_layout,
                },
                style=graph_styles,
            ),
            dcc.Graph(
                id="content-bar-chart",
                figure={
                    'data': content_traces,
                    'layout': content_graph_layout,
                },
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


# Run App

# @app.callback(
#     Output('')
#     [Input('')]
# )
# def update_graphs():
#     return df['net-promoter-score'].mean()


if __name__ == '__main__':
    app.run_server()
