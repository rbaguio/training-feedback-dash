import pandas as pd
pd.set_option('display.max_rows', None)

# df = pd.read_csv(
#     'https://docs.google.com/spreadsheets/d/e/'
#     '2PACX-1vSOCS4BDmbBhGzQcslQzx2iBnFuAjsmCsa'
#     '2i0fE3mmliWLidL1hRVFH_g2g61ku_5kHXd5HKDKU'
#     'lHd9/pub?gid=1988319498&single=true&output=csv'
# )
# df.drop(df.columns[-2:], axis=1, inplace=True)


df = pd.read_csv('sample-data.csv')
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
