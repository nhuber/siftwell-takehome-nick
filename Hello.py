# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

import pickle
import lzma

import numpy as np

st.set_page_config(page_title="Siftwell Take-Home Model Explorer", page_icon="📊")
st.markdown("# Siftwell Take-Home Model Explorer")
st.write(
"""This page allows the user to interact with the champion models to predict `total_future_cost` and `treatment__mental_health`, respectively.
Every time you change the inputs, the model predictions are automatically updated to build transparency and interpretability into the system.
"""
)

def model_explorer():
    
    regr_best, rf_cf_best, full_pipeline, df_test = load_models_and_data()

    df_test_processed = process_df_test(df_test)

    prediction_inputs = []

    for col in df_test_processed.columns:
        if (type(df_test_processed[col].iloc[0]) is str):
            prediction_inputs.append(df_test_processed[col].mode().iloc[0])
        else:
            prediction_inputs.append(df_test_processed[col].mean())

    row_to_predict = pd.DataFrame(columns = df_test_processed.columns)
    row_to_predict.loc[0] = prediction_inputs

    col1, col2, col3 = st.columns(3)

    with col1:
        for col in df_test_processed.columns:
            if (type(df_test_processed[col].iloc[0]) is str and len(df_test_processed[col].unique()) > 1):
                row_to_predict[col] = st.select_slider("`" + col + "` variable", options=df_test_processed[col].unique(), value=df_test_processed[col].mode().iloc[0])

    with col2:
        for col in df_test_processed.columns:
            if (type(df_test_processed[col].iloc[0]) is not str and (df_test_processed[col].max() - df_test_processed[col].min() != 0)):
                row_to_predict[col] = st.slider("`" + col + "` variable", min_value=float(df_test_processed[col].min()), max_value=float(df_test_processed[col].max()), value=float(df_test_processed[col].mean()))

    prediction_cost = regr_best.predict(full_pipeline.transform(row_to_predict))
    prediction_mental_health = rf_cf_best.predict_proba(full_pipeline.transform(row_to_predict))

    with col3:
        predicted_cost_st = st.metric(label="Predicted `total_future_cost`:", value=round(prediction_cost[0], 1))
        predicted_mental_health_st = st.metric(label="Predicted `treatment__mental_health`:", value=prediction_mental_health.mean(axis=0)[1])

# load initial models and data
# lzma-compressed pickle files used to be below Github's 100 MB file limit
# results are cached as this step takes ~5-10 seconds
@st.cache_data
def load_models_and_data():
    return pickle.load(lzma.open('regr_best.xz')), pickle.load(lzma.open('rf_cf_best.xz')), pickle.load(lzma.open('full_pipeline.xz')), pd.read_csv("coding_challenge_test_without_labels.csv")

# prepare dataframe to be transformed by `full_pipeline`
# results are cached for performance reasons
@st.cache_data
def process_df_test(df):
    df_processed = df

    if 'line_number' in df_processed.columns:
        del df_processed['line_number']

    df_processed['prediction_date'] = pd.to_datetime(df_processed['prediction_date'], format='%Y%m%d')
    df_processed['prediction_date_day'] = df_processed['prediction_date'].dt.day
    df_processed['prediction_date_month'] = df_processed['prediction_date'].dt.month
    df_processed['prediction_date_year'] = df_processed['prediction_date'].dt.year
    if 'prediction_date' in df_processed.columns:
        del df_processed['prediction_date']

    return df_processed

model_explorer()

show_code(model_explorer)

# change name to publicly recognizable
# mention that increase cost, lower likelihood of mental health treatment