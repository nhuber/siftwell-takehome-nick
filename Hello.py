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

from urllib.error import URLError

import altair as alt
import pandas as pd

import streamlit as st
from streamlit.hello.utils import show_code

import pickle
import lzma

import sklearn

def model_explorer():
    st.set_page_config(page_title="Siftwell Take-Home Model Explorer", page_icon="📊")
    st.markdown("# Siftwell Take-Home Model Explorer")
    st.write(
        """This page allows the user to interact with the champion models to predict `total_future_cost` and `treatment__mental_health`, respectively.
    """
    )
    
    @st.cache_data

    regr_best = pickle.load(lzma.open('regr_best.xz'))
    rf_cf_best = pickle.load(lzma.open('rf_cf_best.xz'))
    df_train_prepared = pickle.load(lzma.open('df_train_prepared.xz'))
    full_pipeline = pickle.load(lzma.open('full_pipeline.xz'))

    st.slider('Slide me', min_value=0, max_value=10)

model_explorer()

show_code(model_explorer)
