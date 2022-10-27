# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: siddhardhan
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
from pandas import json_normalize
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shap
from shap.plots import waterfall
import matplotlib.pyplot as plt

# ----------------------------------------------------
# main function
# ----------------------------------------------------

API_URL = "https://streamlit-loan.herokuapp.com/"

# loading the saved models

scoring_credit_model = pickle.load(open('https://github.com/KH-spec/Loan-dashboard-using-Streamlit-on-Heroku/blob/main/scoring_credit_model.sav', 'rb'))






