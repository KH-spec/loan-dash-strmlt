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


# loading the saved models

import pandas as pd
import streamlit as st
import requests
import json
from pandas import json_normalize

#import joblib as jlb
#import plotly.graph_objects as go
#import seaborn as sns
#import shap
#from shap.plots import waterfall
#import matplotlib.pyplot as plt
from PIL import Image

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

    
 
def main():
    #API_URL = "https://streamlit-loan.herokuapp.com/"

    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    
    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
     ['MLflow'])
    #st.set_page_config(page_icon='🧊',layout='centered',initial_sidebar_state='auto')
    
    # Display the title
    st.markdown("""<h2 style='text-align: center; color: black;'>LOAN APP SCORING DASHBOARD</h1>,
    <h2 style='text-align: center; color: gray;'>KHAYREDDINE ROUIBAH</h2>""", unsafe_allow_html=True)
    
    #st.title('LOAN APP SCORING DASHBOARD')
    #st.subheader("KHAYREDDINE ROUIBAH")
    st.sidebar.title("Loan Applicant")
    
    #---------------------------------------------------------------------------------------------
    # Display the LOGO
    img = Image.open("C:/Users/Win/streamlit/logo.png")
    st.sidebar.image(img, width=250)
    
    # Display the loan image
    col1, col2, col3 = st.columns(3)
    img = Image.open("C:/Users/Win/streamlit/loan_.png")
    #st.image(img, width=200)
    with col2:
        st.image(img, width=200)
    st.write("""
    To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
    the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.

    These challenges get more complicated as the count of applications increases that are reviewed by loan officers.
    Human approval requires extensive hour effort to review each application, however, the company will always seek
    cost optimization and improve human productivity. This sometimes causes human error and bias, as itâ€™s not practical
    to digest a large number of applicants considering all the factors involved.""")
    
    def get_list_display_features(f, def_n, key):
        all_feat = f
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choose the features to display:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols
    ###############################################################################
    #                      LIST OF API REQUEST FUNCTIONS
    ###############################################################################
    # Get list of ID (cached)
    @st.cache(suppress_st_warning=True)
    def get_id_list():
        # URL of the sk_id API
        id_api_url = "C:/Users/Win/streamlit/Scoring_credit/id.pkl"
        # Requesting the API and saving the response
        response = requests.get(id_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        id_customers = pd.Series(content['data']).values
        return id_customers
    # Get selected customer's data (cached)
    # local test api : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=165690
   
    #############################################################################
    #                          Selected id
    #############################################################################
    # list of customer's ID's
  
        
    #Load the saved model
    cust_id = jlb.load(open("id.pkl","rb"))    
    #cust_id = get_id_list()
    # Selected customer's ID
    selected_id = st.sidebar.selectbox('Select customer ID from list:', cust_id, key=18)
    st.write('Your selected ID = ', selected_id)
    
    
    
    
    
    
    
    
    
    
    
    
    @st.cache
    def get_selected_cust_data(selected_id):
        # URL of the sk_id API
        data_api_url = MLFLOW_URI + "data_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        x_custom = pd.DataFrame(content['data'])
        # x_cust = json_normalize(content['data'])
        y_customer = (pd.Series(content['y_cust']).rename('TARGET'))
        # y_customer = json_normalize(content['y_cust'].rename('TARGET'))
        return x_custom, y_customer

    @st.cache
    def get_all_cust_data():
        # URL of the sk_id API
        data_api_url = MLFLOW_URI + "all_proc_data_tr/"
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))  #
        x_all_cust = json_normalize(content['X_train'])  # Results contain the required data
        y_all_cust = json_normalize(content['y_train'].rename('TARGET'))  # Results contain the required data
        return x_all_cust, y_all_cust

    # Get score (cached)
    @st.cache
    def get_score_model(selected_id):
        # URL of the sk_id API
        score_api_url = MLFLOW_URI + "scoring_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(score_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the values of "ID" from the content
        score_model = (content['score'])
        threshold = content['thresh']
        return score_model, threshold

    # Get list of shap_values (cached)
    # local test api : http://127.0.0.1:5000/app/shap_val//?SK_ID_CURR=10002
    @st.cache
    def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = MLFLOW_URI + "shap_val/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.DataFrame(content['shap_val_cust'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals

    #############################################
    #############################################
    # Get list of expected values (cached)
    @st.cache
    def values_expect():
        # URL of the sk_id API
        expected_values_api_url = MLFLOW_URI + "exp_val/"
        # Requesting the API and saving the response
        response = requests.get(expected_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        expect_vals = pd.Series(content['data']).values
        return expect_vals

    # Get list of feature names
    @st.cache
    def feat():
        # URL of the sk_id API
        feat_api_url = MLFLOW_URI + "feat/"
        # Requesting the API and saving the response
        response = requests.get(feat_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        features_name = pd.Series(content['data']).values
        return features_name

    # Get the list of feature importances (according to lgbm classification model)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        feat_imp_api_url = MLFLOW_URI + "feat_imp/"
        # Requesting the API and save the response
        response = requests.get(feat_imp_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(selected_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        neight_data_api_url = MLFLOW_URI + "neigh_cust/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
        # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
        # data_all_customers = pd.DataFrame(content['data_all_cust'])
        data_neig = pd.DataFrame(content['data_neigh'])
        target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
        return data_neig, target_neig

    # Get data from 1000 nearest neighbors in train set (cached)
    @st.cache
    def get_data_thousand_neigh(selected_id):
        thousand_neight_data_api_url = MLFLOW_URI + "thousand_neigh/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(thousand_neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
        # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
        # data_all_customers = pd.DataFrame(content['data_all_cust'])
        data_thousand_neig = pd.DataFrame(content['X_thousand_neigh'])
        x_custo = pd.DataFrame(content['x_custom'])
        target_thousand_neig = (pd.Series(content['y_thousand_neigh']).rename('TARGET'))
        return data_thousand_neig, target_thousand_neig, x_custo
    # --------------------------------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
                                 min_value=0., value=3.87, step=1.)

    age_med = st.number_input('Âge médian des maisons dans le secteur',
                              min_value=0., value=28., step=1.)

    nb_piece_med = st.number_input('Nombre moyen de pièces',
                                   min_value=0., value=5., step=1.)

    nb_chambre_moy = st.number_input('Nombre moyen de chambres',
                                     min_value=0., value=1., step=1.)

    taille_pop = st.number_input('Taille de la population dans le secteur',
                                 min_value=0, value=1425, step=100)

    occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
                                     min_value=0., value=3., step=1.)

    latitude = st.number_input('Latitude du secteur',
                               value=35., step=1.)

    longitude = st.number_input('Longitude du secteur',
                                value=-119., step=1.)

    predict_btn = st.button('Prédire')
    if predict_btn:
        data = [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
                 taille_pop, occupation_moy, latitude, longitude]]
        pred = None

        if api_choice == 'MLflow':
            pred = request_prediction(MLFLOW_URI, data)[0] * 100000
            'Le prix médian d\'une habitation est de {:.2f}'.format(pred)


if __name__ == '__main__':
    main()
