# Load librairies
import joblib
import pandas as pd
from flask import Flask, request, jsonify,redirect, flash, make_response
import json
import shap
import pickle as pkl

def data_load():
    # -----------------------------------------------------------------------------------------------
    #                                   loadings
    # -----------------------------------------------------------------------------------------------
    # Model and threshold loading 
    # ---------------------------
    model     = joblib.load(open("scoring_credit_model.pkl","rb")) 
    threshold = joblib.load('threshold_model.pkl',"rb")
    # -----------------------------------------------------------------------------------------------
    # Data loading
    # -------------
    X_test            = pd.read_csv('X_test_sample.csv')
    X_train           = pd.read_csv('X_train_sample.csv')
    y_test            = pd.read_csv('y_test.csv')
    y_train           = pd.read_csv('y_train.csv')
        
    return model, threshold, X_test, X_train, y_test, y_train
  
# -----------------------------------------------------------------------------------------------
#                                   loadings
# -----------------------------------------------------------------------------------------------   
model, threshold, X_test, X_train, y_test, y_train=   data_load()


# instantiate Flask object
app = Flask(__name__)

@app.route("/")
def index():
    return "APP loaded, model and data loaded............"
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------    
# Customers id list 
# test local : http://127.0.0.1:5000/app/id/
@app.route('/app/id/')
def ids_list():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    customers_id_list = pd.Series(list(X_test.index.sort_values()))  # X_test
    # Convert pd.Series to JSON
    customers_id_list_json = json.loads(customers_id_list.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': customers_id_list_json})
                    
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------                    
# Customer selected Id data
# test local : http://127.0.0.1:5000/app/data_cust/?ID=0
@app.route('/app/data_cust/')  # ==> OK
def selected_cust_data():  # selected_id
    selected_id_customer = int(request.args.get('ID'))
    x_cust = X_test.loc[selected_id_customer: selected_id_customer]  # X_test
    y_cust = y_test.loc[selected_id_customer: selected_id_customer] # y_test
    # Convert pd.Series to JSO
    data_x_json = json.loads(x_cust.to_json())
    y_cust_json = json.loads(y_cust.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'y_cust': y_cust_json,
                    'data': data_x_json})    
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------
# list of features
# test local : http://127.0.0.1:5000/app/feat/
@app.route('/app/feat/')
def features():
    feat = X_train.columns
    f = pd.Series(feat)
    # Convert pd.Series to JSON
    feat_json = json.loads(f.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': feat_json})                    
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------                  
# return json object of feature importance (lgbm attribute)
# Test local : http://127.0.0.1:5000/app/feat_imp/
@app.route('/app/feat_imp/')
def send_feat_imp():
    feat_imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    # Convert pd.Series to JSON
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
                    'data': feat_imp_json})                    
                    
# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------                  
# answer when asking for score and decision about one customer
# Test local : http://127.0.0.1:5000/app/scoring_cust/?ID=0
@app.route('/app/scoring_cust/')  # == > OK
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    customer_id = int(request.args.get('ID'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[customer_id:customer_id] # X_test
    # X_cust = X_cust.drop(["SK_ID_CURR"], axis=1)
    # Compute the score of the customer (using the whole pipeline)
    score_cust = model.predict_proba(X_cust)[:, 1][0]
    # Return score
    return jsonify({'status': 'ok',
                    'ID': customer_id,
                    'score': score_cust,
                    'thresh': threshold,})

# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------        
# return all data of testing set when requested
# Test local : http://127.0.0.1:5000/app/all_data/
@app.route('/app/all_data/')  # ==> OK
def all_data():
    # get all data from X_train, X_test and y_train data
    # and convert the data to JSON
    X_test_json = json.loads(X_test.to_json())
    y_test_json = json.loads(y_test.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'X_test': X_test_json,
                    'y_test': y_test_json
                    })
         
## -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------        
# return all data of testing set when requested
# Test local : http://127.0.0.1:5000/app/data_train/
@app.route('/app/data_train/')  
def data_train():
    # get all data from X_train, X_test and y_train data
    # and convert the data to JSON
    X_train_json = json.loads(X_train.to_json())
    y_train_json = json.loads(y_train.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'X_train': X_train_json,
                    'y_train': y_train_json
                    })     
         


if __name__ == "__main__":
    app.run()