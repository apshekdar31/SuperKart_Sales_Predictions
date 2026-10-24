import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initializing Flask app with a name
superkart_app = Flask("SuperKart Sales Predictor")

# Load the trained sales prediction model
model = joblib.load("superkart_rf_model.joblib")

# Define a route for the home page
@superkart_app.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API"

# Define an endpoint to predict churn for a single customer
@superkart_app.post('/v1/predictsales')
def predict_sales():
    # Get JSON data from the request
    product_data = request.get_json()

    # Extract relevant customer features from the input data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_char': product_data['Product_Id_char'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a churn prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': prediction})

# Define an endpoint to predict churn for a batch of customers
@superkart_app.post('/v1/predictsalesbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch sales data and convert raw predictions into a readable format
    predictions = model.predict(input_data).tolist()

    # Create dictionary to map index with predictions
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output_dict

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_app.run(debug=True)
