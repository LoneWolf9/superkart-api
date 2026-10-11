import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
superkart_api = Flask('SuperKart')

# Load the trained SuperKart model
model = joblib.load('superkart_model.joblib')

# Define a route for the homepage
@superkart_api.get('/')
def home():
  return 'Welcome to the SuperKart!'

# Define an endpoint to predict sales for single product
@superkart_api.post('/v1/predict')
def predict_sales():

  # Get JSON data from request
  data2 = request.get_json()

  # Extract relevant features from input data
  sample = {
      'Product_Weight': data2['Product_Weight'],
      'Product_Sugar_Content': data2['Product_Sugar_Content'],
      'Product_Allocated_Area': data2['Product_Allocated_Area'],
      'Product_MRP': data2['Product_MRP'],
      'Store_Size': data2['Store_Size'],
      'Store_Location_City_Type': data2['Store_Location_City_Type'],
      'Store_Type': data2['Store_Type'],
      'Product_Id_Char': data2['Product_Id_Char'],
      'Store_Age_Years': data2['Store_Age_Years'],
      'Product_Type_Category': data2['Product_Type_Category']
  }

  # Convert the extracted data into a dataframe
  input_data = pd.DataFrame([sample])

  # Make a predictin using the saved model
  prediction = model.predict(input_data).tolist()[0]

  # Return the prediction as a JSON response
  return jsonify({'Sales': prediction})

# Define the end point to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():

  # Get the uploaded CSV file from the request
  file = request.files['file']

  # Read the file into a dataframe
  input_data = pd.read_csv(file)

  # Make predictions for the batch data
  predictions = model.predict(input_data).tolist()

  # Create an output dictionary mapping row index to predicted sales
  output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

  return output_dict

# Run the Flask app in debug mode
if __name__ == '__main__':
  superkart_api.run(debug=True)

