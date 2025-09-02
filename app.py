from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and pipeline
model = joblib.load("model.pkl")         # GradientBoostingRegressor
pipeline = joblib.load("pipeline.pkl")   # Preprocessing pipeline

# Route to display the form
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', form_data={})

# Route to handle form submission and display result
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    try:
        # Create DataFrame from form
        input_df = pd.DataFrame([form_data])

        # Convert numerical values
        numeric_fields = ['bedrooms', 'bathrooms', 'guests', 'beds', 'rating', 'reviews']
        for field in numeric_fields:
            input_df[field] = pd.to_numeric(input_df[field], errors='coerce')

        # Transform data
        transformed_input = pipeline.transform(input_df)

        # Predict
        prediction = round(model.predict(transformed_input)[0], 2)

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        return render_template('home.html', form_data=form_data, prediction=error_message)

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request
# import pandas as pd
# import joblib

# app = Flask(__name__)

# # Load model and pipeline
# model = joblib.load("model.pkl")         # GradientBoostingRegressor
# pipeline = joblib.load("pipeline.pkl")   # Preprocessing pipeline

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     form_data = {}

#     if request.method == 'POST':
#         form_data = request.form.to_dict()

#         try:
#             # Create DataFrame from form
#             input_df = pd.DataFrame([form_data])

#             # Convert numerical values
#             numeric_fields = ['bedrooms', 'bathrooms', 'guests', 'beds', 'rating', 'reviews']
#             for field in numeric_fields:
#                 input_df[field] = pd.to_numeric(input_df[field], errors='coerce')

#             # Transform data
#             transformed_input = pipeline.transform(input_df)

#             # Predict
#             prediction = round(model.predict(transformed_input)[0], 2)

#         except Exception as e:
#             prediction = f"Error: {str(e)}"

#     return render_template('home.html', prediction=prediction, form_data=form_data)

# if __name__ == '__main__':
#     app.run(debug=True)
