# Airbnb-Price-Prediction
Airbnb Price Prediction using Gradient Boosting. Flask web app that predicts price per bed from features like rooms, guests, reviews, ratings &amp; country.

🏡 Airbnb Price Prediction (Gradient Boosting) — Flask App

This project is an end-to-end ML web app that predicts price per bed for Airbnb listings. Users enter features (bedrooms, bathrooms, guests, beds, country, rating, reviews, check-in/out times) and the app returns an estimated price.
Under the hood, a Gradient Boosting Regressor learns non-linear relationships between features and price. The pipeline covers data cleaning (especially messy time strings), model training/evaluation, and a Flask front end with a clean UI.

Use it to showcase full-stack ML skills: data → model → deployment. It’s lightweight, easy to run locally, and simple to extend (e.g., add city/seasonality, dynamic pricing).

You can fill the essential details like the bedrooms, bathrooms, beds, guests, country and reviews and you will be able to see the predicted price.


<img width="1896" height="919" alt="Screenshot 2025-09-02 172234" src="https://github.com/user-attachments/assets/4c3a6526-a7e0-4e82-8fc3-61e627d85c2a" />


<img width="1892" height="905" alt="Screenshot 2025-09-02 172326" src="https://github.com/user-attachments/assets/03d3ad2c-4db0-49b7-81ad-05ca99f9c79d" />


<img width="1919" height="447" alt="Screenshot 2025-09-02 172341" src="https://github.com/user-attachments/assets/7a9d3131-136d-467f-8f8f-d7092c224dfb" />



🔧 Tech Stack & Libraries

Python, Flask, Jinja2 (web)

pandas, numpy (data prep)

scikit-learn (GradientBoostingRegressor, metrics)

joblib (model persistence)

(Optional) matplotlib (EDA visuals)



⬇️ Download / Clone
Go to the repo page → Code → Download ZIP → unzip → open folder in your terminal/IDE.
You can also clone the the code

▶️ Setup & Run (Windows/Mac/Linux)

1.Create a virtual environment

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate

2.Install dependencies

pip install -r requirements.txt


Example requirements.txt:

Flask
pandas
numpy
scikit-learn
joblib

3.Prepare data & train the model

Put your dataset as airbnb.csv in the project root.

Train:

python train.py


This saves model.pkl.

4.Run the web app

python app.py

