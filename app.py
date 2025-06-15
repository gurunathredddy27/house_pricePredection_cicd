from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and feature columns
model = pickle.load(open('house.pkl', 'rb'))
rf_model = pickle.load(open('house_rf.pkl', 'rb'))
try:
    features = pickle.load(open('features.pkl', 'rb'))
except Exception as e:
    print("Error loading features.pkl:", e)
    features = None  # fallback

cities = [
    'Shoreline', 'Seattle', 'Kent', 'Bellevue', 'Redmond',
    'Maple Valley', 'North Bend', 'Lake Forest Park', 'Sammamish',
    'Auburn', 'Des Moines', 'Bothell', 'Federal Way', 'Kirkland',
    'Issaquah', 'Woodinville', 'Normandy Park', 'Fall City', 'Renton',
    'Carnation', 'Snoqualmie', 'Duvall', 'Burien', 'Covington',
    'Inglewood-Finn Hill', 'Kenmore', 'Newcastle', 'Mercer Island',
    'Black Diamond', 'Ravensdale', 'Clyde Hill', 'Algona', 'Skykomish',
    'Tukwila', 'Vashon', 'Yarrow Point', 'SeaTac', 'Medina',
    'Enumclaw', 'Snoqualmie Pass', 'Pacific', 'Beaux Arts Village',
    'Preston', 'Milton'
]

countries = ['USA']

@app.route('/')
def home():
    return render_template('index.html', cities=cities, countries=countries, prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    sqft_living = float(request.form['sqft_living'])
    floors = float(request.form['floors'])
    yr_built = int(request.form['yr_built'])
    city_selected = request.form['city']
    country_selected = request.form['country']

    input_data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'floors': floors,
        'yr_built': yr_built,
    }

    df = pd.DataFrame([input_data])

    # One-hot encode city
    for c in cities:
        df[f'city_{c}'] = 1 if c == city_selected else 0

    # One-hot encode country
    for c in countries:
        df[f'country_{c}'] = 1 if c == country_selected else 0

    if features is not None:
        # Align columns with training features and fill missing with zeros
        df = df.reindex(columns=features, fill_value=0)
    else:
        # If features not loaded, just keep all columns as is (may cause errors)
        print("Warning: features.pkl not loaded, columns might not match.")

    prediction = model.predict(df)[0]
    prediction1 = rf_model.predict(df)[0]
    
# used both the random forest and linear regression
    return render_template('index.html', cities=cities, countries=countries,
                       linear_pred=f"Linear Regression House Price: ${prediction:,.2f}",
                       rf_pred=f"Random forest House Price: ${prediction1:,.2f}")



    # return render_template('index.html', cities=cities, countries=countries,
    #                        prediction_text=f'Estimated House Price: ${prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
