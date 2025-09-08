from django.shortcuts import render
from .models import Customers
import pandas as pd
import pickle
import os

def customer_list(request):
    result = None
    if request.method == 'POST':
        try:
            # Get input values from the form
            features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
            
            input_data = []
            for feature in features:
                input_data.append(float(request.POST.get(feature.lower(), 0)))

            # Prepare the data for the model
            data = pd.DataFrame([input_data], columns=features)
            
            # Load the model and make a prediction
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
            prediction = model.predict(data)
            result = prediction[0]

        except FileNotFoundError:
            result = "Error: Model file not found."
        except Exception as e:
            result = f"An error occurred: {e}"

    context = {
        'result': result
    }
    return render(request, 'customers.html', context)