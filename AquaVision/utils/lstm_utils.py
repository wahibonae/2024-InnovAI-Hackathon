import numpy as np
import joblib
from pandas import read_csv
import keras
import matplotlib.pyplot as plt

class AIWaterForcasting:
    """Handles water parameter predictions using LSTM model for African countries"""
    
    def __init__(self, country_name, dataname, model_path, scaler_path):
        # Load the dataset and trained model
        self.country_name = country_name
        self.data = read_csv(dataname, header=0, index_col=0)
        self.model = keras.models.load_model(model_path)
        self.scaler_path = scaler_path

    def SeparateCountries(self):
        # Split dataset by country and sort alphabetically
        values = self.data.values
        countries = sorted(list(set(values[:,0])))
        
        # Group data points by country
        i = 1
        sep_values = []
        eachcount = []
        for country in values:
            if country[0] != countries[i]:
                eachcount.append(country.tolist())
            else:
                sep_values.append(eachcount)
                eachcount = []
                i = i + 1
                if(i >= len(countries)):
                    break
        return sep_values
  
    def getPastData(self):
        # Get historical data for specified country
        sep = self.SeparateCountries()
        country_id = 0
        for i in range(len(sep)):
            if sep[i][0][0].upper() == self.country_name.upper():
                country_id = i
                break
        return np.array(sep[country_id])[:,2:]

    def format_float(self, value):
        return "{:.2f}".format(value)
    
    def Predict(self, num):
        """Generate predictions for the next 'num' time steps"""
        allforecast = []
        sep = self.SeparateCountries()
        
        # Find the target country's data
        country_id = 0
        for i in range(len(sep)):
            if sep[i][0][0].upper() == self.country_name.upper():
                country_id = i
                break

        # Load scaler used during training
        standard_scaler = joblib.load(self.scaler_path)
        
        # Make predictions for each time step
        for i in range(num):
            if i == 0:
                # Use last 10 time steps for initial prediction
                chosen = np.array(sep[country_id])[-10:,2:]
                country = np.array(chosen)
            else:
                country = np.array(country)
                
            # Scale data, reshape for LSTM, and predict
            scaled_test_df = standard_scaler.transform(country)
            scaled_test_df = scaled_test_df.reshape((-1, 10, 6))
            y_pred = self.model.predict(scaled_test_df)
            y_pred = standard_scaler.inverse_transform(y_pred)
            
            # Slide window forward
            country = country.tolist()
            y_pred = y_pred.tolist()[0]
            country.append(y_pred)
            country = country[1:]
            allforecast.append(y_pred)
            
        return allforecast

# Helper functions to format data for display
def get_water_data(country):
    """Format historical water data for display"""
    Predictor = AIWaterForcasting(country,'lstm/african_countries_data.csv','lstm/af.keras','lstm/standard_scaler.pkl')
    data = ""
    year = 2000

    for i in Predictor.getPastData()[:,:-1]:
        data += "Year: " + str(year)
        for j in i:
            data += " " + str(Predictor.format_float(float(j))) + " "
        data += "\n"
        year = year + 1

    output = "Country: " + country + "\n" + "Water Parameters: AFW agriculture, AFW domestic, AFW industry, Water stress, Water Use Efficiency" + "\n" + "Previous decade data: " + data
    return output

def get_predictions(country):
    """Format prediction data for display"""
    Predictor = AIWaterForcasting(country,'lstm/african_countries_data.csv','lstm/af.keras','lstm/standard_scaler.pkl')
    Allpredicts = Predictor.Predict(5)
    prediction = ""
    year = 2021  # Starting year for predictions
    
    Allpredicts = np.array(Allpredicts)
    for i in Allpredicts[:,:-1]:
        prediction += "Year: " + str(year)
        for j in i:
            prediction += " " + str(Predictor.format_float(float(j))) + " "
        prediction += "\n"
        year = year + 1

    return "Prediction for the next years: " + prediction

def contextify(country):
    """Combine historical and prediction data into one string"""
    historical_context = get_water_data(country)
    predictions = get_predictions(country)
    return historical_context + "\n" + predictions

def draw(Predictor, feature_id):
    """Create visualization of historical data and predictions"""
    # Prepare historical data
    years = []
    year = 2000
    past_data = Predictor.getPastData()[:, feature_id]
        
    for _ in past_data:
        year += 1
        years.append(year)

    values = past_data.astype(float)

    # Get and prepare prediction data
    predictions = Predictor.Predict(5)
    pred_values = np.array(predictions)[:, feature_id].astype(float)
    years.append(2021)
    values = np.append(values, pred_values[0])
    pred_years = []
      
    for i in range(5):
        year += 1
        pred_years.append(year)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, values, marker='o', linestyle='-', color='b', label='Past Data')
    ax.plot(pred_years, pred_values, marker='o', linestyle='--', color='r', label='Predictions')

    ax.set_xlabel('Year')
    ax.set_ylabel('Value')

    # Add padding to y-axis limits
    all_values = np.concatenate([values, pred_values])
    y_min, y_max = np.min(all_values), np.max(all_values)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.grid(True)
    ax.legend()
    return fig