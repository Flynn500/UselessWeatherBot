import openmeteo_requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import requests_cache
import pandas as pd
from retry_requests import retry
from transformers import pipeline
import os

##call OpenMetro api and store in dataframe
def call_api() -> pd.DataFrame:
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -42,
        "longitude": 174,
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "rain"]
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_rain = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["rain"] = hourly_rain

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe
    


 ##get our data locally or call the api
def get_data(from_local=False) -> pd.DataFrame:
    if from_local:
        df = pd.read_csv("weather_data.csv")
    else:
        df = call_api()
        save_data(df)
        
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

##convert our data to kelvin for no reason at all
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df['temperature_2m'] = df['temperature_2m'] + 273.15
    df['apparent_temperature'] = df['apparent_temperature'] + 273.15
    return df

##store our data in a csv
def save_data(df: pd.DataFrame):
    cols = ["date","temperature_2m","relative_humidity_2m","apparent_temperature","rain"]
    
    wd = os.path.dirname(__file__)
    path = os.path.join(wd, "forecasts")
    if not os.path.exists("forecasts"):
        os.makedirs("forecasts")
    path = os.path.join("forecasts", "weather_data.csv")
    df.to_csv(path, mode='a', header=cols, index=False)

##make our report and save as pdf
def generate_report(df: pd.DataFrame):
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.8, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['apparent_temperature'])
    plt.title("OpenMetro weather forecasts")
    plt.xlabel("Date")
    plt.ylabel("Temperature (Kelvin)")
    plt.tight_layout(rect=[0, 0, 1, 1])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    fig.text(0.1, 0.4, call_face_hugger(),fontsize=9, ha='left')

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(df.index, df['relative_humidity_2m'])
    plt.title("")
    plt.ylabel("Humidity")
    plt.tight_layout(rect=[0, 0, 1, 1])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ##save our results
    wd = os.path.dirname(__file__)
    path = os.path.join(wd, "forecasts")
    path = os.path.join("forecasts", "report.pdf")
    plt.savefig(path)

def call_face_hugger() -> str:
    input = "The weather forecasts for today is"
    generator = pipeline('text-generation', model='gpt2')
    response = generator(input, max_length=100)

    return parse_response(response)

def parse_response(response) -> str:
    output = ""
    counter = 0
    words = response[0]['generated_text'].split(" ")
    for i in words:
        output = output + " " + i
        counter += len(i)
        if counter > 60:
            output += "\n"
            counter = 0
    return output


df = get_data()
df = transform_data(df)
generate_report(df)




