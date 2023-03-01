from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/weather", methods=["POST"])
def weather():
    api_key = "6b7579b332fc4adb1e0501bdc205a8c2"
    city = request.form["city"]
    pincode = request.form["pin"]
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},{pincode}&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract weather information from the JSON response
        weather_description = data["weather"][0]["description"].capitalize()
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return f"Weather in {city}: {weather_description}<br>Temperature: {temperature} Celsius<br>Humidity: {humidity}%"
    else:
        return "Error fetching weather data"

if __name__ == "__main__":
    app.run(debug=True)
