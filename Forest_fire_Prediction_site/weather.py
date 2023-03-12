from flask import Flask, render_template, request

# import json to load JSON data to a python dictionary
import json

# urllib.request to make a request to api
import urllib.request

app = Flask(__name__)

@app.route('/', methods =['POST', 'GET'])
def weather():
	if request.method == 'POST':
		city = request.form['city']
		pincode = request.form['pin']
	else:
		# for default name
		city = "Chennai"
		pincode = "600091"

	# your API key will come here
	api = "6b7579b332fc4adb1e0501bdc205a8c2"
    
	link = f"https://api.openweathermap.org/data/2.5/weather?q={city},{pincode}&units=metric&appid={api}"

	# source contain json data from api
	source = urllib.request.urlopen(link).read()

	# converting JSON data to a dictionary
	list_of_data = json.loads(source)

	# data for variable list_of_data
	data = {
		"temperature": str(list_of_data['main']['temp']) + ' Celsius',
		"pressure": str(list_of_data['main']['pressure']),
		"humidity": str(list_of_data['main']['humidity']),
		"windspeed": float(str(list_of_data['wind']['speed']))
	}
	print(data)
	print(list_of_data)
	return render_template('index.html', data = data)



if __name__ == '__main__':
	app.run(debug = True)

'''import math

# Retrieve temperature, humidity, and wind speed from OpenWeather API
temp = 293.15 # in Kelvin
humidity = 80 # in percentage
wind_speed = 10 # in km/h

# Calculate FFMC
ffmc = (59.5 * (math.exp((temp - 273) / -13.7))) - (0.1 * humidity) + (wind_speed * 0.15)

# Calculate DMC
dmc = 15 * math.exp(0.007 * ffmc)

# Calculate DC
dc = 1.5 * dmc

# Calculate BUI
bui = (dmc + dc) / 2

# Calculate ISI
isi = (wind_speed / 0.4) * math.sqrt(fwi)

# Calculate FWI
fwi = (0.0272 * pow(bui, 0.9)) * pow(ffmc, 0.5) * pow(1.0 + (0.5 * wind_speed / 3.6), 2.0)

print("FFMC:", ffmc)
print("DMC:", dmc)
print("DC:", dc)
print("BUI:", bui)
print("ISI:", isi)
print("FWI:", fwi)'''