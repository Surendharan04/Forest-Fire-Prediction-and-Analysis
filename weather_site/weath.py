from flask import Flask, render_template, request
import math
import numpy as np

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

	temp = list_of_data['main']['temp']
	pressure = list_of_data['main']['pressure']
	humidity = list_of_data['main']['humidity']
	wind_speed = float(list_of_data['wind']['speed']) * 3.6

	# data for variable list_of_data
	data = {
		"temperature": f"{temp} Celsius",
		"pressure": pressure,
		"humidity": humidity,
		"windspeed": f"{wind_speed} km/hr"
	}

	# Calculate FFMC
	ffmc = (59.5 * (math.exp((temp / -13.7)))) - (0.1 * humidity) + (wind_speed * 0.15)

	# Calculate DMC
	dmc = 15 * np.exp(0.007 * ffmc)

	# Calculate DC
	dc = 1.5 * dmc

	# Calculate BUI
	bui = (dmc + dc) / 2

	# Define FWI
	fwi = (0.0272 * pow(bui, 0.9)) * pow(ffmc, 0.5) * pow(1.0 + (0.5 * wind_speed / 3.6), 2.0)

	# Calculate ISI
	isi = (wind_speed / 0.4) * math.sqrt(fwi)

	# Calculate FFMC
	ffmc = (59.5 * (math.exp((temp - 273) / -13.7))) - (0.1 * humidity) + (wind_speed * 0.15)

	# Round
	ffmc = round(ffmc, 1)
	dmc = round(dmc, 1)
	dc = round(dc, 1)
	bui = round(bui, 1)
	isi = round(isi, 1)
	fwi = round(fwi, 1)

	print("FFMC:", ffmc)
	print("DMC:", dmc)
	print("DC:", dc)
	print("BUI:", bui)
	print("ISI:", isi)
	print("FWI:", fwi)

	print(data)
	print(list_of_data)
	return render_template('index.html', data = data)



if __name__ == '__main__':
	app.run(debug = True)


'''	# Rounding
	ffmc = round(ffmc, 1)
	dmc = round(dmc, 1)
	dc = round(dc, 1)
	bui = round(bui, 1)
	isi = round(isi, 1)
	fwi = round(fwi, 1)'''
