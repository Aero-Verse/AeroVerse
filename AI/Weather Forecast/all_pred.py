from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import HTMLResponse
import requests
import folium
from datetime import datetime, timedelta
import pytz
import numpy as np

router = APIRouter()  

API_KEY = "39357f117056c535298fe0df516ce3e3"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def degrees_to_direction(degrees):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = round(degrees / (360. / len(directions))) % len(directions)
    return directions[index]

def predict_rain_probability(temp, humidity):
    prob = (100 - humidity) * 0.5 + (30 - min(30, temp)) * 0.5
    return min(100, max(0, prob))

@router.get("/api/forecast")
async def get_forecast(
    city: str = Query(..., min_length=2),
    hours: int = Query(12, ge=1, le=24)
):
    try:
        url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        current = response.json()

        m = folium.Map(location=[current['coord']['lat'], current['coord']['lon']], zoom_start=10)
        folium.Marker(
            [current['coord']['lat'], current['coord']['lon']],
            popup=f"{current['name']}<br>Temp: {current['main']['temp']}°C"
        ).add_to(m)

        forecast = []
        now = datetime.now(pytz.UTC)
        base_temp = current['main']['temp']
        base_humidity = current['main']['humidity']
        base_wind = current['wind']['speed']
        base_dir = current['wind'].get('deg', 0)

        for i in range(1, hours + 1):
            temp_change = np.sin(i * 0.5) * 2
            humidity_change = np.cos(i * 0.3) * 10
            wind_change = np.random.uniform(-1, 1)
            
            forecast.append({
                "time": (now + timedelta(hours=i)).strftime("%H:%M"),
                "temp": round(base_temp + temp_change, 1),
                "humidity": int(base_humidity + humidity_change),
                "wind_direction": degrees_to_direction(base_dir + i * 10),
                "rain_probability": round(predict_rain_probability(
                    base_temp + temp_change,
                    base_humidity + humidity_change
                ))
            })

        return {
            "status": "success",
            "city": current['name'],
            "country": current['sys'].get('country', 'N/A'),
            "forecast": forecast,
            "map_html": m._repr_html_()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboard", response_class=HTMLResponse)
async def weather_dashboard(
    city: str = Query("Cairo"),
    hours: int = Query(12, ge=1, le=24)
):
    try:
        data = await get_forecast(city, hours)
        
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Weather Forecast - {data['city']}</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                #map {{ height: 500px; width: 100%; border-radius: 8px; }}
                .forecast-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .forecast-table th, .forecast-table td {{
                    padding: 8px; text-align: center; border: 1px solid #ddd;
                }}
                .forecast-table th {{ background-color: #f2f2f2; }}
                .rain-high {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Weather Forecast for {data['city']}, {data['country']}</h1>
                <div id="map">{data['map_html']}</div>
                
                <h2>{hours}-Hour Forecast</h2>
                <table class="forecast-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Temp (°C)</th>
                            <th>Humidity (%)</th>
                            <th>Direction</th>
                            <th>Rain Chance</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(
                            f'<tr>'
                            f'<td>{f["time"]}</td>'
                            f'<td>{f["temp"]}</td>'
                            f'<td>{f["humidity"]}</td>'
                            f'<td>{f["wind_direction"]}</td>'
                            f'<td class="{"rain-high" if f["rain_probability"] > 50 else ""}">'
                            f'{f["rain_probability"]}%</td>'
                            f'</tr>'
                            for f in data['forecast']
                        )}
                    </tbody>
                </table>
            </div>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)