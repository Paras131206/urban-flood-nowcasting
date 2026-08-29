import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_rainfall_forecast(latitude, longitude, forecast_hours=48):
    """
    Get hourly rainfall forecast and calculate
    forward cumulative rainfall.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "rain",
        "forecast_hours": forecast_hours,
        "timezone": "auto",
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    times = data["hourly"]["time"]
    rainfall = data["hourly"]["rain"]

    result = []

    for i, time in enumerate(times):

        def forward_sum(hours):
            end = min(len(rainfall), i + hours)
            values = rainfall[i:end]

            return round(
                sum(v or 0 for v in values),
                2
            )

        result.append({
            "time": time,
            "rainfall_mm": round(
                rainfall[i] or 0,
                2
            ),
            "next_1h_mm": forward_sum(1),
            "next_3h_mm": forward_sum(3),
            "next_6h_mm": forward_sum(6),
            "next_24h_mm": forward_sum(24),
        })

    return result



def get_current_rainfall_summary(latitude, longitude):
    """Return rainfall forecast starting from the first forecast hour."""

    forecast = get_rainfall_forecast(
        latitude,
        longitude,
        forecast_hours=48,
    )

    if not forecast:
        return None

    current = forecast[0]

    return {
        "time": current["time"],
        "rainfall_mm": current["rainfall_mm"],
        "next_1h_mm": current["next_1h_mm"],
        "next_3h_mm": current["next_3h_mm"],
        "next_6h_mm": current["next_6h_mm"],
        "next_24h_mm": current["next_24h_mm"],
    }
