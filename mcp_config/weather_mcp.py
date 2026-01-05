"""Weather MCP server using Open-Meteo API (free, no API key required)."""
import requests
from typing import Dict, Any, Optional
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError


class WeatherMCP:
    """
    Weather MCP server that fetches weather data from Open-Meteo API.
    No API key required - completely free!
    """
    
    # Open-Meteo API endpoints
    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self):
        """Initialize the Weather MCP server."""
        pass
    
    def get_weather(self, location: str, day_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get weather for a location (current or forecast).
        
        Args:
            location: City name (e.g., "Paris", "New York")
            day_index: Optional day index (0 = today, 1 = tomorrow, 2 = day after tomorrow, etc., max 6)
                      If None or 0, returns current weather. If > 0, returns forecast for that day.
            
        Returns:
            Dictionary with weather data or error information
        """
        try:
            # Step 1: Get coordinates for the location
            geo_data = self._geocode_location(location)
            if not geo_data:
                return {
                    "success": False,
                    "error": f"Location '{location}' not found. Please check the spelling.",
                    "error_type": "location_not_found"
                }
            
            # Step 2: Get weather data (with forecast if needed)
            weather_data = self._fetch_weather(
                latitude=geo_data["latitude"],
                longitude=geo_data["longitude"],
                day_index=day_index
            )
            
            # Step 3: Format and return result
            return {
                "success": True,
                "location": {
                    "name": geo_data["name"],
                    "country": geo_data.get("country", ""),
                    "latitude": geo_data["latitude"],
                    "longitude": geo_data["longitude"]
                },
                "weather": weather_data,
                "day_index": day_index
            }
            
        except (ConnectionError, Timeout) as e:
            return {
                "success": False,
                "error": "Sorry, but I am not able to provide you information due to connection problems. Please check your internet connection and try again.",
                "error_type": "connection_error",
                "details": str(e)
            }
        except HTTPError as e:
            return {
                "success": False,
                "error": f"Weather service returned an error (HTTP {e.response.status_code if hasattr(e, 'response') else 'unknown'}). Please try again later.",
                "error_type": "http_error",
                "details": str(e)
            }
        except RequestException as e:
            return {
                "success": False,
                "error": "Failed to fetch weather data from external service. Please try again later.",
                "error_type": "api_error",
                "details": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "error": "An unexpected error occurred while fetching weather data. Please try again later.",
                "error_type": "unexpected_error",
                "details": str(e)
            }
    
    def _geocode_location(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Convert location name to coordinates.
        
        Args:
            location: City name
            
        Returns:
            Dictionary with latitude, longitude, name, country or None
        """
        try:
            response = requests.get(
                self.GEOCODING_URL,
                params={
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                },
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"],
                    "name": result["name"],
                    "country": result.get("country", "")
                }
            return None
            
        except (ConnectionError, Timeout) as e:
            # Re-raise connection errors to be handled by get_weather
            # Preserve the original exception type
            raise
        except HTTPError as e:
            # Re-raise HTTP errors to be handled by get_weather
            raise HTTPError(f"Geocoding service returned an error: {str(e)}")
        except RequestException as e:
            # Re-raise request errors to be handled by get_weather
            raise RequestException(f"Failed to geocode location: {str(e)}")
        except Exception as e:
            # Re-raise unexpected errors to be handled by get_weather
            raise Exception(f"Unexpected error during geocoding: {str(e)}")
    
    def _fetch_weather(self, latitude: float, longitude: float, day_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch weather data from Open-Meteo API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            day_index: Optional day index (0 = today, 1 = tomorrow, etc., max 6)
                      If None or 0, returns current weather. If > 0, returns forecast for that day.
            
        Returns:
            Dictionary with weather data (current or forecast)
        """
        # Always fetch both current and daily forecast (extend to 7 days for flexibility)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "forecast_days": 7
        }
        
        try:
            response = requests.get(self.WEATHER_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except (ConnectionError, Timeout) as e:
            # Re-raise to preserve exception type for get_weather handler
            raise
        except HTTPError as e:
            raise HTTPError(f"Weather API returned an error: {str(e)}")
        except RequestException as e:
            raise RequestException(f"Failed to fetch weather data: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error while fetching weather: {str(e)}")
        
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        # If day_index > 0, return forecast for that specific day
        if day_index is not None and day_index > 0:
            # Return forecast for specific day
            if daily.get("time") and len(daily["time"]) > day_index:
                return {
                    "temperature": daily.get("temperature_2m_max", [None])[day_index] if daily.get("temperature_2m_max") else None,
                    "temperature_min": daily.get("temperature_2m_min", [None])[day_index] if daily.get("temperature_2m_min") else None,
                    "weather": self._get_weather_description(daily.get("weather_code", [0])[day_index] if daily.get("weather_code") else 0),
                    "precipitation": daily.get("precipitation_sum", [0])[day_index] if daily.get("precipitation_sum") else 0,
                    "date": daily.get("time", [""])[day_index] if daily.get("time") else "",
                    "day_index": day_index,
                    "is_forecast": True,
                    "all_forecast": self._format_forecast(daily) if daily else None
                }
        
        # Default: return current weather with full forecast
        weather_code = current.get("weather_code", 0)
        weather_desc = self._get_weather_description(weather_code)
        
        return {
            "temperature": current.get("temperature_2m", 0),
            "humidity": current.get("relative_humidity_2m", 0),
            "weather": weather_desc,
            "wind_speed": current.get("wind_speed_10m", 0),
            "is_forecast": False,
            "day_index": day_index or 0,
            "forecast": self._format_forecast(daily) if daily else None
        }
    
    def _format_forecast(self, daily: Dict[str, Any]) -> Dict[str, Any]:
        """Format daily forecast data."""
        if not daily.get("time"):
            return None
        
        forecast = []
        for i in range(min(3, len(daily["time"]))):
            forecast.append({
                "date": daily["time"][i],
                "max_temp": daily.get("temperature_2m_max", [None])[i] if daily.get("temperature_2m_max") else None,
                "min_temp": daily.get("temperature_2m_min", [None])[i] if daily.get("temperature_2m_min") else None,
                "precipitation": daily.get("precipitation_sum", [0])[i] if daily.get("precipitation_sum") else 0,
                "weather": self._get_weather_description(daily.get("weather_code", [0])[i] if daily.get("weather_code") else 0)
            })
        return forecast
    
    def _get_weather_description(self, code: int) -> str:
        """
        Convert WMO weather code to human-readable description.
        
        Args:
            code: WMO weather code
            
        Returns:
            Weather description string
        """
        # Common weather codes from WMO
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(code, "Unknown")

