"""Climate tool providing static climate data and failure simulation."""
from typing import Any, Dict


CLIMATE_DATABASE = {
    "kerala": {
        "location": "Kerala",
        "climate": "hot_humid",
        "temperature_c": 32,
        "humidity_percent": 78,
        "wind_speed_ms": 2.5,
        "rainfall_level": "heavy",
        "description": "Tropical monsoon climate with high humidity and substantial rainfall."
    },
    "rajasthan": {
        "location": "Rajasthan",
        "climate": "hot_arid",
        "temperature_c": 42,
        "humidity_percent": 22,
        "wind_speed_ms": 3.8,
        "rainfall_level": "scanty",
        "description": "Arid desert climate with extreme daytime heat and low humidity."
    },
    "assam": {
        "location": "Assam",
        "climate": "hot_humid",
        "temperature_c": 30,
        "humidity_percent": 82,
        "wind_speed_ms": 1.8,
        "rainfall_level": "very_heavy",
        "description": "Subtropical humid climate prone to flooding and heavy monsoons."
    },
    "ladakh": {
        "location": "Ladakh",
        "climate": "cold_arid",
        "temperature_c": -5,
        "humidity_percent": 30,
        "wind_speed_ms": 4.5,
        "rainfall_level": "very_low",
        "description": "High altitude cold desert with extreme winter temperatures."
    },
    "default": {
        "location": "Generic Hot-Humid Zone",
        "climate": "hot_humid",
        "temperature_c": 31,
        "humidity_percent": 75,
        "wind_speed_ms": 2.2,
        "rainfall_level": "moderate",
        "description": "Standard hot-humid reference baseline."
    }
}


def get_climate(location: str = "Kerala", simulate_fail: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Retrieve static climate data for a given location or trigger simulated failure.
    
    Args:
        location: City, state, or region name.
        simulate_fail: If True, simulates an external API failure.
        
    Returns:
        Climate dictionary or error structure.
    """
    if simulate_fail or location.lower() in ["fail", "error", "unavailable"]:
        return {
            "success": False,
            "error": f"Climate service unavailable for location '{location}' (simulated network timeout)."
        }

    clean_loc = location.strip().lower()
    for key, data in CLIMATE_DATABASE.items():
        if key in clean_loc or clean_loc in key:
            res = dict(data)
            res["success"] = True
            return res

    # Fallback to default
    res = dict(CLIMATE_DATABASE["default"])
    res["location"] = location
    res["success"] = True
    return res
