from geopy.geocoders import Nominatim

def test_geopy(lat, lon):
    try:
        # Nominatim requires a user_agent
        geolocator = Nominatim(user_agent="my_weather_app_1.0")
        location = geolocator.reverse((lat, lon), language='en')
        
        print(f"Input: {lat}, {lon}")
        if location:
            print(f"Address: {location.address}")
            print(f"Raw: {location.raw}")
            # Try to extract best name
            addr = location.raw.get('address', {})
            # Priorities: Village -> Town -> City -> Suburb
            name = addr.get('village') or addr.get('town') or addr.get('city') or addr.get('suburb')
            print(f"Extracted Name: {name}")
        else:
            print("Location not found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_geopy(23.275, 72.695)
