import geocoder

def test_reverse_geocode(lat, lon):
    try:
        # Using OpenStreetMap (OSM) for free reverse geocoding
        g = geocoder.osm([lat, lon], method='reverse')
        print(f"Input: {lat}, {lon}")
        print(f"Status: {g.status}")
        if g.ok:
            print(f"City: {g.city}")
            print(f"Town: {g.town}")
            print(f"Village: {g.village}")
            print(f"Address: {g.address}")
        else:
            print("Geocoding failed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Approx coords for Kolavda
    test_reverse_geocode(23.275, 72.695)
