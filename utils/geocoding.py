import re
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode
import pandas as pd
from config import UK_POSTCODE_REGEX, UK_ADDRESS_REGEX, OPENCAGE_API_KEY


def resolve_postcode_to_address(postcode, api_key=OPENCAGE_API_KEY):
    geocoder = OpenCageGeocode(api_key)
    try:
        results = geocoder.geocode(f"{postcode}, UK")
        if results and len(results) > 0:
            return results[0]['formatted']
        return None
    except Exception as e:
        print(f"Error resolving postcode {postcode}: {str(e)}")
        return None


def geocode_addresses(addresses, api_key=OPENCAGE_API_KEY):
    geocoder = OpenCageGeocode(api_key)
    locations = []

    for address in addresses:
        if not address:
            locations.append((None, None))
            continue

        try:
            if re.match(r'^[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s*\d[A-Za-z]{2}$', address.strip()):
                full_address = resolve_postcode_to_address(address, api_key)
                if not full_address:
                    print(f"Skipping postcode {address} (could not resolve to full address).")
                    locations.append((None, None))
                    continue
            else:
                full_address = f"{address}, Bolsover, UK"

            results = geocoder.geocode(full_address)
            if results and len(results) > 0:
                locations.append((results[0]['geometry']['lat'], results[0]['geometry']['lng']))
            else:
                locations.append((None, None))
        except Exception as e:
            print(f"Geocoding error for {address}: {str(e)}")
            locations.append((None, None))

    return locations


def geocode_location(location_name):
    geolocator = Nominatim(user_agent="bolsover_analysis")
    postcode_match = re.search(UK_POSTCODE_REGEX, location_name, re.IGNORECASE)

    if postcode_match:
        try:
            location = geolocator.geocode(postcode_match.group(0), exactly_one=True)
            if location:
                return (location.latitude, location.longitude)
        except Exception