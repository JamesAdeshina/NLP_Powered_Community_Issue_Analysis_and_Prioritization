import re
import ssl
import certifi
import logging
from typing import List, Tuple, Optional
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode
import pandas as pd
from config import UK_POSTCODE_REGEX, UK_ADDRESS_REGEX, OPENCAGE_API_KEY

# Create an SSL context using certifi's CA bundle, then disable verification (for testing only)
ctx = ssl.create_default_context(cafile=certifi.where())
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Initialize a module-level logger
logger = logging.getLogger(__name__)

def clean_address(address: str) -> str:
    """
    Remove common prefixes (like 'Address:') and extra whitespace.
    """
    return address.replace("Address:", "").strip()

def resolve_postcode_to_address(postcode: str, api_key: str = OPENCAGE_API_KEY) -> Optional[str]:
    """
    Resolves a postcode to a full address using the OpenCage Geocoder.
    """
    geocoder = OpenCageGeocode(api_key)
    try:
        results = geocoder.geocode(f"{postcode}, UK")
        if results and len(results) > 0:
            return results[0]['formatted']
        logger.warning(f"Could not resolve postcode {postcode} to a full address.")
        return None
    except Exception as e:
        logger.error(f"Error resolving postcode {postcode}: {str(e)}", exc_info=True)
        return None

def geocode_addresses(addresses: List[str], api_key: str = OPENCAGE_API_KEY) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Geocodes a list of addresses or postcodes using the OpenCage Geocoder.
    """
    geocoder = OpenCageGeocode(api_key)
    locations = []

    for address in addresses:
        address = clean_address(address)
        logger.debug(f"Processing address: {address}")
        if not address:
            locations.append((None, None))
            continue

        try:
            if re.match(r'^[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s*\d[A-Za-z]{2}$', address.strip()):
                full_address = resolve_postcode_to_address(address, api_key)
                if not full_address:
                    logger.warning(f"Skipping postcode {address} (could not resolve to full address).")
                    locations.append((None, None))
                    continue
            else:
                full_address = f"{address}, Bolsover, UK"

            results = geocoder.geocode(full_address)
            if results and len(results) > 0:
                lat = results[0]['geometry']['lat']
                lon = results[0]['geometry']['lng']
                locations.append((lat, lon))
            else:
                logger.warning(f"Geocoding failed for address: {full_address}")
                locations.append((None, None))
        except Exception as e:
            logger.error(f"Geocoding error for {address}: {str(e)}", exc_info=True)
            locations.append((None, None))
    return locations

def geocode_location(location_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Geocode a single location using Nominatim.
    """
    geolocator = Nominatim(user_agent="bolsover_analysis", ssl_context=ctx)
    location_name = clean_address(location_name)
    postcode_match = re.search(UK_POSTCODE_REGEX, location_name, re.IGNORECASE)

    if postcode_match:
        try:
            location = geolocator.geocode(postcode_match.group(0), exactly_one=True)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            logger.error(f"Postcode geocoding error: {str(e)}", exc_info=True)

    try:
        location = geolocator.geocode(f"{location_name}, Bolsover District, UK", exactly_one=True)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        logger.error(f"Address geocoding error: {str(e)}", exc_info=True)

    return (None, None)

def extract_locations(text: str) -> Optional[str]:
    """
    Extracts the first valid address or postcode from the given text.
    """
    addresses = re.findall(UK_ADDRESS_REGEX, text, flags=re.IGNORECASE)
    postcodes = re.findall(UK_POSTCODE_REGEX, text, flags=re.IGNORECASE)
    candidates = set(addresses + postcodes)

    for loc in candidates:
        if re.match(UK_ADDRESS_REGEX, loc, flags=re.IGNORECASE) or re.match(UK_POSTCODE_REGEX, loc, flags=re.IGNORECASE):
            return loc
    return None
