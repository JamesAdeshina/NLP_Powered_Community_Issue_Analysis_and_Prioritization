from geopy.geocoders import Nominatim
import ssl, certifi

ctx = ssl.create_default_context(cafile=certifi.where())
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

geolocator = Nominatim(user_agent="bolsover_analysis", ssl_context=ctx)
test_address = "1 Church Street, Bolsover, S44 6JJ, UK"
location = geolocator.geocode(test_address, exactly_one=True)
print("Test location:", location)
