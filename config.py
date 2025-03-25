# Standard configuration and constants
import os

# SSL Context for NLTK Downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Page configuration
PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "Bolsover District Council - Analysis",
    "page_icon": "🏛️"
}

# UK address patterns
UK_POSTCODE_REGEX = r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}\b'
UK_ADDRESS_REGEX = r'\b\d+\s[\w\s]+\b,\s[\w\s]+,\s' + UK_POSTCODE_REGEX

# Topic candidate labels
# ------------------ Expanded Candidate Labels for Topic Detection ------------------
CANDIDATE_LABELS_TOPIC: tuple[str, ...] = (
    "Waste Management / Public Cleanliness",
    "Water Scarcity",
    "Food Insecurity",
    "Cybersecurity Threats",
    "Delays in NHS Treatment",
    "Underfunded Healthcare Services",
    "Decline in Local Shops / High Street Businesses",
    "High Cost of Living",
    "Overcrowded Public Transport",
    "Homelessness",
    "Lack of Affordable Housing",
    "Noise Pollution",
    "Potholes / Road Maintenance",
    "Traffic Congestion",
    "Air Pollution",
    "School Overcrowding",
    "Crime Rates in Urban Areas",
    "Limited Green Spaces",
    "Aging Infrastructure",
    "Digital Divide",
    "Rising Energy Costs",
    "Housing Quality Issues",
    "Lack of Social Mobility",
    "Climate Change Adaptation",
    "Elderly Care Shortages",
    "Rural Transport Accessibility",
    "Mental Health Service Shortages",
    "Drug and Alcohol Abuse",
    "Gender Pay Gap",
    "Age Discrimination in Employment",
    "Child Poverty",
    "Bureaucratic Delays in Government Services",
    "Lack of Public Restrooms in Urban Areas",
    "Unsafe Cycling Infrastructure",
    "Tackling Modern Slavery",
    "Gentrification and Displacement",
    "Rise in Anti-Social Behaviour",
    "Tackling Fake News and Misinformation",
    "Integration of Immigrant Communities",
    "Parking Problems",
    "Littering in Public Spaces",
    "Speeding Vehicles",
    "Crumbling Pavements",
    "Public Wi-Fi Gaps",
    "Youth Services Cuts",
    "Erosion of Coastal Areas",
    "Flooding in Residential Areas",
    "Loneliness and Social Isolation",
    "Domestic Violence and Abuse",
    "Racial Inequality and Discrimination",
    "LGBTQ+ Rights and Inclusion",
    "Disability Access",
    "Childcare Costs and Availability",
    "Veteran Support",
    "Community Cohesion",
    "Access to Arts and Culture",
    "Biodiversity Loss",
    "Urban Heat Islands",
    "Single-Use Plastics",
    "Education / Skills Development",
    "Community Workshops",
    "Renewable Energy Transition",
    "Food Waste",
    "Deforestation and Land Use",
    "Light Pollution",
    "Soil Degradation",
    "Marine Pollution",
    "Gig Economy Exploitation",
    "Regional Economic Disparities",
    "Skills Shortages",
    "Zero-Hours Contracts",
    "Pension Inequality",
    "Rising Inflation",
    "Small Business Struggles",
    "Post-Brexit Trade Challenges",
    "Automation and Job Loss",
    "Unpaid Internships",
    "Obesity Epidemic",
    "Dental Care Access",
    "Vaccine Hesitancy",
    "Pandemic Preparedness",
    "Nutritional Education",
    "Physical Inactivity",
    "Student Debt",
    "Teacher Shortages",
    "School Funding Cuts",
    "Bullying in Schools",
    "Access to Higher Education",
    "Vocational Training Gaps",
    "Digital Exclusion",
    "Extracurricular Activity Cuts",
    "Aging Public Buildings",
    "Smart City Development",
    "Electric Vehicle Infrastructure",
    "5G Rollout Delays",
    "Flood Defence Upgrades",
    "Rail Network Overcrowding",
    "AI Ethics and Regulation",
    "Space Debris Management",
    "Genetic Engineering Ethics",
    "Climate Migration",
    "Aging Population",
    "Urbanisation Pressures",
    "Data Privacy Concerns",
    "Sustainable Fashion"
)

# Classification labels
CLASSIFICATION_LABELS = ["Local Problem", "New Initiatives"]


# API Keys (should be moved to secrets in production)
OPENCAGE_API_KEY = "e760785d8c7944888beefc24aa42eb66"

# NLTK resources
NLTK_RESOURCES = [
    'punkt',
    'stopwords',
    'wordnet',
    'vader_lexicon',
    'averaged_perceptron_tagger',
    'omw-1.4'
]