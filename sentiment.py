from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)

    # Check the compound score for overall sentiment
    if sentiment_score['compound'] < -0.1:  # Make it more sensitive to negative sentiment
        return "Negative"
    elif sentiment_score['compound'] > 0.1:
        return "Positive"
    else:
        # Consider a more nuanced check for specific concerns
        negative_keywords = ["concern", "pollution", "health", "deteriorating", "respiratory", "smog"]
        if any(keyword in text.lower() for keyword in negative_keywords):
            return "Negative"
        return "Neutral"

# Sample Text
text1 = """
Dear Sir/Madam, I am writing to express my deep concern regarding the deteriorating air quality in Birmingham. 
Over the past few months, the city has experienced a significant increase in smog and industrial emissions, 
which I believe is severely affecting public health. I have observed that the polluted air is causing respiratory 
issues among residents, particularly vulnerable groups such as the elderly and children. I urge the council to undertake 
immediate measures, including stricter emissions controls and increased monitoring, to mitigate this hazardous situation. 
Your prompt action in addressing this matter would be greatly appreciated.
"""

text = """
Dear Sir/Madam, I am deeply disappointed and alarmed by the worsening air quality in Birmingham. In recent months, the city has seen a concerning rise in smog and industrial emissions, which are undoubtedly putting public health at great risk. The polluted air is clearly contributing to respiratory problems, particularly among vulnerable groups like the elderly and children. The council’s failure to effectively address this issue is unacceptable. Immediate action is required, including stricter emissions controls and better monitoring, to prevent further harm. It is extremely frustrating that this matter has not been prioritized.
"""

# Analyze sentiment
print(analyze_sentiment(text))
