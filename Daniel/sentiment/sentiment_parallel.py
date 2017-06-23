"""
In order to run the sentiment analysis in parallel, we need to move this
method out of class, see jobblib documentation.
"""


def get_sentiment_for_sentence(text, sentiment_analyzer):
    """
    Returns the sentiment of a single sentence.
    :param text: [string], input string
    :param sentiment_analyzer: [obj], a vader instance. It's faster if we just
           pass it in and not initialize it for every sentence.
    :return: compound sentiment score: -1 is neg, 0 is neutral,
            1 is pos
    """
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']