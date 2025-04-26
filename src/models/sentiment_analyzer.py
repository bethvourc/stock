import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

nltk.download('punkt')
nltk.download('stopwords')

class EnhancedSentimentAnalyzerV2:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.classifier = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
        self.stop_words = set(stopwords.words('english'))
        self.quality_threshold = 0.5  

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def calculate_quality_score(self, post):
        score = 0
        if len(post['text']) > 100:
            score += 0.3
        if post['score'] > 10:
            score += 0.3
        sentiment_scores = self.sia.polarity_scores(post['text'])
        score += abs(sentiment_scores['compound']) * 0.4
        return score

    def filter_low_quality_posts(self, posts_df):
        posts_df = posts_df.copy()
        posts_df['quality_score'] = posts_df.apply(self.calculate_quality_score, axis=1)
        return posts_df[posts_df['quality_score'] > self.quality_threshold]

    def vectorize_text(self, texts):
        return self.embedding_model.encode(texts)

    def extract_additional_features(self, text):
        tokens = word_tokenize(text)
        pos_words = sum(1 for word in tokens if self.sia.polarity_scores(word)['compound'] > 0.3)
        neg_words = sum(1 for word in tokens if self.sia.polarity_scores(word)['compound'] < -0.3)
        sentences = sent_tokenize(text)
        first_sentiment = self.sia.polarity_scores(sentences[0])['compound'] if sentences else 0
        last_sentiment = self.sia.polarity_scores(sentences[-1])['compound'] if sentences else 0
        return pos_words, neg_words, first_sentiment, last_sentiment

    def analyze_sentiment(self, text):
        vader_sentiment = self.sia.polarity_scores(text)
        text_length = len(text)
        word_count = len(text.split())
        has_question = 1 if '?' in text else 0
        has_exclamation = 1 if '!' in text else 0
        pos_words, neg_words, first_sentiment, last_sentiment = self.extract_additional_features(text)
        
        sentiment_score = (
            vader_sentiment['compound'] * 0.5 +
            (first_sentiment + last_sentiment) * 0.1 +
            (pos_words - neg_words) * 0.05 +
            (has_exclamation * 0.1) +
            (has_question * -0.05)
        )

        confidence_score = (
            (text_length / 500) * 0.2 +
            (1 - abs(vader_sentiment['compound'])) * -0.8
        )

        return sentiment_score, max(min(confidence_score, 1.0), 0.0)  # Clamp between 0 and 1

    def analyze_trend(self, posts_df, window_size=7):
        if posts_df.empty:
            return {
                'trend': 'Neutral',
                'daily_sentiment': pd.Series(dtype=float),
                'moving_avg': pd.Series(dtype=float),
                'current_sentiment': 0
            }

        posts_df = posts_df.copy()
        posts_df['date'] = pd.to_datetime(posts_df['created_utc'])
        posts_df.set_index('date', inplace=True)
        
        daily_sentiment = posts_df['sentiment'].resample('D').mean()
        ema_sentiment = daily_sentiment.ewm(span=window_size, adjust=False).mean()

        trend = 'Neutral'
        if not ema_sentiment.empty:
            if ema_sentiment.iloc[-1] > 0.2:
                trend = 'Bullish'
            elif ema_sentiment.iloc[-1] < -0.2:
                trend = 'Bearish'
            elif -0.05 <= ema_sentiment.iloc[-1] <= 0.05:
                trend = 'Neutral'

        return {
            'trend': trend,
            'daily_sentiment': daily_sentiment,
            'moving_avg': ema_sentiment,
            'current_sentiment': ema_sentiment.iloc[-1] if not ema_sentiment.empty else 0
        }
