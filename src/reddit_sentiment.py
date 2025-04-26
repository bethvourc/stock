import praw
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re


nltk.download('vader_lexicon')

load_dotenv()

class RedditSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.cache = {}  # Simple in-memory cache

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_sentiment_score(self, text):
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return 0
        score = self.sentiment_analyzer.polarity_scores(cleaned_text)
        return score['compound']  # VADER returns 'compound' score from -1 to 1

    def get_reddit_posts(self, stock_symbol, limit=100):
        if stock_symbol in self.cache:
            return self.cache[stock_symbol]

        posts = []
        try:
            for post in self.reddit.subreddit('stocks+investing+wallstreetbets').search(
                f'{stock_symbol} stock', limit=limit, time_filter='month'
            ):
                comments = []
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:7]:  # Pull top 7 comments
                        comments.append(comment.body)
                except Exception as e:
                    print(f"Error fetching comments: {str(e)}")

                full_text = f"{post.title} {post.selftext} {' '.join(comments)}"
                sentiment_score = self.get_sentiment_score(full_text)

                posts.append({
                    'title': post.title,
                    'text': post.selftext,
                    'comments': comments,
                    'score': post.score,
                    'sentiment': sentiment_score,
                    'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    'url': f'https://reddit.com{post.permalink}',
                    'subreddit': post.subreddit.display_name
                })

        except Exception as e:
            print(f"Error fetching Reddit posts: {str(e)}")

        posts_df = pd.DataFrame(posts)
        self.cache[stock_symbol] = posts_df
        return posts_df

    def analyze_sentiment(self, stock_symbol):
        posts_df = self.get_reddit_posts(stock_symbol)

        if len(posts_df) == 0:
            return {
                'success': False,
                'error': f'No Reddit posts found for {stock_symbol}'
            }

        # Weighted sentiment calculation
        posts_df['weighted_sentiment'] = posts_df['sentiment'] * posts_df['score']
        weighted_sum = posts_df['weighted_sentiment'].sum()
        total_score = posts_df['score'].sum()

        if total_score == 0:
            avg_sentiment = posts_df['sentiment'].mean()
        else:
            avg_sentiment = weighted_sum / total_score

        # Sentiment categories
        posts_df['sentiment_category'] = posts_df['sentiment'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )

        sentiment_counts = posts_df['sentiment_category'].value_counts().to_dict()

        top_posts = posts_df.nlargest(5, 'score').to_dict('records')

        return {
            'success': True,
            'average_sentiment': float(avg_sentiment),
            'post_count': len(posts_df),
            'sentiment_distribution': sentiment_counts,
            'top_posts': top_posts
        }
