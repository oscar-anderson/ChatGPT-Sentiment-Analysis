'''
ChatGPT Twitter Sentiment Analysis

This script performs a sentiment analysis on a dataset of tweets related to
OpenAI's Large Language Model (LLM), ChatGPT. Using Natural Language Processing
(NLP), this script includes a series of functions for loading data,
preprocessing text, visualising the sentiment and themes within the data and
fitting the machine learning model to allow for the sentiment of further text
data to be predicted.

Author: Oscar Anderson
Date: 2024-01-11

Requirements:
- pandas
- matplotlib
- nltk
- scikit-learn
- wordcloud

Usage:
1. Install the required dependencies using: pip install pandas matplotlib nltk scikit-learn wordcloud.
2. Update the 'file_path' variable with the path to your CSV file containing tweet data.
3. Run the script to perform sentiment analysis and visualise the results.

'''
# Import dependencies.
import re
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# Define functions.
def load_data(file_path: str, index_col: str = None) -> pd.DataFrame:
    '''
    Load and display data from a CSV file into a pandas DataFrame.

    Input:
    - file_path (str): Path to the CSV file.

    Output:
    - pd.DataFrame: Loaded DataFrame.
    '''
    df = pd.read_csv(file_path, index_col = index_col)
    print(df.head())
    
    return df

def preprocess_text(df: pd.DataFrame, text_column: str, stop_words: set, lemmatiser: WordNetLemmatizer) -> pd.DataFrame:
    '''
    Preprocess tweet data in a DataFrame by converting to lowercase, removing
    URLs, emojis, hashtags, non-alphanumeric characters and stop words, and
    reducing words to their root form.

    Input:
    - df (pd.DataFrame): Input DataFrame.
    - text_column (str): Name of the column containing tweet data.
    - stop_words (set): Set of stop words to be removed.
    - lemmatiser: Lemmatizer object.

    Output:
    - pd.DataFrame: DataFrame with processed tweet data.
    '''
    processed_text = []
    
    for index, row in df.iterrows():
        text = row[text_column].lower()  # Convert all to lowercase.
        text = re.sub(r'http\S+', '', text) # Remove URLs.
        text = text.encode('ascii', 'ignore').decode('ascii') # Remove emojis.
        text = re.sub(r'#\w+', '', text) # Remove hashtags.
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # Remove non-alphanumeric characters.

        # Reduce individual words to root form and remove stop words.
        words = word_tokenize(text)
        words = [lemmatiser.lemmatize(word) for word in words if len(word) > 1 and word not in stop_words]

        processed_text.append(' '.join(words))

    df['processed_tweets'] = processed_text

    return df


def plot_sentiment_freq(labels_data: pd.Series) -> None:
    '''
    Plot the distribution of sentiment labels in the dataset.

    Input:
    - labels_data (pd.Series): Series containing sentiment labels.
    '''
    colours = ['lightcoral', 'lightgreen', 'skyblue']
    sentiment_frequency = labels_data.value_counts()

    ax = sentiment_frequency.plot(kind = 'bar', color = colours)
    plt.title('Sentiment Distribution Among Chat-GPT Related Tweets',
              fontweight = 'bold',
              pad = 10)
    plt.xlabel('Sentiment label', fontweight = 'bold')
    plt.ylabel('Frequency', fontweight = 'bold')
    plt.xticks(rotation = 45, ha = 'right')
    plt.yticks(range(0, max(sentiment_frequency) + 10000, 10000))
    
    for label, count in enumerate(sentiment_frequency):
        ax.annotate(str(count),
                    (label, count + 0.1),
                    ha = 'center',
                    va = 'bottom',
                    fontsize = 8)
    
    plt.grid(axis = 'y')
    plt.show()

def generate_word_cloud(text_data: pd.Series) -> None:
    '''
    Generate and display a word cloud based on the input text data.

    Input:
    - text_data (pd.Series): Series containing text data.
    '''
    text_data_string = ' '.join(text_data)
    wordcloud = WordCloud(width = 1000,
                          height = 500,
                          background_color = 'white',
                          collocations = False,
                          colormap = 'winter')
    wordcloud.generate(text_data_string)

    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Word Cloud of ChatGPT-Related Tweet Contents',
              fontweight = 'bold',
              pad = 7)
    plt.show()

def fit_model(df: pd.DataFrame, text_data_col: str, labels_col: str) -> None:
    '''
    Fit a Multinomial Naive Bayes model and display accuracy, confusion matrix, and classification report.

    Input:
    - df (pd.DataFrame): DataFrame containing the dataset.
    - text_data_col (str): Name of the column containing processed text data.
    - labels_col (str): Name of the column containing sentiment labels.
    '''
    vectoriser = CountVectorizer()
    X = vectoriser.fit_transform(df[text_data_col])
    x_train, x_test, y_train, y_test = train_test_split(X, df[labels_col], test_size = 0.2)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model, vectoriser

# Load and inspect data.
file_path = 'C:/Users/Oscar/Documents/Projects/chatGPT-sentiment-analysis/chatGPT_tweets.csv'
df = load_data(file_path, 'index')

# Pre-processing.
stop_words = set(stopwords.words('english'))
lemmatiser = WordNetLemmatizer()
tweet_data = df['tweets']
df = preprocess_text(df, 'tweets', stop_words, lemmatiser)

# Exploratory data analysis (EDA).
plot_sentiment_freq(df['labels'])
generate_word_cloud(df['processed_tweets'])

# Fit Multinomial Naive Bayes model.
model, vectoriser = fit_model(df, 'processed_tweets', 'labels')
