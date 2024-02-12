'''
Sentiment Analysis of Tweets Related to ChatGPT with Multinomial Naive Bayes.
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

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    '''
    df = pd.read_csv(file_path)
    index_col = index_col
    print(df.head())
    
    return df

def preprocess_data(data: str, stop_words: list, lemmatizer: WordNetLemmatizer) -> str:
    '''
    Preprocess tweet data by converting to lowercase, removing URLs, reducing
    words to root form, and removing non-alphanumeric/stop words.

    Parameters:
    - data (str): Input tweet data.
    - stop_words (list): List of stop words to be removed.
    - lemmatizer: Lemmatiser object.

    Returns:
    - str: Processed tweet data.
    '''
    data = data.lower() # Convert all to lowercase.
    data = re.sub(r'http\S+', '', data) # Remove any URLs.
    
    # Reduce individual words to root form and remove non-alphanumeric/stop words.
    words = word_tokenize(data)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    
    return ' '.join(words)

def explore_data(df: pd.DataFrame, text_data_col: str) -> None:
    text_data = df[text_data_col]
    colours = ['skyblue', 'lightcoral', 'lightgreen']
    text_data.value_counts().plot(kind = 'bar', color = colours) # Inspect sentiment frequencies.
    plt.title('Data Count for Each Sentiment Label')
    plt.xlabel('Sentiment label')
    plt.ylabel('Count')
    plt.grid()
    plt.xticks(rotation = 45, ha = 'right')
    plt.show()
    
    text_data_string = ' '.join(text_data)
    wordcloud = WordCloud(width = 1000, 
                          height = 500,
                          background_color = 'white',
                          collocations = False) # Generate word cloud.
    wordcloud.generate(text_data_string)
    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Word Cloud of Tweet Data')
    plt.show()
    
def fit_model(df: pd.DataFrame, text_data_col: str, labels_col: str) -> None:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_data_col])
    x_train, x_test, y_train, y_test = train_test_split(X, df[labels_col], test_size = 0.2)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Load and inspect data.
file_path = 'C:/Users/Oscar/Documents/Projects/chatGPT-sentiment-analysis/chatGPT_tweets.csv'
df = load_data(file_path, 'index')

# Pre-processing.
stop_words = stopwords.words('english') # Remove stop words.
lemmatizer = WordNetLemmatizer()
df['processed_tweets'] = df['tweets'].apply(lambda tweet: preprocess_data(tweet, stop_words, lemmatizer))

# Exploratory data analysis (EDA).
explore_data(df, 'processed_tweets')

# Fit Multinomial Naive Bayes model.
fit_model(df, 'processed_tweets', 'labels')

