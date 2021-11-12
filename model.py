# %%
"""
# Problem Statement

  The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.
  
  Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
  
  With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.
  As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py

color = sns.color_palette()
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)
import plotly.tools as tls
import warnings

warnings.filterwarnings('ignore')
import re
import string
from textblob import TextBlob
from sklearn import metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from imblearn import over_sampling

from sklearn.model_selection import RandomizedSearchCV

from collections import Counter
import pickle
from pathlib import Path

# %%
"""
### Reading the Data Set
"""

# %%
df = pd.read_csv('sample30.csv')
pd.options.display.max_colwidth = 1000
df.head()

# %%
"""
## 1. Data Cleaning and Preprocessing
"""

# %%
df.shape

# %%
df.dtypes

# %%
df.isnull().sum()

# %%
# Replacing Null values with No Title"
df["reviews_title"].fillna("No Title", inplace=True)

# %%
"""
#### Concatenate "Review Text" and "Review Title" Columns for better analysis
"""

# %%
df["Reviews_Text_and_Title"] = df["reviews_text"] + " " + df["reviews_title"]

# %%
## Checking null values
df.isnull().sum()

# %%
"""
#### Only id,name,reviews_rating,reviews_username,user_sentiment and Reviews_Text_and_Title columns require for analysis
"""

# %%
df = df[['id', 'name', 'reviews_rating', 'reviews_username', 'user_sentiment', 'Reviews_Text_and_Title']]

# %%
df.head()

# %%
# Checking Null values
df.isnull().sum()

# %%
# Dropping Null rows as they are less than 1%
df.dropna(subset=['user_sentiment', 'reviews_username'], inplace=True)

# %%
# Checking Null values
df.isnull().sum()

# %%
df.shape

# %%
sns.countplot(data=df, x='reviews_rating')

# %%
df['user_sentiment'].value_counts()

# %%
# Plotting User sentiments against the 5 ratings 

sns.countplot(df.reviews_rating, hue=df.user_sentiment)
plt.show()

# %%
"""
## 2. Text Processing
"""

# %%
pd.options.display.max_colwidth = 1000
df.head()

# %%
## Unique Products
df['name'].unique()

# %%
df.user_sentiment.value_counts(normalize=True)

# %%
# Mapping Negative to 0 and Positive to 1 

df['user_sentiment'] = df.user_sentiment.apply(lambda x: 1 if x == 'Positive' else 0)

# %%
"""
#### Expand Contractions Dictionary of English Contractions
Contractions are the shortened versions of words like don’t for do not and how’ll for how will. These are used to reduce the speaking and writing time of words. We need to expand these contractions for a better analysis of the reviews.
"""

# %%

contractions_dict = {"ain't": "are not", "'s": " is", "aren't": "are not",
                     "can't": "cannot", "can't've": "cannot have",
                     "'cause": "because", "could've": "could have", "couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
                     "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                     "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                     "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                     "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not",
                     "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                     "it'll've": "it will have", "let's": "let us", "ma'am": "madam",
                     "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                     "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                     "that'd": "that would", "that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have", "they'll": "they will",
                     "they'll've": "they will have", "they're": "they are", "they've": "they have",
                     "to've": "to have", "wasn't": "was not", "we'd": "we would",
                     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                     "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                     "what'll've": "what will have", "what're": "what are", "what've": "what have",
                     "when've": "when have", "where'd": "where did", "where've": "where have",
                     "who'll": "who will", "who'll've": "who will have", "who've": "who have",
                     "why've": "why have", "will've": "will have", "won't": "will not",
                     "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                     "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have", "y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                     "you'll": "you will", "you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


# Function for expanding contractions
def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


# Expanding Contractions in the reviews
df['Reviews_Text_and_Title'] = df['Reviews_Text_and_Title'].apply(lambda x: expand_contractions(x))

# %%
"""
#### Lowercase the reviews
"""

# %%
df['cleaned'] = df['Reviews_Text_and_Title'].apply(lambda x: x.lower())

# %%
"""
#### Remove digits and words containing digits
"""

# %%
df['cleaned'] = df['cleaned'].apply(lambda x: re.sub('\w*\d\w*', '', x))

# %%
"""
#### Remove Punctuations
"""

# %%
df['cleaned'] = df['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))

# %%
"""
#### Removing extra spaces
"""

# %%
df['cleaned'] = df['cleaned'].apply(lambda x: re.sub(' +', ' ', x))

# %%
## This is how text looks after cleaning
for index, text in enumerate(df['cleaned'][35:40]):
    print('Review %d:\n' % (index + 1), text)

# %%
"""
### Preparing Text Data for Exploratory Data Analysis (EDA)
"""

# %%
"""
#### Stopwords Removal  and lemmatizing the reviews
"""

# %%
# Importing spacy
import spacy

# Loading model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Lemmatization with stopwords removal
df['lemmatized'] = df['cleaned'].apply(
    lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop == False)]))

# %%
"""
#### group them according to the products:
"""

# %%
df_grouped = df[['name', 'lemmatized']].groupby(by='name').agg(lambda x: ' '.join(x))
df_grouped.head()

# %%
"""
#### Document Term Matrix
"""

# %%
# Creating Document Term Matrix
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer='word')
data = cv.fit_transform(df_grouped['lemmatized'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index = df_grouped.index
df_dtm.head()

# %%
"""
#### Word cloud for each product to check common words
"""

# %%
# Importing wordcloud for plotting word clouds and textwrap for wrapping longer text
from wordcloud import WordCloud
from textwrap import wrap


# Function for generating word clouds
def generate_wordcloud(data, title):
    wc = WordCloud(width=400, height=330, max_words=150, colormap="Dark2").generate_from_frequencies(data)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('\n'.join(wrap(title, 60)), fontsize=13)
    plt.show()


# Transposing document term matrix
df_dtm = df_dtm.transpose()

# Plotting word cloud for each product
for index, product in enumerate(df_dtm.columns):
    generate_wordcloud(df_dtm[product].sort_values(ascending=False), product)

# %%
"""
#### Now checking the polarity, i.e., how much a text is positive or negative, is sufficient
"""

# %%
# Polarity
from textblob import TextBlob

df['polarity'] = df['lemmatized'].apply(lambda x: TextBlob(x).sentiment.polarity)

# %%
print("3 Random Reviews with Highest Polarity:")
for index, review in enumerate(
        df.iloc[df['polarity'].sort_values(ascending=False)[:3].index]['Reviews_Text_and_Title']):
    print('Review {}:\n'.format(index + 1), review)

# %%
print("3 Random Reviews with Lowest Polarity:")
for index, review in enumerate(df.iloc[df['polarity'].sort_values(ascending=True)[:3].index]['Reviews_Text_and_Title']):
    print('Review {}:\n'.format(index + 1), review)

# %%
"""
#### plot polarities of reviews for each product and compare them
"""

# %%
product_polarity_sorted = pd.DataFrame(df.groupby('name')['polarity'].mean().sort_values(ascending=True))

plt.figure(figsize=(16, 8))
plt.xlabel('Polarity')
plt.ylabel('Products')
plt.title('Polarity of Different Amazon Product Reviews')
polarity_graph = plt.barh(np.arange(len(product_polarity_sorted.index)), product_polarity_sorted['polarity'],
                          color='purple', )

# Writing product names on bar
for bar, product in zip(polarity_graph, product_polarity_sorted.index):
    plt.text(0.005, bar.get_y() + bar.get_width(), '{}'.format(product), va='center', fontsize=11, color='white')

# Writing polarity values on graph
for bar, polarity in zip(polarity_graph, product_polarity_sorted['polarity']):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_width(), '%.3f' % polarity, va='center', fontsize=11,
             color='black')

plt.yticks([])
plt.show()

# %%
df.head()

# %%
## Dropping the columns which are not required
df_latest = df.drop(columns=['Reviews_Text_and_Title', 'cleaned', 'polarity'])

# %%
df_latest.head()

# %%
df_latest.columns = ['id', 'name', 'reviews_rating', 'reviews_username', 'user_sentiment', 'Reviews_Text_and_Title']

# %%
df_latest.shape

# %%
df_latest.head()

# %%
# Save the preprocessed data for future use
pickle.dump(df_latest, open('processed_data1.pkl', 'wb'))

# %%
"""
## 3. Feature Extraction
"""

# %%
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

x = df_latest['Reviews_Text_and_Title']
y = df_latest['user_sentiment']

# %%
"""
#### TF-IDF Vectorization
"""

# %%
"""
#### Word Vectorization
"""

# %%
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 3))

# %%
"""
## 4. Model Building
"""

# %%
"""
#### Train and Test Split
"""

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# %%
# Shape of X_Train Y_Train
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

# %%
"""
#### Clearly, a case of class imbalance. Hence,applying Random Over Sampler for handling this class imbalance
"""

# %%
counter = Counter(y_train)
print("Before: ", counter)

sampling = over_sampling.RandomOverSampler(random_state=0)
X_train, y_train = sampling.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))
X_train = pd.DataFrame(X_train).iloc[:, 0].tolist()

counter = Counter(y_train)
print("After: ", counter)

# %%
X_train_transformed = word_vectorizer.fit_transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())

# %%
"""
#### Functions for dispaly score,confusion matrix,calculating Sensitivity and Specificity
"""


# %%
# Function to display 
def display_score(classifier):
    cm = confusion_matrix(y_test, classifier.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    print(classifier)
    print('Accuracy is ', accuracy_score(y_test, classifier.predict(X_test)))
    print('Sensitivity is {}'.format(cm[1][1] / sum(cm[1])))
    print('Specificity is {}'.format(cm[0][0] / sum(cm[0])))


# %%

# create a function for plotting confusion matrix

def cm_plot(cm_train, cm_test):
    print("Confusion matrix ")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_train / np.sum(cm_train), annot=True, fmt=' .2%', cmap="Greens")
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_test / np.sum(cm_test), annot=True, fmt=' .2%', cmap="Greens")
    plt.show()


# %%

# create a function for calculating Sensitivity and Specificity
def spec_sensitivity(cm_train, cm_test):
    # Train
    tn, fp, fn, tp = cm_train.ravel()
    specificity_train = tn / (tn + fp)
    sensitivity_train = tp / float(fn + tp)

    print("sensitivity for train set: ", sensitivity_train)
    print("specificity for train set: ", specificity_train)
    print("\n****\n")

    # Test
    tn, fp, fn, tp = cm_test.ravel()
    specificity_test = tn / (tn + fp)
    sensitivity_test = tp / float(fn + tp)

    print("sensitivity for test set: ", sensitivity_test)
    print("specificity for test set: ", specificity_test)


# %%
"""
### Logistic Regression 
"""

# %%
logit = LogisticRegression()

logit.fit(X_train_transformed, y_train)

# %%
y_train_pred_logit = logit.predict(X_train_transformed)

print("Logistic Regression train accuracy ", accuracy_score(y_train_pred_logit, y_train), "\n")
print(classification_report(y_train_pred_logit, y_train))

# %%
y_test_pred_logit = logit.predict(X_test_transformed)

print("Logistic Regression accuracy on test data", accuracy_score(y_test_pred_logit, y_test), "\n")
print(classification_report(y_test_pred_logit, y_test))

# %%
cm_train = metrics.confusion_matrix(y_train, y_train_pred_logit)
cm_test = metrics.confusion_matrix(y_test, y_test_pred_logit)

# %%
spec_sensitivity(cm_train, cm_test)

# %%
cm_plot(cm_train, cm_test)

# %%
"""
### Random Forest Calssifier 
"""

# %%
rf = RandomForestClassifier(n_estimators=50, random_state=101, n_jobs=-1)

rf.fit(X_train_transformed, y_train)

# %%
y_train_pred_rf = rf.predict(X_train_transformed)

print("Random Forest Classifier train accuracy ", accuracy_score(y_train_pred_rf, y_train), "\n")
print(classification_report(y_train_pred_rf, y_train))

# %%
y_test_pred_rf = rf.predict(X_test_transformed)

print("Random Forest Classifier accuracy on test data", accuracy_score(y_test_pred_rf, y_test), "\n")
print(classification_report(y_test_pred_rf, y_test))

# %%
cm_train_rf = metrics.confusion_matrix(y_train, y_train_pred_rf)
cm_test_rf = metrics.confusion_matrix(y_test, y_test_pred_rf)

# %%
spec_sensitivity(cm_train, cm_test)

# %%
cm_plot(cm_train_rf, cm_test_rf)

# %%
"""
## XG Boost 
"""

# %%
import xgboost as xgb

xgb = xgb.XGBClassifier()

xgb.fit(X_train_transformed, y_train)

# %%
y_train_pred_xgb = xgb.predict(X_train_transformed)

print("XG Boost train accuracy ", accuracy_score(y_train_pred_xgb, y_train), "\n")
print(classification_report(y_train_pred_xgb, y_train))

# %%
y_test_pred_xgb = xgb.predict(X_test_transformed)

print("XG Boost accuracy on test data", accuracy_score(y_test_pred_xgb, y_test), "\n")
print(classification_report(y_test_pred_xgb, y_test))

# %%
cm_train_xgb = metrics.confusion_matrix(y_train, y_train_pred_xgb)
cm_test_xgb = metrics.confusion_matrix(y_test, y_test_pred_xgb)

# %%
spec_sensitivity(cm_train_xgb, cm_test_xgb)

# %%
cm_plot(cm_train_xgb, cm_test_xgb)

# %%
"""
## Naive Bayes 
"""

# %%
nb = MultinomialNB()
nb.fit(X_train_transformed, y_train)

# %%
y_train_pred_nb = nb.predict(X_train_transformed)

print("Naive Bayes train accuracy ", accuracy_score(y_train_pred_nb, y_train), "\n")
print(classification_report(y_train_pred_nb, y_train))

# %%
y_test_pred_nb = nb.predict(X_test_transformed)

print("Naive Bayes accuracy on test data", accuracy_score(y_test_pred_nb, y_test), "\n")
print(classification_report(y_test_pred_nb, y_test))

# %%
cm_train_nb = metrics.confusion_matrix(y_train, y_train_pred_nb)
cm_test_nb = metrics.confusion_matrix(y_test, y_test_pred_nb)

# %%
spec_sensitivity(cm_train_nb, cm_test_nb)

# %%
cm_plot(cm_train_nb, cm_test_nb)

# %%
"""
## From the Results above the Logistic regression shows a better F1 score for macro Avg , Hence selecting Logistic regression
"""

# %%
df_latest.columns

# %%
"""
## 5. Building The Recommendation System
"""

# %%
"""
#### Columns identified for recommendation system
#### reviews_username,id,reviews_rating
"""

# %%
recommendation_df = df_latest[["name", "reviews_username", "reviews_rating"]]

recommendation_df.head()

# %%
recommendation_df.shape

# %%
recommendation_df.dtypes

# %%
recommendation_df.isnull().sum()

# %%
"""
#### Dividing the dataset into train and test
"""

# %%
# Test and Train split of the dataset.
from sklearn.model_selection import train_test_split

train, test = train_test_split(recommendation_df, test_size=0.30, random_state=101)

# %%
print(train.shape)
print(test.shape)

# %%
train.head()

# %%
"""
### USER - USER Based Recommendation System
"""

# %%
# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user names.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(0)

# %%
df_pivot.head()

# %%
"""
#### Creating dummy train & dummy test dataset
These dataset will be used for prediction 
- Dummy train will be used later for prediction of the movies which has not been rated by the user. To ignore the movies rated by the user, we will mark it as 0 during prediction. The movies not rated by user is marked as 1 for prediction in dummy train dataset. 

- Dummy test will be used for evaluation. To evaluate, we will only make prediction on the movies rated by the user. So, this is marked as 1. This is just opposite of dummy_train.
"""

# %%
# Copy the train dataset into dummy_train
dummy_train = train.copy()

# %%
# The movies not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)

# %%
# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)

# %%
dummy_train.head()

# %%
"""
 #### Cosine Similarity
"""

# %%
from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

# %%
user_correlation.shape

# %%
"""
#### Using adjusted Cosine
#### Here, we are not removing the NaN values and calculating the mean only for the movies rated by the use
"""

# %%
# Create a user-movie matrix.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)

# %%
df_pivot.head()

# %%
"""
#### Normalising the rating of the product for each user around 0 mean
"""

# %%
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T - mean).T

# %%
df_subtracted.head()

# %%
"""
#### Finding cosine similarity
"""

# %%
from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

# %%
"""
#### Prediction - User User
"""

# %%
"""
Going the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0
"""

# %%
user_correlation[user_correlation < 0] = 0
user_correlation

# %%
"""
Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset).

"""

# %%
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

# %%
user_predicted_ratings

# %%
user_predicted_ratings.shape

# %%
"""
Since we are interested only in the products not rated by the user, we will ignore the products rated by the user by making it zero.
"""

# %%
user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
user_final_rating.head()

# %%
"""
  #### Evaluation - User User
"""

# %%
"""
Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the product already rated by the user insead of predicting it for the product not rated by the user.
"""

# %%
# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape

# %%
common.head()

# %%
# convert into the user-movie matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username',
                                              columns='name',
                                              values='reviews_rating')

# %%
# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

# %%
df_subtracted.head(1)

# %%
user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username', inplace=True)
user_correlation_df.head()

# %%
common.head(1)

# %%
list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()

user_correlation_df_1 = user_correlation_df[user_correlation_df.index.isin(list_name)]

# %%
user_correlation_df_1.shape

# %%
user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T

# %%
user_correlation_df_3.head()

# %%
user_correlation_df_3.shape

# %%
user_correlation_df_3[user_correlation_df_3 < 0] = 0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings

# %%
dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x >= 1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username',
                                    columns='name',
                                    values='reviews_rating').fillna(0)

# %%
dummy_test.shape

# %%
common_user_predicted_ratings = np.multiply(common_user_predicted_ratings, dummy_test)

# %%
common_user_predicted_ratings.head(2)

# %%
"""
#### Calculating the RMSE for only the Products rated by user. For RMSE, normalising the rating to (1,5) range
"""

# %%
from sklearn.preprocessing import MinMaxScaler
from numpy import *

X = common_user_predicted_ratings.copy()
X = X[X > 0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)

# %%
common_ = common.pivot_table(index='reviews_username',
                             columns='name',
                             values='reviews_rating')

# %%
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

# %%
user_rmse = (sum(sum((common_ - y) ** 2)) / total_non_nan) ** 0.5
print(user_rmse)

# %%
"""
 ### ITEM -  ITEM Based Similarity

Taking the transpose of the rating matrix to normalize the rating around the mean for different Product Id ID. In the user based similarity, we had taken mean for each user instead of each Product.
"""

# %%
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).T

df_pivot.head()

# %%
"""
#### Normalising the Review rating for each Product for using the Adujsted Cosin
"""

# %%
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T - mean).T

# %%
df_subtracted.head()

# %%
"""
#### Finding the cosine similarity using pairwise distances approach
"""

# %%
from sklearn.metrics.pairwise import pairwise_distances

# %%
# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)

# %%
"""
#### Filtering the correlation only for which the value is greater than 0. (Positively correlated)
"""

# %%
item_correlation[item_correlation < 0] = 0
item_correlation

# %%
"""
#### Prediction Item-Item
"""

# %%
item_predicted_ratings = np.dot((df_pivot.fillna(0).T), item_correlation)
item_predicted_ratings

# %%
item_predicted_ratings.shape

# %%
dummy_train.shape

# %%
"""
#### Filtering the rating only for the Products not rated by the user for recommendation
"""

# %%
item_final_rating = np.multiply(item_predicted_ratings, dummy_train)
item_final_rating.head()

# %%
"""
#### Evaluation - Item Item
"""

# %%
"""
Evaluation will we same as you have seen above for the prediction. The only difference being, you will evaluate for the Product already rated by the user insead of predicting it for the Product not rated by the user.
"""

# %%
test.columns

# %%
common = test[test.name.isin(train.name)]
common.shape

# %%
common.head()

# %%
common_item_based_matrix = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T

# %%
common_item_based_matrix.shape

# %%
item_correlation_df = pd.DataFrame(item_correlation)

# %%
item_correlation_df.head()

# %%
item_correlation_df['name'] = df_subtracted.index
item_correlation_df.set_index('name', inplace=True)
item_correlation_df.head()

# %%
list_name = common.name.tolist()

# %%
item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 = item_correlation_df[item_correlation_df.index.isin(list_name)]

# %%
item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T

# %%
item_correlation_df_3.head()

# %%
item_correlation_df_3[item_correlation_df_3 < 0] = 0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings

# %%
common_item_predicted_ratings.shape

# %%
"""
Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train
"""

# %%
dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x >= 1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings, dummy_test)

# %%
"""
The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.
"""

# %%
common = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T

# %%
from sklearn.preprocessing import MinMaxScaler
from numpy import *

X = common_item_predicted_ratings.copy()
X = X[X > 0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)

# %%
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

# %%
item_rmse = (sum(sum((common - y) ** 2)) / total_non_nan) ** 0.5

# %%
print("ITEM Based Root Mean Square Error: ", item_rmse)

# %%
print("USER Based Root Mean Square Error: ", user_rmse)

# %%
"""
### Since User Based Recomendation System has less RMSE hence choosing User Based Recommendation system
"""

# %%
"""
## 6. Recommendation of Top 20 Products to a Specified User
"""

# %%
# Take the user ID as input.
user_input = "anne"

# %%
print(user_input)

# %%
user_final_rating.head(2)

# %%
d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d

# %%
"""
## 7. Fine-Tuning the Recommendation System and Recommendation of Top 5 Products
"""

# %%
# save the respective files and models through Pickle 
import pickle

pickle.dump(logit, open('logistic_model.pkl', 'wb'))
# loading pickle object
logit = pickle.load(open('logistic_model.pkl', 'rb'))

pickle.dump(word_vectorizer, open('word_vectorizer.pkl', 'wb'))
# loading pickle object
word_vectorizer = pickle.load(open('word_vectorizer.pkl', 'rb'))

pickle.dump(user_final_rating, open('user_predicted_ratings.pkl', 'wb'))
user_predicted_ratings = pickle.load(open('user_predicted_ratings.pkl', 'rb'))

# %%
"""
#### Function for Top 5 Recommended Products
"""


# %%
def top_5_recommendation(user_input):
    arr = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i = 0
    a = {}
    for prod_name in arr.index.tolist():
        product = prod_name
        product_name_review_list = df_latest[df_latest['name'] == product]['Reviews_Text_and_Title'].tolist()
        features = word_vectorizer.transform(product_name_review_list)
        logit.predict(features)
        a[product] = logit.predict(features).mean() * 100
    b = pd.Series(a).sort_values(ascending=False).head(5).index.tolist()
    print("Enter Username : ", user_input)
    print("Five Recommendations for you :")
    for i, val in enumerate(b):
        print(i + 1, val)


# %%
top_5_recommendation(user_input)

# %%
df_latest.to_csv("final_data.csv", index=False)

# %%
