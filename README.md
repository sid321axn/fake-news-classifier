# Fake News Classification using NLP based Machine Learning & Deep Learning (BERT)

## Project Overview

Fake news is a form of news consisting of deliberate disinformation or hoaxes spread via traditional news media or online social media.In this project, I have used different natural language processing (NLP) based machine learning and deep learning approaches including BERT to detect fake news from news headlines. Generally, a fake headline is a news headline which may read one way or state something as fact, but then the body of the article says something different. The Internet term for this type of misleading fake news is **“clickbait”** —headlines that catch a reader’s attention to make them click on the fake news. This type of fake news is misleading at best and untrue at worst.

![Screenshot Predict](https://media.giphy.com/media/xT4ApznCZTtuBjmHHa/giphy.gif)

In this project, I have extracted interesting patterns from the headline text using NLP and perform exploratory data analysis to provide useful insight about fake headlines by creating intuitive features. This project includes work detailed below:

- Exploratory data analysis & Feature Engineering using NLP
- Machine Learning Modeling using text based features
- Deep Learning Modeling (LSTM) using text based features
- BERT model building from tex based features

## About Data

The dataset used in this project is the [ISOT Fake News Dataset](https://drive.google.com/open?id=1IoTRrJNDJqvaG3hnUpnHQyGvPAJbO8y3).The dataset contains two types of articles fake and real News. This dataset was collected from realworld sources; the truthful articles were obtained by crawling articles from Reuters.com (News website). As for the fake news articles, they were collected from different sources. The fake news articles were collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. The dataset contains different types of articles on different topics, however, the majority of articles focus on political and World news topics.

The dataset consists of two CSV files. The first file named ```True.csv``` contains more than ***12,600*** articles from reuter.com. The second file named ```Fake.csv``` contains more than ***12,600*** articles from different fake news outlet resources. Each article contains the following information: 

- **article title (News Headline)**, 
- **text**,
- **type (REAL or FAKE)**
- **the date the article was published on***

## Word Cloud of dataset

In wordcloud most frequent occuring words in the corpus to be shown. The size of words in the wordcloud is based on their frequency in the corpus.The more the word appears, the largers the word font will be. As we can see from the wordclouds most frequent words in fake news are Video,Obama, Hillary, Trump and Republican whereas Real news comprise Trump, White House, North Korea, China etc.

![Screenshot Predict](https://i.ibb.co/NpMZzcq/wc.png)


## Results Achieved
Following results achieved in this project using different modeling approaches.


| Model  | Accuracy | Precision | Recall | F1- Score |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Multinomial Naive Baye's TFIDF (Bi-gram) | 94.10% | 91.95%% | 97.17% | 94.48% |
| Passive Aggressive Classifier TFID (Bi-gram) | 95.9%  | 95.53% | 96.73% | 96.12% |
| Logistic Regression TFID (Bi-gram) | 94.6%  | 94.63%% | 94.95% | 94.78% | 
| LSTM with GLOVE embedding | 94.69%  | 95.00%% | 95.00% | 95.00% |
| BERT (1 epoch) | **98.43%**  | **98.00%** | **98.00%** | **98.00%** |



## Installations:

This project requires Python 3.7x and the following Python libraries should be installed to get the project started:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Seaborn](https://seaborn.pydata.org/)
- [KTrain (for BERT)](https://pypi.org/project/ktrain/)
- [NLTK](https://www.nltk.org/install.html)
- [NLTK Data](https://www.nltk.org/data.html)
- [Keras](https://keras.io/)

I also reccommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project which also include jupyter notebook to run and execute [IPython Notebook](http://ipython.org/notebook.html).

# Run :
In a terminal or command window, navigate to the top-level project directory fake-news-classifier/ (that contains this README) and run one of the following commands:

```ipython notebook Fake_news_preprocessing.ipynb```

or

```ipython notebook fake news Analysis.ipynb```

or

```ipython notebook fake news headline LSTM.ipynb```

or

```ipython notebook fake_news_classification_machine_learning_approach.ipynb```

or

```ipython notebook fake_news_classification_using_BERT.ipynb```

This will open the Jupyter Notebook software and project file in your browser.



