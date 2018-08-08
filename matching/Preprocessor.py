import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import sys
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

stop = stopwords.words('french')
tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()


def preprocess_data(df, column_names):
    """
    stop word, punctuation removal, lower case and lemmatization
    :param df: dataframe to transform
    :param column_names: list of column names to preprocess
    :return:
    """
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
        df[column_name] = df[column_name].apply(lambda x: ' '.join([item for item in x.split(' ') if item not in stop]))
        df[column_name] = df[column_name].apply(
            lambda text: ' '.join([lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]))


def vectorize_data(df, column_names, transformers=None):
    all_vectorized_columns = None

    if not transformers:
        transformers = []
        for column_name in column_names:
            transformer = CountVectorizer()
            transformers.append(transformer)
            vectorized_column = pd.DataFrame(transformer.fit_transform(df[column_name]).todense())
            df = df.drop([column_name], axis=1)
            all_vectorized_columns = pd.concat([all_vectorized_columns, vectorized_column], ignore_index=True, axis=1)
    else:
        for column_name, transformer in zip(column_names, transformers):
            vectorized_column = pd.DataFrame(transformer.transform(df[column_name]).todense())
            df = df.drop([column_name], axis=1)
            all_vectorized_columns = pd.concat([all_vectorized_columns, vectorized_column], ignore_index=True, axis=1)
    return pd.concat([df, all_vectorized_columns], axis=1), transformers
