from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def train_test_vectorizing(X_train, X_test):
    description_transformer = CountVectorizer()
    libelle_transformer = CountVectorizer()
    data_vectorized_description = pd.DataFrame(description_transformer.fit_transform(X_train['Description']).todense())
    data_vectorized_libelle = pd.DataFrame(libelle_transformer.fit_transform(X_train['Libelle']).todense())
    data_vectorized_description_test = pd.DataFrame(description_transformer.transform(X_test['Description']).todense())
    data_vectorized_libelle_test = pd.DataFrame(libelle_transformer.transform(X_test['Libelle']).todense())

    return pd.concat([data_vectorized_description, data_vectorized_libelle], ignore_index=True, axis=1), pd.concat(
        [data_vectorized_description_test, data_vectorized_libelle_test], ignore_index=True, axis=1)
