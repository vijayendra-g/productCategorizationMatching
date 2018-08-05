from Extractor import extract_data
import pandas as pd
from sklearn.model_selection import train_test_split

from Predictor import level_predict, evaluate
from Preprocessor import train_test_vectorizing


def main():
    # Extacting
    data = extract_data("datasets/classification/train_data.csv", explore=False).drop(0, axis=0)

    # Splitting
    random_state = 8
    test_size = 0.33

    y_cat1 = data.loc[:, 'Categorie1'].values
    y_cat2 = data.loc[:, 'Categorie2'].values
    y_cat3 = data.loc[:, 'Categorie3'].values

    X_train, X_test, y1_train, y1_test = train_test_split(data, y_cat1, test_size=test_size, random_state=random_state)
    _, _, y2_train, y2_test = train_test_split(data, y_cat2, test_size=test_size, random_state=random_state)
    _, _, y3_train, y3_test = train_test_split(data, y_cat3, test_size=test_size, random_state=random_state)

    X_train, X_test = train_test_vectorizing(X_train, X_test)

    # level 1 prediction
    level1_predict = level_predict(X_train, X_test, y1_train)
    evaluate(level1_predict, y1_test)

    # level2 prediction
    level2_predict = level_predict(pd.concat([X_train, pd.get_dummies(y1_train)], ignore_index=True, axis=1),
                                   pd.concat([X_test, pd.get_dummies(level1_predict)], ignore_index=True, axis=1),
                                   y2_train)
    evaluate(level2_predict, y2_test)

    # level3 prediction
    level3_predections = level_predict(pd.concat([X_train, pd.get_dummies(y1_train), pd.get_dummies(y2_train)],
                                                 ignore_index=True,
                                                 axis=1),
                                       pd.concat(
                                           [X_test, pd.get_dummies(level1_predict), pd.get_dummies(level2_predict)],
                                           ignore_index=True,
                                           axis=1),
                                       y3_train)
    evaluate(level3_predections, y3_test)


if __name__ == '__main__':
    main()
