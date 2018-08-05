import pandas as pd
from IPython.core.display import display


def extract_data(path, n, explore=False):
    """
    :param path: data path
    :param n: select every N-th line in the file and ignore the rest
    :param explore: explore data to get some statistics
    :return:
    """
    skip_idx = [x for x in range(1, sum(1 for l in open(path))) if x % n != 0]
    raw_data = pd.read_csv(path,
                           error_bad_lines=False,
                           delimiter=";",
                           skiprows=skip_idx,
                           index_col="Identifiant_Produit")
    if explore:
        # Explore data
        pd.set_option('display.max_columns', 20)
        raw_data_exp = raw_data

        # drop columns with std =< 0.2
        mask = (raw_data_exp.std() >= .0).values
        raw_data_exp = raw_data_exp.loc[:, mask]

        # transform continuous values
        for column in raw_data_exp.columns:
            display(pd.crosstab(index=raw_data_exp[column], columns='% observations', normalize='columns'))
        # Results:

        # Categorie1 : 48 items
        # Categorie2 : 514 items
        # Categorie3 : 5609 items

        # price :  std <0.2
        # Description / Marque / libelle : text
        # 95% don't have discount
    return raw_data.drop(['Produit_Cdiscount', 'prix', 'Marque'], axis=1)
