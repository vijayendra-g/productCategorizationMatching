import pandas as pd


def extract_data(fnac_path, darty_path, n):
    """
    :param fnac_path: fnac data path
    :param darty_path: darty data path
    :param n: select every N-th line in the file and ignore the rest
    :return: fnac_raw_dara, darty_raw_data
    """
    skip_idx = [x for x in range(1, sum(1 for l in open(fnac_path))) if x % n != 0]
    fnac_raw_data = pd.read_csv(fnac_path,
                                error_bad_lines=False,
                                delimiter=",",
                                skiprows=skip_idx,
                                encoding='utf-8',
                                names=['brand', 'product_name', 'category', 'subcategory', 'price']).drop(0, axis=0)

    skip_idx = [x for x in range(1, sum(1 for l in open(darty_path))) if x % n != 0]
    darty_raw_data = pd.read_csv(darty_path,
                                 error_bad_lines=False,
                                 delimiter=",",
                                 skiprows=skip_idx,
                                 encoding='utf-8',
                                 names=['url', 'brand', 'product_name', 'description', 'category', 'subcategory',
                                        'price']).drop(0, axis=0)

    # dealing with missing values
    darty_raw_data['price'].fillna(darty_raw_data['price'].mean, inplace=True)
    darty_raw_data.loc[:, ['brand', 'product_name', 'category', 'subcategory']] = darty_raw_data.loc[:,
                                                                                 ['brand', 'product_name', 'category',
                                                                                  'subcategory']].fillna('unknown')

    fnac_raw_data['price'].fillna(fnac_raw_data['price'].mean, inplace=True)
    fnac_raw_data.loc[:, ['brand', 'product_name', 'category', 'subcategory']] = fnac_raw_data.loc[:,
                                                                                 ['brand', 'product_name', 'category',
                                                                                  'subcategory']].fillna('unknown')

    return fnac_raw_data, darty_raw_data
