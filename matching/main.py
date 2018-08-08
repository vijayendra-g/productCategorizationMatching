import os

from Extractor import extract_data
from matching.Preprocessor import preprocess_data, vectorize_data


def main():
    # Extract data
    os.chdir(r'../datasets')
    fnac_path = "product_matching/fnac_sample.csv"
    darty_path = "product_matching/darty_sample.csv"
    ground_trouth = "matching_examples_final"

    fnac_raw_data, darty_raw_data = extract_data(fnac_path, darty_path, 1)

    # save primary keys
    fnac_primary_key = (fnac_raw_data.loc[:, 'brand'], fnac_raw_data.loc[:, 'product_name'])
    darty_primary_key = darty_raw_data.loc[:, 'url']
    darty_raw_data = darty_raw_data.drop(['url'], axis=1)

    # Pre process data
    preprocess_data(fnac_raw_data, ['product_name', 'category', 'subcategory'])
    preprocess_data(darty_raw_data, ['product_name', 'category', 'subcategory', 'description'])

    fnac_raw_data['description'] = (
        fnac_raw_data[['brand', 'product_name', 'category', 'subcategory']].apply(lambda x: ' '.join(x), axis=1))

    fnac_preprocessed_data, transformers = vectorize_data(fnac_raw_data,
                                                          ['brand', 'product_name', 'category', 'subcategory',
                                                           'description'])
    darty_preprocessed_data, _ = vectorize_data(darty_raw_data,
                                                ['brand', 'product_name', 'category', 'subcategory', 'description'],
                                                transformers=transformers)
    # free up memory
    del fnac_raw_data, darty_raw_data


if __name__ == '__main__':
    main()
