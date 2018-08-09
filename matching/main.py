import os
import csv
import pandas as pd

from Extractor import extract_data
from matching.Preprocessor import preprocess_data, vectorize_data
from scipy.spatial.distance import cosine


def calculate_area(row, fnac_vector):
    row = pd.to_numeric(row.drop(row.index[0]), errors='ignore')
    fnac_vector = pd.to_numeric(fnac_vector.drop(fnac_vector[0]), errors='ignore')
    return 1 - cosine(row, fnac_vector)


def main():
    # Extract data
    os.chdir(r'../datasets')
    fnac_path = "product_matching/fnac_sample.csv"
    darty_path = "product_matching/darty_sample.csv"
    ground_trouth = "product_matching/matching_examples_final_samples.csv"
    matching_result = "product_matching/matching_result.csv"

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

    # compute similarity
    with open(ground_trouth, 'rb') as ground_trouth, open(matching_result, 'wb') as matching_result:
        ground_trouth_reader = csv.reader(ground_trouth)

        next(ground_trouth_reader, None)  # skip the header
        for row in ground_trouth_reader:
            fnac_brand = row[0]
            fnac_product_name = row[1]

            index = 0
            with open(fnac_path, 'rb') as fnac_data:
                fnac_data_reader = csv.reader(fnac_data)
                next(fnac_data_reader, None)
                for fnac_row in fnac_data_reader:
                    brand = fnac_row[0]
                    product_name = fnac_row[1]
                    if fnac_brand == brand and product_name == fnac_product_name:
                        break
                    index += 1
            fnac_vector = fnac_preprocessed_data.iloc[index, :]
            similarities = darty_preprocessed_data.apply(
                lambda x: calculate_area(x, fnac_vector), axis=1)

            max = similarities[similarities == similarities.max()]
            if max.size != 0:
                darty_url = darty_primary_key.iloc[max.index[0], 0]

            else:
                darty_url = 'None'

            matching_result.write(
                ",".join([fnac_brand, fnac_product_name, darty_url]))


if __name__ == '__main__':
    main()
