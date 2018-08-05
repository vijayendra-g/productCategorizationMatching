# productCategorizationMatching

### Background
The problem consists of 2 subproblems: 
1. product categorisation 
1. product matching. 
For each there is a separate set of datasets. The dataset can be downloaded following this link: http://im04.internetmemory.org/data/exensa/datasets.zip

this file contains two directories, each for one problem:

1. product categorisation/classification:
In this problem, what is sought is a method that will for a given set of features {productName, brand, description, price} will get a category from a category hierarchy that is known before hand - preferably with a confidence score of such a guess/classification. There are 2 datasets provided for this problem - the training data (train_data.csv) and the testing data (test_data_final.csv). The training data contains products represented by their features where among those is a 3 level categorization - Categorie1, Categorie2, Categorie3. In the testing data, the list of products is represented by their features. As a result we expect a file where for each product from test data file, will be on each line a pair of <Identifiant_Produit>;<Categorie3>.
1. product matching:
For this problem, you are given three data files - fnac.csv, darty.csv and matching_examples_final.csv. In the fnac and darty data fies, you find products represented by common features (brand, name, price, etc). The idea is to find a corresponding match beween products from darty dataset in the fnac dataset using the available features. The primary key in fnac is "brand" and "product_name". The primary key in darty is the url. The data file matching_examples.csv contains 1000 examples of matches from darty to fnac that can be used as a ground truth for training validation. The idea is to create a method that will try to match any given product to a product in fnac data set, and asses its confidence. As a result, we expect a file, where for each product from darty dataset, will be on each line fnac.brand, fnac.product_name, darty.url and confidence_score. The confidence score quantifies the algorithm's confidence in found match.




