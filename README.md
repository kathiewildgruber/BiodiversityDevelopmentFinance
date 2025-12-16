# BiodiversityDevelopmentFinance
Supplementary code for Wildgruber et al. (2026): A multi-dimensional view to map biodiversity development finance globally. Currently under review in Nature Sustainability.

# Setup
Tu run the code, use Python 3.10 or higher. The code has been developed and tested on Python 3.12.
To install the required packages, run the following command in your terminal: 

pip install requirements.txt

# Items
## Classifiers
Use the classifiers to replicate the training and validation of the classifiers used in the paper. 

### Relevance classification (BERT)
The relevance classifier (Relevance classifier (BERT).py) uses the underlying transformer model (https://huggingface.co/NoYo25/BiodivBERT) which is stored locally. 

### Multi classification (LLM)
For multi-classification, the DeepSeek prompt requires to setup a token for accessing the API. The operation can be performed on the translated and relevance-filtered input dataset (inputs/crs_all_translated_2000-2023_unique_binarylabeled_1only.csv). The resulting output file after multi-classification via DeepSeek is stored in outputs/unique 170k_llm labeled_final.
Note that the multi-classification was performed on unique entries only and multi labels were postpopulated to the full dataset (see Data section below).

## Plots and tables


# Data
The final classified OECD CRS dataset (2000-2023) can be downloaded here: XXX


