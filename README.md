# BiodiversityDevelopmentFinance
Supplementary code for Wildgruber et al. (2026): A multi-dimensional view to map biodiversity development finance globally. Currently under review.

# Setup
Tu run the code, use Python 3.10 or higher. The code has been developed and tested on Python 3.12.
To install the required packages, run the following command in your terminal: 

pip install requirements.txt

# Items
## 📁 Classifiers
Use the classifiers to replicate the training and validation of the classifiers used in the paper. 

### 📁 Relevance classification (BERT)
The relevance classifier '_Relevance classifier (BERT).py_' uses the underlying transformer model BiodivBERT which can be downloaded from  https://huggingface.co/NoYo25/BiodivBERT or from 
[0] (see Data section). It performes the training on a randomized data subset of 2000 data points [1]. The resulting final model is stored in [2] and can be applied to unique entries of the OECD CRS data set [3] via the script '_Binary labeling_Full data set.py_'. 

When running the code for replication, [0], [1], [2], and [3] need to be downloaded from the Drive and stored locally in an '_inputs_' folder, before running the scripts. The resulting model and classification will be stored in '_outputs_'.

### 📁 Multi classification (LLM)
For multi-classification, the '_DeepSeek Prompt.py_' is performed on the translated and relevance-filtered input dataset [4]. The resulting final output file after multi-classification via DeepSeek that was used in the paper is stored as [5]. Note that the multi-classification was performed on unique entries only and multi labels were postpopulated to the full dataset. 
The ex-post performance analysis was performed on dataset [6] using '_LLM ex-post performance.py_'.

When running the code for replication, it is required to setup a token for accessing the DeepSeek API and to download and store the file [4] in an '_inputs_' folder.the resulting multi-classified dataset will be stored in '_outputs_'.

## 📁 Plots and tables
The plots and tables that are displayed in the main paper or the supplementary information (SI) can be reproduced via the script '_Figures.py_' and with the stored files in '_inputs_' on the final dataset [7].

# Data
The final classified OECD CRS dataset (2000-2023) and all mentioned interim datasets can be downloaded here: [https://drive.google.com/drive/folders/1VIr1uq24tj3lfwnDTtK-d5eW2pKabvKO?usp=drive_link](https://drive.google.com/drive/folders/1cGY55poRvnYpdWHACAivPqaYpab2x692?usp=drive_link)

[0] GoogleDrive/Relevance_Classifier_inputs/NoYo25 BiodivBERT

[1] GoogleDrive/Relevance_Classifier_inputs/BERT_random_2000_vfinal.csv

[2] GoogleDrive/Relevance_Classifier_inputs/fine_tuned_model_paper

[3] GoogleDrive/Relevance_Classifier_inputs/crs_all_translated_2000-2023_unique.csv

[4] GoogleDrive/Multilabel_Classifier_inputs/crs_all_translated_2000-2023_unique_binarylabeled_1only.csv

[5] GoogleDrive/Multilabel_Classifier_inputs/unique 170k_llm labeled_final.csv

[6] GoogleDrive/Multilabel_Classifier_inputs/Ex-Post-LLM Performance Analysis.xlsx

[7] GoogleDrive/Final_dataset/crs_all_translated_bert_llm_2000-2023_vfinal.csv
