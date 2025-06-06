# Natural-Language-Processing-and-Machine-Learning-Project
This repository contains a collection of Jupyter Notebooks demonstrating various concepts and techniques in Natural Language Processing (NLP) and Machine Learning (ML). The projects primarily focus on text analysis, natural language understanding, and classification tasks.


**Contents**
**Text Analytics Coursework** **(text_analytics_part1.ipynb):** This notebook covers fundamental text analytics tasks, including data loading, cleaning, tokenization, part-of-speech (POS) tagging, and the identification of meaningful words. It also explores word co-occurrence and clustering based on word distances calculated using co-occurrence and Dijkstra's algorithm. This project likely involves sentiment analysis or similar classification, as indicated by the mention of "Emotion dataset" and "risk," "neutral," or "opportunity" sentiments.

**Named Entity Recognition (Task.ipynb)**: This notebook focuses on building and evaluating a Named Entity Recognition (NER) model. It leverages the HuggingFace transformers library for tasks such as tokenization with AutoTokenizer, token classification with AutoModelForTokenClassification, and training with Trainer. The project utilizes the "Broad Twitter Corpus (BTC)" dataset and employs seqeval for comprehensive metric computation.

**NLP and Machine Learning Fundamentals (AI.ipynb):** This notebook delves into foundational NLP and ML concepts. It includes:

Data Loading and Cleaning: Processing text from various sources like Project Gutenberg.
Text Preprocessing: Normalization, sentence tokenization, and filtering of meaningful words.
Word Relationship Analysis: Calculating word co-occurrence and shortest path distances using Dijkstra's algorithm, followed by hierarchical clustering to visualize word relationships.
Classification Models: Implementation and evaluation of Logistic Regression and potentially Neural Network models for classification, including data generation and accuracy assessment.

Here's a good GitHub description for your code, incorporating details from the provided notebooks:

Natural Language Processing and Machine Learning Projects
This repository contains a collection of Jupyter Notebooks demonstrating various concepts and techniques in Natural Language Processing (NLP) and Machine Learning (ML). The projects primarily focus on text analysis, natural language understanding, and classification tasks.

Contents
Text Analytics Coursework (text_analytics_part1 (4).ipynb): This notebook covers fundamental text analytics tasks, including data loading, cleaning, tokenization, part-of-speech (POS) tagging, and the identification of meaningful words. It also explores word co-occurrence and clustering based on word distances calculated using co-occurrence and Dijkstra's algorithm. This project likely involves sentiment analysis or similar classification, as indicated by the mention of "Emotion dataset" and "risk," "neutral," or "opportunity" sentiments.

Named Entity Recognition (Task_3.ipynb): This notebook focuses on building and evaluating a Named Entity Recognition (NER) model. It leverages the HuggingFace transformers library for tasks such as tokenization with AutoTokenizer, token classification with AutoModelForTokenClassification, and training with Trainer. The project utilizes the "Broad Twitter Corpus (BTC)" dataset and employs seqeval for comprehensive metric computation.

NLP and Machine Learning Fundamentals (AI (1).ipynb): This notebook delves into foundational NLP and ML concepts. It includes:

Data Loading and Cleaning: Processing text from various sources like Project Gutenberg.
Text Preprocessing: Normalization, sentence tokenization, and filtering of meaningful words.
Word Relationship Analysis: Calculating word co-occurrence and shortest path distances using Dijkstra's algorithm, followed by hierarchical clustering to visualize word relationships.
Classification Models: Implementation and evaluation of Logistic Regression and potentially Neural Network models for classification, including data generation and accuracy assessment.


**Technologies Used**
Python
Jupyter Notebook
nltk (Natural Language Toolkit)
requests
re (Regular Expressions)
numpy
scipy
matplotlib
scikit-learn
transformers (HuggingFace)
datasets (HuggingFace)
seqeval
torch (PyTorch) - Potentially, as indicated by installation commands in text_analytics_part1
pandas
networkx

**Install the required packages:**
pip install -r requirements.txt # (You may need to create this file based on the imports in the notebooks)
# Alternatively, install individually as seen in the notebooks:
pip install requests re nltk numpy scipy matplotlib scikit-learn transformers datasets evaluate seqeval torch pandas networkx


Feel free to explore the code, experiment with the parameters, and adapt it for your own projects!
