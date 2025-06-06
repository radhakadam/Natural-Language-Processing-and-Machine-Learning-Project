**Natural Language Processing and Machine Learning Coursework Projects**
This repository contains a collection of Jupyter Notebooks and associated code for the Text Analytics Coursework, Spring 2025. The projects focus on applying and analyzing various techniques in Natural Language Processing (NLP) and Machine Learning (ML) to address specific text analytics problems.

**Overview of Projects**
The coursework is divided into three main parts, each addressing a distinct problem statement:

**Part 1: Climate Sentiment Analysis (Jupyter Notebook)**
This section focuses on classifying the sentiment of corporate disclosures related to climate-related developments. The objective is to categorize text as representing a climate "risk," "opportunity," or "neutral" sentiment. This task utilizes the ClimateBERT dataset. The accompanying Jupyter Notebook (text_analytics_part1 (4).ipynb) contains detailed tasks (1.1, 1.2, and 1.3) for implementation.


**Task 2: Climate Sentiment Report**
This task involves a comprehensive evaluation of the methods implemented in Part 1 for climate sentiment classification. The report should include:

An explanation of modifications made to the naive Bayes classifier in task 1.1d, detailing changes, benefits, and any unsuccessful attempts.
A comparison of results from different methods, presented in a table or plot, along with an interpretation of their performance. Discussions may incorporate concepts like transfer learning and suggestions for future improvements.

An analysis to identify topics associated with climate-related risks or opportunities within the dataset. This includes explaining the chosen method for topic identification, justifying the approach, comparing variations of the method, and interpreting results with a summary of limitations.
**

Task 3: Named Entity Recognition (NER) on Twitter**
The goal of this task is to develop a tool for Named Entity Recognition from Twitter posts. This tool aims to extract information about particular people, organizations, and locations from unstructured social media text. The project utilizes the Broad Twitter Corpus (BTC) dataset for training and testing the NER tagger. 

Key aspects include:
Designing and implementing a sequence tagger for the BTC dataset.
Providing a detailed explanation of the chosen NER method, its strengths, and limitations.
Evaluating the performance of the implemented method, including the choice of performance metrics, testing procedures, and analysis of results (e.g., plots/tables).
Identifying specific types of errors made by the methods and suggesting potential improvements.

**Technologies Used:**
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
torch (PyTorch)
pandas
networkx

**Install the required packages:**
pip install -r requirements.txt # (You may need to create this file based on the imports in the notebooks)
# Alternatively, install individually as seen in the notebooks and problem statement:
pip install requests re nltk numpy scipy matplotlib scikit-learn transformers datasets evaluate seqeval torch pandas networkx
