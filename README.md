# NLP Project 1.2
## Authors
- Benedikt Einar Utsch (486106)
- Hassan Ahmed Hassan Hussein Bassiouny (413011)

## Prerequisites
Before running the project, please ensure you have installed all required packages:
```bash 
pip install -r ./requirements.txt
```
## Getting started
The results of our Project will be presented in the report.
To view our generated logs please visit this Github repo: [GitHub Repository](https://github.com/hassan1595/NLP-assignment01)
or run 
```bash
git clone https://github.com/hassan1595/NLP-assignment01.git
```

open a python shell and run:
```bash
>>import nltk
>>nltk.download('punkt')
>>nltk.download('stopwords')
```

## Run the analyse file
The 'analyse' script will extract general insights from the data and generate plots.
To reproduce the plots, execute the analysis script run:
```bash
python analyze.py
```

## Run the apply file
Execute the 'apply' script to train the models and tests their performance. This script will also generate the logs and metrics.:
```bash
python apply.py
```