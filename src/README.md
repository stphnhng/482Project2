Project Name: Poly Alexa
Team Members: Hanson Egbert, Stephen Hung, Dmitriy Timokhin

# Steps to Install 
1. Pip install / import libraries
   - Python3
   - tensorflow_hub
   - spacy
      - en_core_web_lg
   - nltk
      - nltk.download('punkt')
      - nltk.download('stopwords')
      - nltk.download('averaged_perceptron_tagger')
      - nltk.download('wordnet') 

2. Make sure data is under data/ directory (data/ and src/ should be on the same level)
   - Files required:
      1. data_wInfo.json
      2. cp_alexa_qa.json
      3. generated_qa.json

3. Make sure python script "project-script.py" is under src/ folder
   - Make sure the data/ folder and the src/ folder are in the same location.

4. How to run cal poly program?
   1. python3 project-script.py
      - It takes a while to run for the first time because it requires loading models and big libraries.
   2. Wait till the program queries you with this statement: "Please ask a question about Cal Poly: "
   3. Enter a question about Cal Poly
   4. Get your answer!

# Running the Alexa Program

Due to prerequisites and the AWS account requirements as well as Alexa developer accounts, you cannot run the program yourself.
A demo in our presentation will be provided though, as well as the AWS lambda code. 

