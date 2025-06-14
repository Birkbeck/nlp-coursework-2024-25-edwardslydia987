#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import pandas as pd
nltk.download('cmudict')
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt_tab')
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import string
from collections import Counter

d = cmudict.dict()

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    words = word_tokenize(text)
    word_count = len(words)
    sentances = sent_tokenize(text)
    sentance_count = len(sentances)

    syllable_count = sum(count_syl(word, d) for word in words)
    flesch_kincaid = 0.39 * (word_count / sentance_count) + 11.8 * (syllable_count / word_count) - 15.59
    
    return(flesch_kincaid)

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower().strip(string.punctuation)
    if word in d:
        return len([y for y in d[word][0] if y[-1].isdigit()])
    else:
        return max(1, len(re.findall(r'[aeiou]+', word)))


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    for file_path in path.glob("*.txt"):
        with open(file_path) as file:
            text = file.read()
            file_name = file_path.stem
            column_split = file_name.split("-")
            title = column_split[0]
            author = column_split[1]
            year = column_split[2]
        data.append({"text": text, "title": title, "author": author, "year": year})
    df = pd.DataFrame(data)
    df.sort_values(by=["year"])
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path.mkdir(parents=True, exist_ok=True)
    parsed_docs = [nlp(text) for text in df['text']]
    df["Parsed"] = parsed_docs
    df.to_pickle(store_path / out_name)
    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = nltk.word_tokenize(text)
    words = [token.lower() for token in tokens if token not in string.punctuation]
    types = set(words)
    ttr = len(types) / len(words)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results

def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def object_counts(doc):
    """Extracts the most common objects in a parsed document. Returns a list of tuples."""
    all_objects = []
    for i, row in df.iterrows():
        doc = row['Parsed']
        if hasattr(doc, '__iter__') and hasattr(doc[0] if len(doc) > 0 else None, 'dep_'):
            objects = [token.text.lower() for token in doc if token.dep_ == 'dobj']
            all_objects.extend(objects)
    #common = Counter(objects).most_common(10)
    #return common
    
    return Counter(all_objects).most_common(10)


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print(object_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

