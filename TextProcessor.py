import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from unidecode import unidecode

class TextProcessor:
    def preprocess_text(self, text):
        stop_words = stopwords.words("English")
        stop_words.append("n't")
        text = re.sub(r"http\S+", "", text)
        words = word_tokenize(text.lower())
        filtered_text = [word for word in words if word not in stop_words]
        stemmer = PorterStemmer()
        lowercase_words = [word.lower() for word in filtered_text]
        stemmed_words = [stemmer.stem(word) for word in lowercase_words]
        _remove = string.punctuation.replace("", "") 
        _remove_text = " ".join(stemmed_words).translate(str.maketrans("", "", _remove))
        clean_text = re.sub("[^A-Za-z0-9\s]+", "", _remove_text)
        words = clean_text.split()
        filtered_words = [word for word in words if len(word) > 1]
        filtered_text = " ".join(filtered_words)
        return filtered_text

    def text(self):
        full_docs = pd.read_csv("file.tsv", sep="\t", encoding="utf-8")
        full_docs["text"] = full_docs["text"].apply(
            lambda x: unidecode(self.preprocess_text(x))
        )
        rr = full_docs["text"].apply(self.preprocess_text)

        df = pd.read_csv("clean_data.tsv", sep="\t")

        new_data = {"doc_id": full_docs["doc_id"], "text": rr}
        df_new = pd.DataFrame(new_data)
        df = pd.concat([df, df_new], ignore_index=True)