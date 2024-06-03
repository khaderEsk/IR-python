

import pandas as pd

from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()
stemer = PorterStemmer()


class IndexModel:
    def __init__(self, documents_df):
        """
        input:
            documents_df (DataFrame): contains the preprocessed documents
        """
        self.tfidf_vectorizer = TfidfVectorizer()
        self._index = self.tfidf_vectorizer.fit_transform(documents_df["processed_text"])
        self.index_docno = documents_df['docno']

    def getindex(self):
        return self._index

    def getdocno(self, i):
        return self.index_docno.iloc[i]

    def vectorize(self, sentence):
        if isinstance(sentence,str):
            qry=pd.DataFrame([{"text":sentence}])
        else:
            qry=sentence
        return self.tfidf_vectorizer.transform(qry['text'])