import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemer = PorterStemmer()


class Preprocessor:
    @staticmethod
    def process(sentence):
        s = sentence
        s = Preprocessor._remove_numbers(s)
        s = Preprocessor._lower(s)
        s = Preprocessor._remove_web(s)
        s = Preprocessor._remove_special_characters(s)
        # s = Preprocessor._remove_characters(s)
        s = Preprocessor._word_tokenize(s)
        s = Preprocessor._remove_stop_words(s)
        s = Preprocessor._post_tagger(s)
        s = Preprocessor._lemmatization(s)
        return " ".join(s)

    @staticmethod
    def _remove_numbers(sentence):
        return re.sub(r"\d+", "", sentence)

    @staticmethod
    def _lower(sent):
        return sent.lower()

    @staticmethod
    def _remove_characters(sentence):
        words = sentence.split()
        return [word for word in words if len(word) > 1]

    @staticmethod
    def _remove_web(sent):
        return re.sub(r"www.\S+|http:\S|https:\S", "", sent)

    @staticmethod
    def _remove_special_characters(sent):
        return re.sub("[^A-Za-z0-9\s]+", "", sent)

    @staticmethod
    def _word_tokenize(sent):
        return word_tokenize(sent)

    @staticmethod
    def _remove_stop_words(tokens):
        return [token for token in tokens if token not in stop_words]


    @staticmethod
    def _post_tagger(tokens):
        return nltk.pos_tag(tokens)

    @staticmethod
    def _lemmatization(tokens):
        final_words = []
        adjectives = ["JJR", "JJ", "JJS"]
        nouns = ["NN", "NNS", "NNP", "NNPS"]
        verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        adverbs = ["RB", "RBR", "RBS"]
        for word in tokens:
            pos = "n"
            if word[1] in adjectives:
                pos = "a"
            elif word[1] in nouns:
                pos = "n"
            elif word[1] in verbs:
                pos = "v"
            elif word[1] in adverbs:
                pos = "r"

            final_words.append(lemmatizer.lemmatize(word[0], pos))

        return final_words

    @staticmethod
    def _stemming(sent, stemer):
        return [stemer.stem(token) for token in sent]
