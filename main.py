import nltk
from nltk.data import find
from nltk.tokenize import sent_tokenize

# try:
#     find("tokenizers/punkt")
# except LookupError:
#     # nltk.download("punkt_tab")
#     nltk.download("punkt")


# def download_nltk_data():
#     try:
#         find("tokenizers/punkt")
#     except LookupError:
#         nltk.download("punkt")


# # Call this function at the start of your script
# download_nltk_data()


from language_modeling import NgramLanguageModel
from collections import Counter
import numpy as np


class LanguageModel(NgramLanguageModel):
    def __init__(self, infile=None, ngram_size=None):
        super().__init__(infile, ngram_size)

    def prepare_data(self, infile, ngram_size=2):
        with open(infile, "r") as file:
            data = file.read().lower()
        sentences = sent_tokenize(data)
        # print(sentences)
        sentences = [
            ("<s> " * (ngram_size - 1) + sent + " </s>").split() for sent in sentences
        ]

        tokens = Counter(word for sent in sentences for word in sent)

        tokenized_sentences = [
            [word if tokens[word] > 1 else "<UNK>" for word in sent]
            for sent in sentences
        ]
        print(tokenized_sentences)
        return tokenized_sentences

    # def train(self, infile, ngram_size=2):
    #     super().train(infile, ngram_size)

    # def test_perplexity(self, sentence):
    #     return super().perplexity(sentence)

    def generateText(self, ngram_size=2):
        sentence = ["<s>"] * (ngram_size - 1)
        while sentence[-1] != "</s>":
            current_word = sentence[-1]
            next_words = self.model[current_word]
            next_word = np.random.choice(
                list(next_words.keys()), p=list(next_words.values())
            )
            sentence.append(next_word)

        return " ".join(sentence[1:-1])


if __name__ == "__main__":
    lm = LanguageModel("data/big_data.txt", ngram_size=3)
    print(lm.predict_ngram("I'm doing it", ngram_size=3))
    print(lm.test_perplexity("data/ngramv1.test", ngram_size=3))
