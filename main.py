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
import pickle
import os
import numpy as np


class LanguageModel(NgramLanguageModel):
    def __init__(self, infile=None, ngram_size=None):
        super().__init__(infile, ngram_size)

    def prepare_data(self, infile, ngram_size=2):
        tokenized_file = f"{infile}_tokenized_{ngram_size}.pkl"
        if os.path.exists(tokenized_file):
            with open(tokenized_file, "rb") as file:
                tokenized_sentences = pickle.load(file)
            print("Loaded tokenized sentences from file.")

        else:
            with open(infile, "r") as file:
                data = file.read().lower()
            ####### main processing ########
            sentences = sent_tokenize(data)
            sentences = [
                ("<s> " * (ngram_size - 1) + sent + " </s>").split()
                for sent in sentences
            ]

            tokens = Counter(word for sent in sentences for word in sent)

            tokenized_sentences = [
                [word if tokens[word] > 1 else "<UNK>" for word in sent]
                for sent in sentences
            ]
            ################################
            print("Tokenized sentences calculated.")

            # Save the tokenized sentences to a file
            with open(tokenized_file, "wb") as file:
                pickle.dump(tokenized_sentences, file)
                print("Tokenized sentences saved to file.")
        # print(tokenized_sentences)
        return tokenized_sentences

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
    import time

    start = time.time()

    lm = LanguageModel("data/big_data.txt", ngram_size=3)
    print(lm.predict_ngram("I'm doing it", ngram_size=3))
    print(lm.test_perplexity("data/ngramv1.test", ngram_size=3))

    end = time.time()
    print(f"Total time taken: {end - start} seconds")
