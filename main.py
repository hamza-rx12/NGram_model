from language_modeling import NgramLanguageModel
import numpy as np


class LanguageModel(NgramLanguageModel):
    def __init__(self, infile=None, ngram_size=None):
        super().__init__(infile, ngram_size)

    # def prepare_data(self, infile, ngram_size=2):
    #     super().prepare_data(infile, ngram_size)

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
    lm = LanguageModel("data/ngramv1.train", ngram_size=3)
    print(lm.predict_ngram("NOT IN A TREE !", ngram_size=3))
    lm.test_perplexity("data/ngramv1.test", ngram_size=3)
