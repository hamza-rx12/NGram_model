import math
from collections import defaultdict, Counter


class NgramLanguageModel:
    """
    Class with code to train and run n-gram language models with add-k smoothing.
    """

    def __init__(self, infile=None, ngram_size=None):

        # self.trigram_counts = defaultdict(int)
        # self.bigram_counts = defaultdict(int)
        self.ngram_counts = defaultdict(int)  # n gram
        self.context_counts = defaultdict(int)  # n-1 gram
        self.k = 0.01

        if infile is None or ngram_size is None:
            return
        # self.prepare_data(infile, ngram_size)
        self.train(infile, ngram_size)

    def prepare_data(self, infile, ngram_size=2):
        """
        This method takes as input a text file representing the corpus
        normalizes the text, and adds start  and end sentence tokens
        Remember that you have to add a special '<s>' token to the beginning
        and '</s>' token to the end of each sentence to correctly estimate the bigram
        probabilities.

        Remember that you have to add a special         '<s><s>' token to the beginning
        and '</s>' token to the end of each sentence to correctly estimate the trigram
        probabilities.

        This method also handles out-of-vocabulary words (tokens).
        To achieve this, the method searches for words that appear less than N times in the training data and replaces them with <UNK>.

        Parameters
        ----------
        infile : str
            File path to the training corpus.

        ngram_size : int
            specifying which model to use.
            Use this variable in an if/else statement. (n=2 for bigram and n=3 for trigram)

        Returns
        -------
        the preprocessed corpus

        """

        with open(infile, "r") as file:
            data = file.read().lower()

        sentences = data.split("\n")
        sentences = [
            ("<s> " * (ngram_size - 1) + sent + " </s>").split() for sent in sentences
        ]

        tokens = Counter(word for sent in sentences for word in sent)

        tokenized_sentences = [
            [word if tokens[word] > 1 else "<UNK>" for word in sent]
            for sent in sentences
        ]

        # for i in tokenized_sentences:
        #     print(i)

        return tokenized_sentences

    def train(self, infile, ngram_size=2):
        """Trains the language models by calculating n-gram counts from the corpus
        at the path given in the variable `infile`.

        These counts should be accumulated on the trigram_counts and bigram_counts
        objects.
        if ngram_size is set to 2 train only the bigram model, if ngram_size is set to 3 train only a trigram model


        Parameters
        ----------
        infile : str
            File path to the training corpus.

        ngram_size : int
            specifying which model to use.
            Use this variable in an if/else statement. (n=2 for bigram and n=3 for trigram)

        Returns
        -------
        None (updates class attributes self.*_counts)
        """
        toeknized_sentences = self.prepare_data(infile, ngram_size)

        for sent in toeknized_sentences:
            for i in range(len(sent) - ngram_size + 1):
                # ngram
                ngram = tuple(sent[i : i + ngram_size])
                self.ngram_counts[ngram] += 1
                # context
                context = tuple(sent[i : i + ngram_size - 1])
                self.context_counts[context] += 1
            # last one
            context = tuple(sent[1 - ngram_size :])
            self.context_counts[context] += 1

    def predict_ngram(self, sentence, ngram_size=2):
        """Calculates the log probability of the given sentence using an ngram LM.


        Parameters
        ----------
        sentence : str
            A sentence for which to calculate the probability.

        ngram_size :
            specifying which model to use.
            Use this variable in an if/else statement. (n=2 for bigram and n=3 for trigram)
        Returns
        -------
        float
            The log probability of the sentence.
        """
        # prepare the sentence
        tokens = sentence.lower().split()
        tokens = ["<s>"] * (ngram_size - 1) + tokens + ["</s>"]
        log_prob = 0.0

        for i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[i : i + ngram_size])
            context = tuple(tokens[i : i + ngram_size - 1])
            ngram_count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]
            # print(ngram, context, ngram_count, context_count)
            log_prob += math.log(
                (ngram_count + self.k)
                / (context_count + self.k * len(self.ngram_counts))
            )

        return log_prob
        # return math.exp(log_prob)

    def test_perplexity(self, test_file, ngram_size=2):
        """Calculate the perplexity of the trained LM on a test corpus.

        This seems complicated, but is actually quite simple.

        First we need to calculate the total probability of the test corpus.
        We can do this by summing the log probabiities of each sentence in the corpus.

        Then we need to normalize (e.g., divide) this summed log probability by the
        total number of tokens in the test corpus. The one tricky bit here is we need
        to augment this count of the total number of tokens by one for each sentence,
        since we're including the sentence-end token in these probability estimates.

        Finally, to convert this result back to a perplexity, we need to multiply it
        by negative one, and exponentiate it - e.g., if we have the result of the above
        in a variable called 'val', we will return math.exp(val).

        Parameters
        -------
        test_file : str
            File path to a test corpus.
            (assumed pre-tokenized, whitespace-separated, one line per sentence)

        ngram_size : int
            specifying which model to use.
            Use this variable in an if/else statement. (n=2 for bigram and n=3 for trigram)

        Returns
        -------
        float
            The perplexity of the corpus (normalized total log probability).
        """
        total_log_prob = 0.0
        total_tokens = 0

        with open(test_file, "r") as file:
            test_sentences = file.read().lower().split("\n")

        for sentence in test_sentences:
            total_log_prob += self.predict_ngram(sentence, ngram_size)
            total_tokens += len(sentence.split()) + 1  # +1 for the end token

        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)

        # print(perplexity)
        return perplexity


if __name__ == "__main__":
    lm = NgramLanguageModel("ngramv1.train", ngram_size=3)
    print(lm.predict_ngram("NOT IN A TREE !", ngram_size=3))
    lm.test_perplexity("ngramv1.test", ngram_size=3)
