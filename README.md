# Part-of-Speech Tagger using Hidden Markov Model (HMM)


## Overview
This system is a part-of-speech (POS) tagger implemented using the Hidden Markov Model (HMM) and the Viterbi algorithm.

## How to Run the System
1. Replace the training and test corpus files with the files you desire to use (`train_corpus.pos` and `test_corpus.word`). (Modify the file names within the `run()` function in the `main.py` script)
2. Run the `main.py` script using Python 3.
3. The output will be written in a `submission.pos` file.
4. To see the difference between the predicted tags and the standard tags, run `python3 score.py <train_file> <test_file>`.

## Handling Out-of-Vocabulary (OOV) Items
1. The system assigns a default emission probability of `1e-6` for OOV words and recalculates the likelihoods for all words by dividing by the number of unique words in the training set.
