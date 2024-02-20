from collections import defaultdict

'''
Function #1: unique word_list
Input: train_corpus (string) - the name of the training corpus file
Output: word_list (set) - a set of all the unique words in the training corpus
'''
def unique_word_list(train_corpus):
    word_list = set()
    with open(train_corpus, 'r') as file:
        for line in file:
            if line.strip():  
                token, _ = line.strip().split()
                word_list.add(token)
    return word_list

'''
Word list for test corpus
'''
def word_list_test(test_corpus):
    word_list = []
    with open(test_corpus, 'r') as file:
        for line in file:
            if line.strip():  
                word_list.append(line.strip())
    return word_list

'''
Table of frequencies of words that occur with each part of speech
'''
def pos_probabilities(train_corpus):
    word_pos_freq = defaultdict(lambda: defaultdict(int))
    with open(train_corpus, 'r') as file:
        for line in file:
            if line.strip():  
                word, pos = line.strip().split()
                word_pos_freq[pos][word] += 1
    return word_pos_freq

'''
Loop thru hash table and convert frequencies into probabilities
• freq/total = probability
'''
def pos_probabilities_frequencies(word_pos_freq):
    pos_probabilities = defaultdict(lambda: defaultdict(float))
    for pos in word_pos_freq:
        total_count = sum(word_pos_freq[pos].values())
        for word in word_pos_freq[pos]:
            pos_probabilities[pos][word] = word_pos_freq[pos][word] / total_count
    return pos_probabilities

'''
STATE → table of frequencies of following states
● Example: Transition['Begin_Sent'] → {'DT':1000,'NNP':500,'VB':200, …}
● Example: Transition['DT'] → {'NN':500,'NNP:'200,'VB':30,,…}
● Hash table of states with each a value a hash table from states to frequencies
● States = Begin_Sent, End_Sent and all POSs
'''
def transition_probabilities(train_corpus):
    transition_probabilities = defaultdict(lambda: defaultdict(int))
    with open(train_corpus, 'r') as file:
        previous_pos = 'Begin_Sent'
        for line in file:
            if line.strip():  
                _, pos = line.strip().split()
                transition_probabilities[previous_pos][pos] += 1
                previous_pos = pos
            else:
                transition_probabilities[previous_pos]['End_Sent'] += 1
                previous_pos = 'Begin_Sent'
    return transition_probabilities 

'''
State → table of frequencies of following states
Loop thru hash table and convert frequencies into probabilities
• freq/total = probability
'''
def transition_probabilities_frequencies(transition_probabilities):
    transition_prob_freq = defaultdict(lambda: defaultdict(float))
    for pos in transition_probabilities:
        total_count = sum(transition_probabilities[pos].values())
        for next_pos in transition_probabilities[pos]:
            transition_prob_freq[pos][next_pos] = transition_probabilities[pos][next_pos] / total_count
    return transition_prob_freq

'''
Tagger for the corpus
'''
def viterbi_hmm_tagger(test_corpus, pos_prob_freq, transition_prob_freq, train_word_list, test_word_list, unknown_word_prob):
    tagged_corpus = []
    with open(test_corpus, 'r') as file:
        sentence = []  
        for line in file:
            if line.strip():  
                word = line.strip()  
                sentence.append(word)  
            else:
                if sentence:
                    tagged_sentence = viterbi(sentence, pos_prob_freq, transition_prob_freq, train_word_list, test_word_list, unknown_word_prob)
                    tagged_corpus.append(tagged_sentence)  
                    sentence = [] 
    if sentence:
        tagged_sentence = viterbi(sentence, pos_prob_freq, transition_prob_freq, train_word_list, test_word_list, unknown_word_prob)
        tagged_corpus.append(tagged_sentence)
    write_to_output(tagged_corpus)

def write_to_output(tagged_corpus):
    with open('submission.pos', 'w') as file:
        for sentence in tagged_corpus:
            for word, pos in sentence:
                file.write(word +'\t' + pos+'\n')
            file.write('\n')

'''
Make a 2 dimensional array (or equivalent)
– columns represent tokens at positions in the text
• 0 = start of sentence
• N = Nth token (word punctuation at position N)
• Length+1 = end of sentence
– rows represent S states: the start symbol, the end symbol and all possible POS (NN, JJ, ...)
– cells represent the likelihood that a particular word is at a particular state
• Traverse the chart as per the algorithm (fish sleep slides, etc.)
– For all states at position 1, multiply transition probability from Start (position 0) by
likelihood that word at position 1 occurs in that state. Choose highest score for each cell.
– For n from 2 to N (columns)
• for each cell [n,s] in column n and each state [n-1,s'] in column n-1:
• get the product of:
– likelihood that token n occurs in state s
– the transition probability from s' to s
– the score stored in [n-1,s']
• At each position [n,s], record the max of the s scores calculated
The probability of each transition to state N for token T is assumed to be the
product of 3 factors
– Probability that state N occurs with token T
• There is 100% chance that the start state will be at the beginning of the sentence
• There is 100% chance that the end state will be at the end of the sentence
• If a token was observed in the training corpus, look up probability from table
• For Out of Vocabulary words, there are several strategies
– Simple strategy (for first implementation): 1/1000 or 100% divided by number of
states or any fraction that is the same for all POS
– Probability that state N occurs given previous state
• Look up in table, calculate for every possible previous state
– Highest Probability of previous state (calculate for each previous state)
• For each new state, choose the highest score (this is the bigram model)
• Choose the POS tag sequence resulting in the highest score in the end state
'''

def viterbi(sentence, pos_prob_freq, transition_prob_freq, word_train, word_test, unkown_word_prob):
    states = list(pos_prob_freq.keys())
    n = len(sentence)
    m = len(states)

    viterbi_matrix = [[0.0] * m for _ in range(n)]
    backpointers = [[0] * m for _ in range(n)]

    for i, state in enumerate(states):
        emission_prob = pos_prob_freq[state].get(sentence[0], unkown_word_prob) / len(word_train)
        transition_prob = transition_prob_freq['Begin_Sent'].get(state, unkown_word_prob)
        viterbi_matrix[0][i] = emission_prob * transition_prob
        backpointers[0][i] = 0
    for t in range(1, n):
        for j, next_state in enumerate(states):
            max_prob = -1
            max_prob_index = -1
            for i, state in enumerate(states):
                emission_prob = pos_prob_freq[next_state].get(sentence[t], unkown_word_prob) / len(word_train)
                transition_prob = transition_prob_freq[state].get(next_state, unkown_word_prob)
                prob = viterbi_matrix[t - 1][i] * emission_prob * transition_prob
                if prob > max_prob:
                    max_prob = prob
                    max_prob_index = i
            viterbi_matrix[t][j] = max_prob
            backpointers[t][j] = max_prob_index
    best_path = []
    max_prob_index = viterbi_matrix[n - 1].index(max(viterbi_matrix[n - 1]))
    best_path.append((sentence[n - 1], states[max_prob_index]))
    prev_index = max_prob_index
    for t in range(n - 2, -1, -1):
        prev_index = backpointers[t + 1][prev_index]
        best_path.insert(0, (sentence[t], states[prev_index]))
    return best_path


def run():
    dev_corpus = 'merged.pos'
    test_corpus = 'WSJ_23.words'
    probabilities_frequency = pos_probabilities_frequencies(pos_probabilities(dev_corpus))
    transition_prob_freq = transition_probabilities_frequencies(transition_probabilities(dev_corpus))
    unique_word_train = unique_word_list(dev_corpus) 
    word_test = word_list_test(test_corpus) 
    tagged_corpus = viterbi_hmm_tagger(test_corpus, probabilities_frequency, transition_prob_freq, unique_word_train, word_test, 1e-6) # run this function to tag the test corpus
    #Note that output is in submission.pos

if __name__ == "__main__":
    run()
