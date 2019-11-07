# calculate a score for a given class taking into account word commonality
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if word.lower() in class_words[class_name]:
            # treat each word with relative weight
            score += (1 / corpus_words[word.lower()])

            if show_details:
                print ("   match: %s (%s)" % (word.lower(), 1 / corpus_words[word.lower()]))
    return score
