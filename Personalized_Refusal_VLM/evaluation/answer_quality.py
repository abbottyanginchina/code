import nltk

# Function of BLEU score
def bleu_score(reference, hypothesis):
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

# Function of ROUGE score
def rouge_score(reference, hypothesis):
    return nltk.translate.rouge_score.sentence_rouge(reference, hypothesis)

# Function of METEOR score
def meteor_score(reference, hypothesis):
    return nltk.translate.meteor_score.sentence_meteor(reference, hypothesis)

