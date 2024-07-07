import pandas as pd
from collections import defaultdict #for initializing dictionaries.
from testing import extracted_text

def naive_bayes(test, training_data):
    #Ref for math : https://www.naukri.com/code360/library/naive-bayes-and-laplace-smoothing-3905
    sent_counts_array = training_data['sentiment'].value_counts()
    sent_probs = sent_counts_array / len(training_data)
    sentiments = []
    word_counts = defaultdict(lambda: defaultdict(int)) #count of [word | senti]
    sent_counts = defaultdict(int)
    for _, row in training_data.iterrows():
        sentiment = row['sentiment']
        text = row['line'].split()
        for word in text:
            word_counts[word][sentiment] += 1
            sent_counts[sentiment] += 1
    for text in test:
        text = text.split()
        scores = {}
        for sentiment in sent_probs.index:
            scores[sentiment] = sent_probs[sentiment]

        for word in text:
            for sentiment in scores:
                word_prob = (word_counts[word][sentiment] + 1) / (sent_counts[sentiment] + len(word_counts))
                scores[sentiment] *= word_prob

        sentiment = max(scores, key=scores.get) #choosing the most proboble senti.
        sentiments.append(sentiment)

    return sentiments


def main():
    training_data = pd.read_csv('sentiment_analysis_dataset.csv')
    test = extracted_text
    sentiments = naive_bayes(test, training_data)
    for i, sentiment in enumerate(sentiments, 1):
        print(f"Line {i}: Sentiment - {sentiment}")

main()
