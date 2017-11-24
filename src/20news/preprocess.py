from sklearn.datasets import fetch_20newsgroups
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
from collections import defaultdict
import operator
import string 

def remove_punc(word):
    new_word = ''.join([c for c in word if c not in string.punctuation])
    return new_word

def data2npy(data, output_path, terms):
    stop_words = stopwords.words('english')
    stemmer = LancasterStemmer()
    letters = string.ascii_lowercase
    word2idx = {term:i for i, term in enumerate(terms)}
    termset = set(terms)
    word_matrix = np.zeros((len(data), len(terms)), dtype=np.float32)
    for sample_index, sample in enumerate(data):
        words = word_tokenize(sample)
        words = [word.lower() for word in words]
        words = [remove_punc(word) for word in words]
        words = [word for word in words if word not in stop_words \
                and not word.isdigit() and word not in letters]
        words = [stemmer.stem(word) for word in words]
        for word in words:
            if word in termset:
                word_index = word2idx[word]
                word_matrix[sample_index][word_index] += 1
    
    word_matrix = np.log(word_matrix + 1)
    total_matrix = (np.max(word_matrix, axis=1) + 1e-10)
    word_matrix /= np.expand_dims(total_matrix, axis=1)
    print(np.sum(word_matrix.sum(axis=1) == 0))
    np.save(output_path, word_matrix)

    return word_matrix


def get_terms(raw_data, term_path, n_vocabs=5000):
    stop_words = stopwords.words('english')
    stemmer = LancasterStemmer()
    letters = string.ascii_lowercase
    counter = defaultdict(lambda : 0)
    for sample in raw_data:
        words = word_tokenize(sample)
        words = [word.lower() for word in words]
        words = [remove_punc(word) for word in words]
        words = [word for word in words if word not in stop_words \
                and not word.isdigit() and word not in letters]
        words = [stemmer.stem(word) for word in words]
        for word in words:
            counter[word] += 1
    sorted_count = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    with open(term_path, 'w') as f_terms:
        for term, count in sorted_count[:n_vocabs]:
            f_terms.write('{} {}\n'.format(term, count))

    terms = [term for term, _ in sorted_count[:n_vocabs]]
    return terms

if __name__ == '__main__':
    term_path = '../../dataset/20news/terms.txt'
    train_path = '../../dataset/20news/npy/train.npy'
    test_path = '../../dataset/20news/npy/test.npy'

    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    terms = get_terms(newsgroups_train.data + newsgroups_test.data, term_path)
    data2npy(newsgroups_train.data, train_path, terms)
    data2npy(newsgroups_test.data, test_path, terms)

