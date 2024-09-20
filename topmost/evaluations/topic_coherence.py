from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from itertools import combinations
from ..data.file_utils import split_text_word
import os


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary,
                        topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return cv_per_topic, score


def compute_dynamic_TC(train_texts, train_times, vocab, top_words_list, cv_type='c_v'):
    cv_score_list = list()

    for time in tqdm(range(len(top_words_list))):
        # use the texts of each time slice as the reference corpus.
        idx = np.where(train_times == time)[0]
        reference_corpus = [train_texts[i] for i in idx]

        # use the the topics at the time slice
        top_words = top_words_list[time]
        cv_score = compute_topic_coherence(
            reference_corpus, vocab, top_words, cv_type)
        cv_score_list.append(cv_score)

    print("===>CV score list: ", cv_score_list)

    return np.mean(cv_score_list)


def TC_on_wikipedia(top_word_path, cv_type='C_V'):
    """
    Compute the TC score on the Wikipedia dataset
    """
    try:
        jar_dir = os.path.join("topmost", "evaluations")
        wiki_dir = os.path.join(".", 'data')
        random_number = np.random.randint(100000)
        os.system(
            f"java -jar {os.path.join(jar_dir, 'pametto.jar')} {os.path.join(wiki_dir, 'wikipedia', 'wikipedia_bd')} {cv_type} {top_word_path} > tmp{random_number}.txt")
        cv_score = []
        with open(f"tmp{random_number}.txt", "r") as f:
            for line in f.readlines():
                if not line.startswith("202"):
                    cv_score.append(float(line.strip().split()[1]))
        os.remove(f"tmp{random_number}.txt")
        return cv_score, sum(cv_score) / len(cv_score)
    except:
        jar_dir = os.path.join("topmost", "evaluations")
        wiki_dir = "/kaggle/input/wikipedia/"
        random_number = np.random.randint(100000)
        os.system(
            f"java -jar {os.path.join(jar_dir, 'pametto.jar')} {os.path.join(wiki_dir, 'wikipedia', 'wikipedia_bd')} {cv_type} {top_word_path} > tmp{random_number}.txt")
        cv_score = []
        with open(f"tmp{random_number}.txt", "r") as f:
            for line in f.readlines():
                if not line.startswith("202"):
                    cv_score.append(float(line.strip().split()[1]))
        os.remove(f"tmp{random_number}.txt")
        return cv_score, sum(cv_score) / len(cv_score)

def get_average_word2vec_similarity(topic_word_list, model):
    similarity_list = []
    missing_word_count = 0
    for topic, word_list in enumerate(topic_word_list):
        word_list_filtered = [word for word in word_list if model.has_index_for(word)]
        missing_word_count += len(word_list) - len(word_list_filtered)
        for word1, word2 in combinations(word_list_filtered, 2):
            similarity = model.similarity(word1, word2)
            similarity_list.append(similarity)
    return sum(similarity_list) / len(similarity_list)