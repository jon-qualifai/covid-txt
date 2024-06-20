"""
COVID.py - Using NLP to identify topics within academic literature around lockdown & quarantine

This script analyzes the COVID-19 Open Research Dataset Challenge (CORD-19) corpus to identify topics
around lockdown and quarantine. It performs the following tasks:

1. Queries the corpus to identify articles relevant to lockdown and quarantine.
2. Performs topic modeling on BERT sentence embeddings using UMAP dimensionality reduction and HDBSCAN
   clustering.
3. Generates topic word clouds for the identified topics.

Author: Jon Howells
"""

import os
import zipfile
import itertools
import string
from typing import List, Tuple, Dict
from tqdm import tqdm
import re

import fasttext
import fasttext.util
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from wordcloud import WordCloud, STOPWORDS

import pandas as pd
import numpy as np
import umap
import hdbscan

import matplotlib.pyplot as plt

nltk.download('punkt')


def extract_data(data_dir: str, archive_zip: str, results_zip: str) -> None:
    """
    Extracts data from the given archive and results ZIP files.

    Args:
        data_dir (str): The directory path where the data files are located.
        archive_zip (str): The name of the archive ZIP file.
        results_zip (str): The name of the results ZIP file.
    """
    archive_path = os.path.join(data_dir, archive_zip)
    results_path = os.path.join(data_dir, results_zip)
    archive_dir = os.path.join(data_dir, 'archive')
    results_dir = os.path.join(data_dir, 'results')

    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(archive_dir)

    with zipfile.ZipFile(results_path, 'r') as zip_ref:
        zip_ref.extractall(results_dir)


def load_data(archive_dir: str, results_dir: str) -> pd.DataFrame:
    """
    Loads the data from the extracted files.

    Args:
        archive_dir (str): The directory path where the archive files are extracted.
        results_dir (str): The directory path where the results files are extracted.

    Returns:
        pd.DataFrame: A DataFrame containing the combined corpus data.
    """
    biorxiv_clean = pd.read_csv(os.path.join(results_dir, 'biorxiv_clean.csv'))
    clean_comm_use = pd.read_csv(os.path.join(results_dir, 'clean_comm_use.csv'))
    clean_noncomm_use = pd.read_csv(os.path.join(results_dir, 'clean_noncomm_use.csv'))
    clean_pmc = pd.read_csv(os.path.join(results_dir, 'clean_pmc.csv'))

    corpus_df = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc])

    return corpus_df


def clean_text(corpus: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
    """
    Cleans the text data by splitting it into sentences and removing punctuation-only sentences.

    Args:
        corpus (List[Tuple[str, str]]): A list of tuples containing paper IDs and text content.

    Returns:
        List[Tuple[str, List[str]]]: A list of tuples containing paper IDs and cleaned sentences.
    """
    def clean_sentence(sentence: str) -> str:
        """
        Removes titles before line spaces from a sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The cleaned sentence.
        """
        try:
            clean_sentence = re.findall(r'\n\n(.*)', sentence)[-1]
        except IndexError:
            clean_sentence = sentence
        return clean_sentence

    def clean_sentences(sentences: List[str]) -> List[str]:
        """
        Cleans a list of sentences by removing punctuation-only sentences.

        Args:
            sentences (List[str]): A list of sentences.

        Returns:
            List[str]: A list of cleaned sentences.
        """
        cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]
        cleaned_sentences_no_punctuation = [
            sentence for sentence in cleaned_sentences if sentence not in string.punctuation
        ]
        return cleaned_sentences_no_punctuation

    documents = [(paper_id, nltk.sent_tokenize(document)) for paper_id, document in corpus if isinstance(document, str)]
    documents_clean = [(paper_id, clean_sentences(document)) for paper_id, document in documents]

    return documents_clean


def find_keywords(keywords: List[str]) -> Set[str]:
    """
    Finds additional keywords related to the given keywords using FastText word embeddings.

    Args:
        keywords (List[str]): A list of initial keywords.

    Returns:
        Set[str]: A set of extended keywords, including the initial keywords and related keywords.
    """
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

    keywords_extended = []
    for keyword in keywords:
        keywords_extended.append(keyword)
        nearest_neighbours = [word.lower() for similarity, word in ft.get_nearest_neighbors(keyword)]
        keywords_extended += nearest_neighbours

    keywords_set = set(keywords_extended)
    keywords_set = keywords_set - {'active-shooter', 'boil-water', 'high-security', 'bio-security',
                                   'shutdown.the', 'pre-evacuation', 'biosecurity', 'evacuate',
                                   'evacuation', 'evacuations', 'sahm', 'stay-at', 'stay-at-home-dad',
                                   'stay-at-home-mom', 'stay-at-home-moms', 'stay-at-home-mother',
                                   'stay-at-home-mum', 'sahm'}

    keywords_with_spaces = set([word.replace('-', ' ') for word in list(keywords_set)])
    keywords_set = keywords_set.union(keywords_with_spaces)

    return keywords_set


def query_corpus(documents_clean: List[Tuple[str, List[str]]], keywords_set: Set[str]) -> List[str]:
    """
    Queries the corpus to find relevant sentences around the given keywords.

    Args:
        documents_clean (List[Tuple[str, List[str]]]): A list of tuples containing paper IDs and cleaned sentences.
        keywords_set (Set[str]): A set of keywords to search for.

    Returns:
        List[str]: A list of relevant sentences.
    """
    relevant_sentences = []
    for paper_id, sentences in documents_clean:
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords_set):
                relevant_sentences.append(sentence)

    return relevant_sentences


def get_sentence_embeddings(relevant_sentences: List[str]) -> np.ndarray:
    """
    Generates sentence embeddings using DistilBERT from Hugging Face.

    Args:
        relevant_sentences (List[str]): A list of relevant sentences.

    Returns:
        np.ndarray: A NumPy array containing the sentence embeddings.
    """
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(relevant_sentences, show_progress_bar=True)

    return embeddings


def perform_topic_modeling(embeddings: np.ndarray) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
   """
   Performs topic modeling on the sentence embeddings using UMAP dimensionality reduction and HDBSCAN clustering.

   Args:
       embeddings (np.ndarray): A NumPy array containing the sentence embeddings.

   Returns:
       Tuple[np.ndarray, hdbscan.HDBSCAN]: A tuple containing the UMAP embeddings and the HDBSCAN clustering object.
   """
   umap_embeddings = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit_transform(embeddings)

   cluster = hdbscan.HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)

   return umap_embeddings, cluster


def visualize_clusters(umap_embeddings: np.ndarray, cluster: hdbscan.HDBSCAN, topic_labels: Dict[int, str]) -> None:
   """
   Visualizes the clusters and their respective topic labels.

   Args:
       umap_embeddings (np.ndarray): A NumPy array containing the UMAP embeddings.
       cluster (hdbscan.HDBSCAN): The HDBSCAN clustering object.
       topic_labels (Dict[int, str]): A dictionary mapping topic numbers to their labels.
   """
   def get_centroid(arr: np.ndarray) -> Tuple[float, float]:
       """
       Calculates the centroid of a given array.

       Args:
           arr (np.ndarray): A NumPy array containing the data points.

       Returns:
           Tuple[float, float]: A tuple containing the x and y coordinates of the centroid.
       """
       length = arr.shape[0]
       sum_x = np.sum(arr[:, 0])
       sum_y = np.sum(arr[:, 1])
       x_centre, y_centre = sum_x / length, sum_y / length
       return x_centre, y_centre

   umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
   result = pd.DataFrame(umap_data, columns=['x', 'y'])
   result['labels'] = cluster.labels_

   fig, ax = plt.subplots(figsize=(20, 10))
   outliers = result.loc[result.labels == -1, :]
   clustered = result.loc[result.labels != -1, :]
   plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.5)
   plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.5, cmap='hsv_r')
   plt.colorbar()

   for label in result['labels']:
       if label == -1:
           pass
       else:
           label_name = topic_labels[label]
           coords = result[result['labels'] == label][['x', 'y']].values
           x_pos, y_pos = get_centroid(coords)
           plt.text(x_pos, y_pos, label_name)


def extract_topic_info(docs_df: pd.DataFrame, cluster: hdbscan.HDBSCAN) -> Tuple[Dict[int, List[Tuple[str, float]]], pd.DataFrame]:
   """
   Extracts topic information, including top words and topic sizes, from the clustering results.

   Args:
       docs_df (pd.DataFrame): A DataFrame containing the documents and their assigned topic labels.
       cluster (hdbscan.HDBSCAN): The HDBSCAN clustering object.

   Returns:
       Tuple[Dict[int, List[Tuple[str, float]]], pd.DataFrame]: A tuple containing a dictionary of top words per topic and a DataFrame of topic sizes.
   """
   def c_tf_idf(documents: List[str], m: int, ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[np.ndarray, CountVectorizer]:
       """
       Calculates the TF-IDF matrix for the given documents.

       Args:
           documents (List[str]): A list of documents.
           m (int): The total number of documents.
           ngram_range (Tuple[int, int], optional): The range of n-grams to consider. Defaults to (1, 1).

       Returns:
           Tuple[np.ndarray, CountVectorizer]: A tuple containing the TF-IDF matrix and the CountVectorizer object.
       """
       count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
       t = count.transform(documents).toarray()
       w = t.sum(axis=1)
       tf = np.divide(t.T, w)
       sum_t = t.sum(axis=0)
       idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
       tf_idf = np.multiply(tf, idf)

       return tf_idf, count

   def extract_top_n_words_per_topic(tf_idf: np.ndarray, count: CountVectorizer, docs_per_topic: pd.DataFrame, n: int = 20) -> Dict[int, List[Tuple[str, float]]]:
       """
       Extracts the top n words for each topic based on the TF-IDF scores.

       Args:
           tf_idf (np.ndarray): The TF-IDF matrix.
           count (CountVectorizer): The CountVectorizer object.
           docs_per_topic (pd.DataFrame): A DataFrame containing the documents grouped by topic.
           n (int, optional): The number of top words to extract per topic. Defaults to 20.

       Returns:
           Dict[int, List[Tuple[str, float]]]: A dictionary mapping topic labels to lists of top words and their TF-IDF scores.
       """
       words = count.get_feature_names()
       labels = list(docs_per_topic.Topic)
       tf_idf_transposed = tf_idf.T
       indices = tf_idf_transposed.argsort()[:, -n:]
       top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
       return top_n_words

   def extract_topic_sizes(df: pd.DataFrame) -> pd.DataFrame:
       """
       Extracts the sizes (number of documents) for each topic.

       Args:
           df (pd.DataFrame): A DataFrame containing the documents and their assigned topic labels.

       Returns:
           pd.DataFrame: A DataFrame containing the topic labels and their corresponding sizes.
       """
       topic_sizes = (df.groupby(['Topic'])
                        .Doc
                        .count()
                        .reset_index()
                        .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                        .sort_values("Size", ascending=False))
       return topic_sizes

   docs_df['Doc_ID'] = range(len(docs_df))
   docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

   tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(docs_df))
   top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
   topic_sizes = extract_topic_sizes(docs_df)

   return top_n_words, topic_sizes


def get_article_info(topic: int, docs_df: pd.DataFrame, corpus_df: pd.DataFrame, metadata_df: pd.DataFrame, sentence2paper: Dict[str, str]) -> None:
   """
   Retrieves and prints article information for a given topic.

   Args:
       topic (int): The topic number.
       docs_df (pd.DataFrame): A DataFrame containing the documents and their assigned topic labels.
       corpus_df (pd.DataFrame): A DataFrame containing the corpus data.
       metadata_df (pd.DataFrame): A DataFrame containing the metadata information.
       sentence2paper (Dict[str, str]): A dictionary mapping sentences to their corresponding paper IDs.
   """
   examples = docs_df[docs_df['Topic'] == topic].sample(20, random_state=1)['Doc'].values
   for example in examples:
       try:
           example_paper_id = sentence2paper[example]
           example_title = corpus_df[corpus_df['paper_id'] == example_paper_id]['title'].values[0]
           example_abstract = corpus_df[corpus_df['paper_id'] == example_paper_id]['abstract'].values[0]
           example_link = metadata_df[metadata_df['sha'] == example_paper_id]['url'].values[0]

           if len(example_title) > 0 and len(example_abstract) > 0:
               print('**SENTENCE**')
               print(example)
               print('\n')
               print('**TITLE**')
               print(example_title)
               print('\n')
               print('**ABSTRACT**')
               print(example_abstract)
               print('\n')
               print('**LINK**')
               print(example_link)
               print('\n\n\n')
       except:
           pass


def generate_word_clouds(docs_df: pd.DataFrame, topic_labels: Dict[int, str]) -> None:
   """
   Generates and displays word clouds for each topic.

   Args:
       docs_df (pd.DataFrame): A DataFrame containing the documents and their assigned topic labels.
       topic_labels (Dict[int, str]): A dictionary mapping topic numbers to their labels.
   """
   def plot_cloud(wordcloud: WordCloud) -> None:
       """
       Plots the given word cloud.

       Args:
           wordcloud (WordCloud): The word cloud object to be plotted.
       """
       plt.figure(figsize=(40, 30))
       plt.imshow(wordcloud)
       plt.axis("off")
       plt.show()

   for topic, topic_label in topic_labels.items():
       print(f'Topic Number {topic}')
       print(f'Topic Label {topic_label}')
       text = ' '.join(docs_df[docs_df['Topic'] == topic]['Doc'].values)
       wordcloud = WordCloud(
           width=3000,
           height=2000,
           random_state=1,
           background_color='salmon',
           colormap='Pastel1',
           collocations=False,
           stopwords=STOPWORDS
       ).generate(text)
       plot_cloud(wordcloud)


def main() -> None:
   """
   The main function that orchestrates the entire analysis process.
   """
   data_dir = os.path.join('..', '..', 'data')
   archive_zip = 'archive.zip'
   results_zip = 'results.zip'

   extract_data(data_dir, archive_zip, results_zip)

   archive_dir = os.path.join(data_dir, 'archive')
   results_dir = os.path.join(data_dir, 'results')

   corpus_df = load_data(archive_dir, results_dir)
   metadata_df = pd.read_csv(os.path.join(archive_dir, 'metadata.csv'), low_memory=False)

   keywords = ['quarantine', 'stay-at-home', 'shelter-in-place', 'shutdown', 'lockdown']
   keywords_set = find_keywords(keywords)

   corpus = corpus_df[['paper_id', 'text']].values
   corpus = [(paper_id, document) for paper_id, document in corpus if isinstance(document, str)]

   documents_clean = clean_text(corpus)

   sentence2paper = {}
   for paper_id, doc in documents_clean:
       for sentence in doc:
           sentence2paper[sentence] = paper_id

   relevant_sentences = query_corpus(documents_clean, keywords_set)
   embeddings = get_sentence_embeddings(relevant_sentences)
   umap_embeddings, cluster = perform_topic_modeling(embeddings)

   topic_labels_dict = {
       12: 'quarantine measures',
       16: 'lockdown & quarantine consequences',
       10: 'animals',
       20: 'quarantine length',
       1: 'cellular shutdown',
       22: 'china',
       7: 'historical quarantines',
       15: 'studies on quarantined individuals',
       21: 'wuhan',
       25: 'other epidemics',
       18: 'lockdown effect on transmission rate',
       24: 'quantative metrics',
       3: 'korea',
       23: 'early 2020',
       19: 'lockdown reducing infections',
       6: 'canada',
       17: 'increasing cases',
       2: 'australia',
       8: 'sars',
       13: 'authorities',
       11: 'copyright',
       0: 'electrical systems',
       5: 'cities in lockdown',
       14: 'japan',
       4: 'links',
       9: 'thermoscanners',
       -1: 'outliers'
   }

   visualize_clusters(umap_embeddings, cluster, topic_labels_dict)

   docs_df = pd.DataFrame(relevant_sentences, columns=["Doc"])
   docs_df['Topic'] = cluster.labels_
   top_n_words, topic_sizes = extract_topic_info(docs_df, cluster)

   print("Top Words per Topic:")
   for topic in topic_sizes[topic_sizes['Topic'] != -1]['Topic'].values:
       print(f"Topic {topic}: {topic_labels_dict[topic]}")
       print(pd.DataFrame(top_n_words[topic], columns=['Word', 'TF-IDF Score']))
       print('\n')

   get_article_info(18, docs_df, corpus_df, metadata_df, sentence2paper)
   get_article_info(16, docs_df, corpus_df, metadata_df, sentence2paper)
   get_article_info(5, docs_df, corpus_df, metadata_df, sentence2paper)
   get_article_info(24, docs_df, corpus_df, metadata_df, sentence2paper)

   generate_word_clouds(docs_df, topic_labels_dict)


if __name__ == "__main__":
   main()