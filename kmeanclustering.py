#results\2024-08-06_14-35-16

import numpy as np
import argparse
import topmost
from topmost.utils import log, config, static_utils, miscellaneous
import os
from sklearn.cluster import KMeans
import scipy



RESULT_DIR = 'results'
DATA_DIR = 'data'


if __name__ == "__main__":
    parser = topmost.utils.config.new_parser()
    parser.add_argument('--dir_path', type=str)
    args = parser.parse_args()

    dir = args.dir_path

    config_args = config.load_config(os.path.join(dir, 'config.txt'))
    logger = log.setup_logger(
        'main', os.path.join(dir, 'main.log'))

    if config_args.dataset in ['20NG', 'IMDB', 'Rakuten_Amazon', 'YahooAnswers',
                               'NYT', 'ECNews', 'Amazon_Review', 'AGNews']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    if config_args.model in ['YTM', 'ZTM', 'CombinedTM', 'OTClusterTM']:
        dataset = topmost.data.BasicDatasetHandler(
            os.path.join(DATA_DIR, config_args.dataset), device=config_args.device, read_labels=read_labels,
            as_tensor=True, contextual_embed=True, batch_size=config_args.batch_size)
    else:
        dataset = topmost.data.BasicDatasetHandler(
            os.path.join(DATA_DIR, config_args.dataset), device=config_args.device, read_labels=read_labels,
            as_tensor=True, batch_size=config_args.batch_size)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, config_args.dataset, "word_embeddings.npz")).toarray()

    # beta = np.load(os.path.join(dir, 'beta.npy')).reshape(
    #     config_args.num_topics, -1)
    train_theta = np.load(os.path.join(dir, 'train_theta.npy')
                          ).reshape(-1, config_args.num_topics)
    test_theta = np.load(os.path.join(dir, 'test_theta.npy')
                         ).reshape(-1, config_args.num_topics)
    # word_embeddings = np.load(os.path.join(dir, 'word_embeddings.npy'))
    # topic_embeddings = np.load(os.path.join(dir, 'topic_embeddings.npy'))

    purity_list = []
    nmi_list = []
    for K in [19, 29, 39]:
        # seed 43 is da best
        # KMeans clustering
        kmeans = KMeans(n_clusters=K+1, random_state=2)
        kmeans.fit(test_theta)
        pred = kmeans.labels_


        res = topmost.evaluations.evaluate_clustering_with_amax(
            pred, dataset.test_labels)

        purity_list.append(res['Purity'])
        nmi_list.append(res['NMI'])

    print(purity_list)
    print(nmi_list)
