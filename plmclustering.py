from topmost.utils import config, log, miscellaneous, seed
import topmost
import os
import numpy as np
import scipy
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm

RESULT_DIR = 'results'
DATA_DIR = 'data'

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    torch.set_printoptions(threshold=10000)
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_logging_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    args = parser.parse_args()

    seed.seedEverything(args.seed)

    if args.dataset in ['20NG', 'IMDB', 'Rakuten_Amazon', 'NYT', 'ECNews', '20NGu'
                        'Amazon_Review', 'AGNews', 'YahooAnswers']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    dataset = topmost.data.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=True)
    
    purity_list = []
    nmi_list = []
    pur50 = []
    nmi50 = []
    pur100 = []
    nmi100 = []
    for K in [49]:
        # KMeans clustering
        kmeans = KMeans(n_clusters=K+1)
        kmeans.fit(dataset.test_contextual_embed)
        pred = kmeans.labels_

        # # DBSCAN clustering
        # eps = 2.0  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
        # min_samples = 2  # The number of samples in a neighborhood for a point to be considered as a core point
        # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # dbscan.fit(dataset.train_contextual_embed)

        # GMM

        # from sklearn.mixture import GaussianMixture
        # # define the model
        # gaussian_model = GaussianMixture(n_components=K)

        # # train the model
        # gaussian_model.fit(dataset.train_contextual_embed)

        # # assign each data point to a cluster
        # gaussian_result = gaussian_model.predict(dataset.train_contextual_embed)

        res = topmost.evaluations.evaluate_clustering_with_amax(
            pred, dataset.test_labels)

        purity_list.append(res['Purity'])
        nmi_list.append(res['NMI'])
        # if args.dataset == '20NG':
        #     pur50.append(0.623)
        #     nmi50.append(0.570)
        #     pur100.append(0.602)
        #     nmi100.append(0.516)
        # if args.dataset == 'IMDB':
        #     pur50.append(0.709)
        #     nmi50.append(0.061)
        #     pur100.append(0.706)
        #     nmi100.append(0.059)
        # if args.dataset == 'YahooAnswers':
        #     pur50.append(0.588)
        #     nmi50.append(0.331)
        #     pur100.append(0.583)
        #     nmi100.append(0.329)
        # if args.dataset == 'AGNews':
        #     pur50.append(0.804)
        #     nmi50.append(0.410)
        #     pur100.append(0.828)
        #     nmi100.append(0.389)

    print('pur: ', purity_list)
    print('nmi: ', nmi_list)
    print('purity: ', purity_list)
    print('nmi: ', nmi_list)
    print('pur50: ', pur50)
    print('nmi50: ', nmi50)
    print('pur100: ', pur100)
    print('nmi100: ', nmi100)
    # plt.plot(purity_list)
    # plt.plot(nmi_list)
    # plt.plot(pur50)
    # plt.plot(nmi50)
    # plt.plot(pur100)
    # plt.plot(nmi100)
    # plt.title(f'{args.dataset} - KMeans')
    # plt.xlabel('Number of topics')
    # plt.ylabel('Purity, NMI')
    # plt.legend(['purity', 'nmi', 'purity50', 'nmi50', 'purity100', 'nmi100'])
    # plt.savefig(f'clusteringres_GMM/{args.dataset}.png')
