import numpy as np
import argparse
import topmost
from topmost.utils import log, config, static_utils, miscellaneous
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import scipy

DATA_DIR = 'data'


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


if __name__ == "__main__":
    parser = topmost.utils.config.new_parser()
    parser.add_argument('--dir_path', type=str)
    args = parser.parse_args()

    dir = args.dir_path

    config_args = config.load_config(os.path.join(dir, 'config.txt'))
    logger = log.setup_logger(
        'main', os.path.join(dir, 'main.log'))

    if config_args.dataset in ['20NG', 'IMDB', 'Rakuten_Amazon',
                               'NYT', 'ECNews', 'Amazon_Review']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    dataset = topmost.data.BasicDatasetHandler(
        "./data/" + config_args.dataset, device=config_args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=True)

    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, config_args.dataset, "word_embeddings.npz")).toarray()

    model = topmost.models.MODEL_DICT[config_args.model](vocab_size=dataset.vocab_size,
                                                    doc_embedding=dataset.train_contextual_embed,
                                                    num_topics=config_args.num_topics,
                                                    num_groups=config_args.num_groups,
                                                    dropout=float(config_args.dropout),
                                                    pretrained_WE=pretrainWE if config_args.use_pretrainWE else None,
                                                    weight_loss_ECR=config_args.weight_ECR,
                                                    alpha_ECR=config_args.alpha_ECR,
                                                    weight_loss_DCR=config_args.weight_DCR,
                                                    alpha_DCR=config_args.alpha_DCR,
                                                    weight_loss_TCR=config_args.weight_TCR,
                                                    alpha_TCR=config_args.alpha_TCR,
                                                    beta_temp=config_args.beta_temp)
    
    model = model.to(config_args.device)


    beta = np.load(os.path.join(dir, 'beta.npy')).reshape(
        config_args.num_topics, -1)
    train_theta = np.load(os.path.join(dir, 'train_theta.npy')
                          ).reshape(-1, config_args.num_topics)
    test_theta = np.load(os.path.join(dir, 'test_theta.npy')
                         ).reshape(-1, config_args.num_topics)
    word_embeddings = np.load(os.path.join(dir, 'word_embeddings.npy'))
    topic_embeddings = np.load(os.path.join(dir, 'topic_embeddings.npy'))
    topic_dist = np.load(os.path.join(dir, 'topic_dist.npy'))

    cluster_emb = np.load(os.path.join(dir, 'cluster_embeddings.npy')).reshape(-1, 384)
    ckpt = torch.load(os.path.join(dir, 'checkpoint.pt'))
    model.load_state_dict(ckpt)
    topic_prj = model.topic_emb_prj(torch.from_numpy(topic_embeddings).to('cuda')).detach().cpu().numpy()
    
    assert topic_prj.shape[1] == cluster_emb.shape[1], "The matrices must have the same size of the second axis."

    # Combine the matrices for t-SNE
    combined_matrix = np.vstack((topic_prj, cluster_emb))

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(combined_matrix)

    # Split the t-SNE results back into topics and clusters
    topic_tsne = tsne_results[:topic_prj.shape[0], :]
    cluster_tsne = tsne_results[topic_prj.shape[0]:, :]

    # Plotting
    plt.figure(figsize=(100, 100))

    # Plot topic projections
    for i in range(topic_tsne.shape[0]):
        plt.scatter(topic_tsne[i, 0], topic_tsne[i, 1], c='blue', label='Topic' if i == 0 else "", alpha=0.6)
        plt.annotate(f'{i}', (topic_tsne[i, 0]+random.random(), topic_tsne[i, 1]+random.random()), textcoords="offset points", xytext=(0,10), ha='center')

    # Plot cluster embeddings
    for j in range(cluster_tsne.shape[0]):
        plt.scatter(cluster_tsne[j, 0], cluster_tsne[j, 1], c='red', label='Cluster' if j == 0 else "", alpha=0.6)
        plt.annotate(f'C{j}', (cluster_tsne[j, 0], cluster_tsne[j, 1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('t-SNE of Topic Projections and Cluster Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc='best')
    plt.savefig('viz.png')
