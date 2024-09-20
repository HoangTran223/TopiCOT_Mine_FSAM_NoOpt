import numpy as np
import argparse
import topmost
from topmost.utils import log, config, static_utils, miscellaneous
import os


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
        as_tensor=True)

    beta = np.load(os.path.join(dir, 'beta.npy')).reshape(
        config_args.num_topics, -1)
    train_theta = np.load(os.path.join(dir, 'train_theta.npy')
                          ).reshape(-1, config_args.num_topics)
    test_theta = np.load(os.path.join(dir, 'test_theta.npy')
                         ).reshape(-1, config_args.num_topics)
    word_embeddings = np.load(os.path.join(dir, 'word_embeddings.npy'))
    topic_embeddings = np.load(os.path.join(dir, 'topic_embeddings.npy'))
    topic_dist = np.load(os.path.join(dir, 'topic_dist.npy'))

    n_groups = 10
    n_topics = config_args.num_topics
    n_topics_per_group = n_topics // n_groups

    group_distance = np.zeros((n_groups, n_groups))
    # group_disance[i, j] = average distance between topics in group i and topics in group j

    for i in range(n_groups):
        for j in range(n_groups):
            sum_distance = 0.
            for k in range(n_topics_per_group):
                for l in range(n_topics_per_group):
                    sum_distance += np.linalg.norm(
                        topic_embeddings[i * n_topics_per_group + k] - topic_embeddings[j * n_topics_per_group + l])
            if i == j:
                group_distance[i, j] = sum_distance / \
                    (n_topics_per_group*(n_topics_per_group-1))
            else:
                group_distance[i, j] = sum_distance / \
                    (n_topics_per_group*n_topics_per_group)

    logger.info(f"Group distance:")
    for i in range(len(group_distance)):
        logger.info(f"{group_distance[i]}")

    create_folder_if_not_exist(os.path.join(dir, 'pairwise_group_tsne'))

    # pairwise group tsne visualization
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            else:
                emb_list_i = np.arange(
                    i*n_topics_per_group, (i+1)*n_topics_per_group)
                emb_list_j = np.arange(
                    j*n_topics_per_group, (j+1)*n_topics_per_group)
                miscellaneous.tsne_viz(topic_embeddings[emb_list_i], topic_embeddings[emb_list_j],
                                       os.path.join(dir, 'pairwise_group_tsne', f'{i}_{j}.png'), viz_group=True)

    np.fill_diagonal(group_distance, np.inf)
    argmin_group_distance = np.argmin(group_distance)
    min_index = np.unravel_index(np.argmin(group_distance), group_distance.shape)
    print("argmin_group_distance: ", min_index)
    print("min_group_distance: ", group_distance.min(), group_distance[min_index])
    logger.info(f"argmin_group_distance: {min_index}")
    logger.info(f"min_group_distance: {group_distance.min()} {group_distance[min_index]}")

    print(group_distance[:5, :5])
