import numpy as np
import argparse
import topmost
from topmost.utils import log, config, static_utils, miscellaneous
import os

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
    
    print((train_theta != 0.).sum(axis=0)[:50])
    
    print((test_theta != 0.).sum(axis=0)[:50])
    
    print(train_theta.argmax(axis=1))
    
    print(test_theta.argmax(axis=1))
    
    # tsne visualization
    miscellaneous.tsne_viz(word_embeddings, topic_embeddings,
                           os.path.join(dir, 'tsne.png'))

    top_words10 = static_utils.print_topic_words(
        beta, dataset.vocab, 10)

    top_words15 = static_utils.print_topic_words(
        beta, dataset.vocab, 15)

    # model evaluation
    # TD
    TD10 = topmost.evaluations.compute_topic_diversity(top_words10, _type="TD")
    print(f"TD10: {TD10:.5f}")
    logger.info(f"TD10: {TD10:.5f}")

    TD15 = topmost.evaluations.compute_topic_diversity(top_words15, _type="TD")
    print(f"TD15: {TD15:.5f}")
    logger.info(f"TD15: {TD15:.5f}")

    # # evaluating clustering
    # if read_labels:
    #     clustering_results = topmost.evaluations.evaluate_clustering(
    #         test_theta, dataset.test_labels)
    #     print(f"NMI: ", clustering_results['NMI'])
    #     print(f'Purity: ', clustering_results['Purity'])
    #     logger.info(f"NMI: {clustering_results['NMI']}")
    #     logger.info(f"Purity: {clustering_results['Purity']}")

    # # evaluate classification
    # if read_labels:
    #     classification_results = topmost.evaluations.evaluate_classification(
    #         train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    #     print(f"Accuracy: ", classification_results['acc'])
    #     logger.info(f"Accuracy: {classification_results['acc']}")
    #     print(f"Macro-f1", classification_results['macro-F1'])
    #     logger.info(f"Macro-f1: {classification_results['macro-F1']}")

    NPMI_train_10_list, NPMI_train_10 = topmost.evaluations.compute_topic_coherence(
        dataset.train_texts, dataset.vocab, top_words10, cv_type='c_npmi')
    print(f"NPMI_train_10: {NPMI_train_10:.5f}, NPMI_train_10_list: {NPMI_train_10_list}")
    # wandb.log({"NPMI_train_10": NPMI_train_10})
    logger.info(f"NPMI_train_10: {NPMI_train_10:.5f}")
    logger.info(f'NPMI_train_10 list: {NPMI_train_10_list}')
    # # TC
    # _, TC = topmost.evaluations.topic_coherence.C_V_on_wikipedia(
    #     os.path.join(dir, 'top_words.txt'))
    # print(f"TC: {TC:.5f}")
    # logger.info(f"TC: {TC:.5f}")
