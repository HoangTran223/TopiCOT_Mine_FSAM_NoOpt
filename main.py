from topmost.utils import config, log, miscellaneous, seed
import topmost
import wandb
import os
import numpy as np
import scipy
import torch

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
    
    prj = args.wandb_prj if args.wandb_prj else 'topmost'

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    wandb.init(project=prj, config=args)
    wandb.log({'time_stamp': current_time})

    if args.dataset in ['20NG', 'IMDB', 'Rakuten_Amazon', 'NYT', 'ECNews', '20NGu'
                        'Amazon_Review', 'AGNews', 'YahooAnswers']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    if args.model in ['YTM', 'ZTM', 'CombinedTM', 'OTClusterTM']:
        dataset = topmost.data.BasicDatasetHandler(
            os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
            as_tensor=True, contextual_embed=True, batch_size=args.batch_size)
    else:
        dataset = topmost.data.BasicDatasetHandler(
            os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
            as_tensor=True, batch_size=args.batch_size)
 

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()

    if args.model == "YTM":
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      weight_loss_MMI=args.weight_MMI,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'XTMv2':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_XGR=args.weight_XGR,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      alpha_XGR=args.alpha_XGR,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'XTMv3':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_XGR=args.weight_XGR,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      alpha_XGR=args.alpha_XGR,
                                                      gating_func=args.gating_func,
                                                      weight_global_expert=args.weight_global_expert,
                                                      weight_local_expert=args.weight_local_expert,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'XTMv4':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_XGR=args.weight_XGR,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      alpha_XGR=args.alpha_XGR,
                                                      gating_func=args.gating_func,
                                                      weight_global_expert=args.weight_global_expert,
                                                      weight_local_expert=args.weight_local_expert,
                                                      k=args.k,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'XTM':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_XGR=args.weight_XGR,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      alpha_XGR=args.alpha_XGR,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'ZTM':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_XGR=args.weight_XGR,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      alpha_XGR=args.alpha_XGR,
                                                      weight_loss_MMI=args.weight_MMI,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'ECRTM':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'OTClusterTM':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      doc_embedding=dataset.train_contextual_embed,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      num_data=len(dataset.train_texts),
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      weight_loss_DCR=args.weight_DCR,
                                                      alpha_DCR=args.alpha_DCR,
                                                      weight_loss_TCR=args.weight_TCR,
                                                      alpha_TCR=args.alpha_TCR,
                                                      beta_temp=args.beta_temp)
    elif args.model == 'CombinedTM':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      dropout=args.dropout,
                                                      contextual_embed_size=dataset.contextual_embed_size)
    elif args.model == 'TraCo':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics_list=[args.num_groups, args.num_topics],
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      dropout=args.dropout,
                                                      beta_temp=args.beta_temp,
                                                      weight_loss_TPD=args.weight_TPD, 
                                                      sinkhorn_alpha=args.alpha_TPD)
    elif args.model == 'TraCoECR':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics_list=[args.num_groups, args.num_topics],
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      dropout=args.dropout,
                                                      beta_temp=args.beta_temp,
                                                      weight_loss_TPD=args.weight_TPD, 
                                                      sinkhorn_alpha=args.alpha_TPD,
                                                      weight_loss_ECR=args.weight_ECR, 
                                                      alpha_ECR=args.alpha_ECR)
    else:
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      dropout=args.dropout)

    if args.model == 'YTM':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'XTMv2':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'XTMv3':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'XTMv4':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'XTM':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'ZTM':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'OTClusterTM':
        model.weight_loss_XGR = args.weight_XGR
        model.weight_loss_ECR = args.weight_ECR
    elif args.model == 'ECRTM':
        model.weight_loss_ECR = args.weight_ECR

    # Thêm đoạn này để in ra số lượng tham số của mô hình
    total_params = sum(p.numel() for p in model.parameters())
    print(f"===> Total parameters: {total_params}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"===> Total trainable parameters: {total_trainable_params}")
    model = model.to(args.device)

    # create a trainer
    if args.model in ['TraCo', 'TraCoECR']:
        trainer = topmost.trainers.HierarchicalTrainer(model, epochs=args.epochs,
                                                       learning_rate=args.lr,
                                                       batch_size=args.batch_size,
                                                       lr_scheduler=args.lr_scheduler,
                                                       lr_step_size=args.lr_step_size)
    else:
        trainer = topmost.trainers.BasicTrainer(model, epochs=args.epochs,
                                                learning_rate=args.lr,
                                                batch_size=args.batch_size,
                                                lr_scheduler=args.lr_scheduler,
                                                lr_step_size=args.lr_step_size,
                                                rho=args.rho,
                                                sigma=args.sigma,
                                                lmbda=args.lmbda
                                                )
                                                
    # train the model
    # trainer.train(dataset, MOO=args.MOO_algo)

    trainer.train(dataset)
    
    torch.save(trainer.model.state_dict(), os.path.join(current_run_dir, 'checkpoint.pt'))

    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    # argmax of train and test theta
    train_theta_argmax = train_theta.argmax(axis=1)
    unique_elements, counts = np.unique(train_theta_argmax, return_counts=True)
    print(f'train theta argmax: {unique_elements, counts}')
    logger.info(f'train theta argmax: {unique_elements, counts}')
    test_theta_argmax = test_theta.argmax(axis=1)
    unique_elements, counts = np.unique(test_theta_argmax, return_counts=True)
    print(f'test theta argmax: {unique_elements, counts}')
    logger.info(f'test theta argmax: {unique_elements, counts}')

    # save word embeddings and topic embeddings
    if args.model in ['ETM', 'ECRTM', 'XTM', 'XTMv2', 'YTM', 'XTMv3', 'ZTM', 'OTClusterTM']:
        trainer.save_embeddings(current_run_dir)
        miscellaneous.tsne_viz(model.word_embeddings.detach().cpu().numpy(),
                               model.topic_embeddings.detach().cpu().numpy(),
                               os.path.join(current_run_dir, 'tsne.png'), logwandb=True)
    elif args.model in ['TraCo', 'TraCoECR']:
        trainer.save_embeddings(current_run_dir)
        miscellaneous.tsne_viz(model.bottom_word_embeddings.detach().cpu().numpy(),
                               model.topic_embeddings_list[-1].detach().cpu().numpy(),
                               os.path.join(current_run_dir, 'tsne.png'), logwandb=True)
        

    if args.model in ['XTMv4']:
        # try:
        trainer.save_embeddings(current_run_dir)
        miscellaneous.tsne_group_viz(model.word_embeddings.detach().cpu().numpy(),
                                     model.topic_embeddings.detach().cpu().numpy(),
                                     model.group_embeddings.detach().cpu().numpy(),
                                     os.path.join(current_run_dir,
                                                  'tsne_wt.png'),
                                     os.path.join(current_run_dir, 'tsne_tg.png'))
        # except:
        #     print("VISUALIZE ERROR!!!")
        #     logger.info("VISUALIZE ERROR!!!")

    # if args.model in ['XTMv2', 'XTMv3', 'ZTM']:
    #     miscellaneous.eval_viz_group(args.num_groups, args.num_topics // args.num_groups,
    #                                  model.topic_embeddings.detach().cpu().numpy(), current_run_dir, logger)

    # model evaluation
    # TD
    # TD = topmost.evaluations.compute_topic_diversity(top_words, _type="TD")
    # print(f"TD: {TD:.5f}")
    # wandb.log({"TD": TD})
    # logger.info(f"TD: {TD:.5f}")

    TD_10 = topmost.evaluations.compute_topic_diversity(
        top_words_10, _type="TD")
    print(f"TD_10: {TD_10:.5f}")
    wandb.log({"TD_10": TD_10})
    logger.info(f"TD_10: {TD_10:.5f}")

    TD_15 = topmost.evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    wandb.log({"TD_15": TD_15})
    logger.info(f"TD_15: {TD_15:.5f}")

    # TD_20 = topmost.evaluations.compute_topic_diversity(
    #     top_words_20, _type="TD")
    # print(f"TD_20: {TD_20:.5f}")
    # wandb.log({"TD_20": TD_20})
    # logger.info(f"TD_20: {TD_20:.5f}")

    # TD_25 = topmost.evaluations.compute_topic_diversity(
    #     top_words_25, _type="TD")
    # print(f"TD_25: {TD_25:.5f}")
    # wandb.log({"TD_25": TD_25})
    # logger.info(f"TD_25: {TD_25:.5f}")

    # evaluating clustering
    if read_labels:
        clustering_results = topmost.evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        wandb.log({"NMI": clustering_results['NMI']})
        wandb.log({"Purity": clustering_results['Purity']})
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")

    # evaluate classification
    if read_labels:
        classification_results = topmost.evaluations.evaluate_classification(
            train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=args.tune_SVM)
        print(f"Accuracy: ", classification_results['acc'])
        wandb.log({"Accuracy": classification_results['acc']})
        logger.info(f"Accuracy: {classification_results['acc']}")
        print(f"Macro-f1", classification_results['macro-F1'])
        wandb.log({"Macro-f1": classification_results['macro-F1']})
        logger.info(f"Macro-f1: {classification_results['macro-F1']}")

    # TC
    TC_15_list, TC_15 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")
    wandb.log({"TC_15": TC_15})
    logger.info(f"TC_15: {TC_15:.5f}")
    logger.info(f'TC_15 list: {TC_15_list}')

    # TC_10_list, TC_10 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
    #     os.path.join(current_run_dir, 'top_words_10.txt'))
    # print(f"TC_10: {TC_10:.5f}")
    # wandb.log({"TC_10": TC_10})
    # logger.info(f"TC_10: {TC_10:.5f}")
    # logger.info(f'TC_10 list: {TC_10_list}')

    # NPMI
    NPMI_train_10_list, NPMI_train_10 = topmost.evaluations.compute_topic_coherence(
        dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_npmi')
    print(f"NPMI_train_10: {NPMI_train_10:.5f}, NPMI_train_10_list: {NPMI_train_10_list}")
    wandb.log({"NPMI_train_10": NPMI_train_10})
    logger.info(f"NPMI_train_10: {NPMI_train_10:.5f}")
    logger.info(f'NPMI_train_10 list: {NPMI_train_10_list}')

    NPMI_wiki_10_list, NPMI_wiki_10 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_10.txt'), cv_type='NPMI')
    print(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}, NPMI_wiki_10_list: {NPMI_wiki_10_list}")
    wandb.log({"NPMI_wiki_10": NPMI_wiki_10})
    logger.info(f"NPMI_wiki_10: {NPMI_wiki_10:.5f}")
    logger.info(f'NPMI_wiki_10 list: {NPMI_wiki_10_list}')

    Cp_wiki_10_list, Cp_wiki_10 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_10.txt'), cv_type='C_P')
    print(f"Cp_wiki_10: {Cp_wiki_10:.5f}, Cp_wiki_10_list: {Cp_wiki_10_list}")
    wandb.log({"Cp_wiki_10": Cp_wiki_10})
    logger.info(f"Cp_wiki_10: {Cp_wiki_10:.5f}")
    logger.info(f'Cp_wiki_10 list: {Cp_wiki_10_list}')
    
    # w2v_list, w2v = topmost.evaluations.topic_coherence.compute_topic_coherence(
    #     dataset.train_texts, dataset.vocab, top_words_10, cv_type='c_w2v')
    # print(f"w2v: {w2v:.5f}, w2v_list: {w2v_list}")
    # wandb.log({"w2v": w2v})
    # logger.info(f"w2v: {w2v:.5f}")
    # logger.info(f'w2v list: {w2v_list}')

    wandb.finish()
