import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from tqdm import tqdm
from topmost.utils import static_utils
import os
import scipy
import wandb
import logging


# transform tensor list to numpy list
def to_nparray(tensor_list):
    return np.asarray([item.detach().cpu().numpy() for item in tensor_list], dtype=object)


class HierarchicalTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, rho=0.05):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.rho = rho
        self.logger = logging.getLogger('main')

    def make_optimizer(self,):
        # args_dict = {
        #     'params': self.model.parameters(),
        #     'lr': self.learning_rate,
        #     'rho': self.rho,
        # }
        # optimizer = torch.optim.Adam(**args_dict)
        optimizer = SAM(
            params=self.model.parameters(),
            base_optimizer=lambda params, **kwargs: torch.optim.SGD(params, **kwargs), 
            rho=self.rho,
            lr=self.learning_rate,  # Thay đổi nếu cần thiết
            )
        return optimizer

    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=True)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta


    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1), leave=False):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            wandb.log({'epoch': epoch})

            for batch_data in dataset_handler.train_dataloader:

                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            for key in loss_rst_dict:
                wandb.log({key: loss_rst_dict[key] / data_size})

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                print(output_log)

    def test(self, bow):
        data_size = bow.shape[0]

        num_topics_list = self.model.num_topics_list
        theta_list = [list() for _ in range(len(num_topics_list))]
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = bow[idx]
                batch_theta_list = self.model.get_theta(batch_input)

                for layer_id in range(len(num_topics_list)):
                    theta_list[layer_id].extend(batch_theta_list[layer_id].cpu().numpy().tolist())

        theta = np.empty(len(num_topics_list), object)
        theta[:] = [np.asarray(item) for item in theta_list]

        return np.asarray(theta[-1])

    def export_phi(self):
        phi = to_nparray(self.model.get_phi_list())
        return phi

    def export_beta(self):
        beta_list = to_nparray(self.model.get_beta()[-1])
        return beta_list

    def export_top_words(self, vocab, num_top_words=15, annotation=False):
        beta = self.export_beta()
        # top_words_list = list()

        # for layer in range(beta.shape[0]):
        # print(f"======= Layer: {layer} number of topics: {beta[layer].shape[0]} =======")
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words=num_top_words)

        # if not annotation:
        #     top_words_list.append(top_words)
        # else:
        #     pass
        #     top_words_list.extend([f'L-{layer}_K-{k} {item}' for k, item in enumerate(top_words)])

        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        test_theta = self.test(dataset_handler.test_data)

        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'bottom_word_embeddings'):
            word_embeddings = self.model.bottom_word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings_list'):
            topic_embeddings = self.model.topic_embeddings_list[-1].detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'topic_embeddings_list'):
            group_embeddings = self.model.topic_embeddings_list[-2].detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings
