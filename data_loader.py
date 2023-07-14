import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
import os

def get_dataset_size(test_data_path, dataset, window_size):
    if dataset == 'Boiler':
        data = pd.read_csv(test_data_path).values
        return data.shape[0] - window_size + 1

    elif dataset in ['HHAR', 'HAR', 'WISDM','FD','EEG']:
        train_dataset = torch.load(test_data_path)
        return len(train_dataset['samples'])

    else:
        raise Exception('unknown dataset!')


def data_transform(data_path, window_size, segments_length, dataset):
    if dataset == 'Boiler':
        data = pd.read_csv(data_path).values
        data = data[:, 2:]  # remove time step
        feature, label = [], []
        for i in range(window_size - 1, len(data)):
            label.append(data[i, -1])

            sample = []
            for length in segments_length:
                a = data[(i - length + 1):(i + 1), :-1]  # [seq_length, x_dim]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)  # [segments_num , max_length, x_dim]
            sample = np.transpose(sample, axes=((2, 0, 1)))  # [ x_dim , segments_num , window_size]

            feature.append(sample)

        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)
    elif dataset in ['HHAR', 'HAR', 'WISDM','FD','EEG']:
        train_dataset = torch.load(data_path)
        all_data = train_dataset['samples']
        feature, label = [], train_dataset["labels"]
        if len(all_data.shape) < 3:
            all_data = all_data.unsqueeze(2)
        if isinstance(all_data, np.ndarray):
            all_data = torch.from_numpy(all_data)
            label = torch.from_numpy(label).long()
        if all_data.shape.index(min(all_data.shape[1], all_data.shape[2])) == 1:  # make sure the Channels in third dim
            all_data = all_data.permute(0, 2, 1)
        for i in range(len(all_data)):
            data = all_data[i]
            sample = []
            for length in segments_length:
                a = data[(128-length):, :]  # [seq_length, x_dim]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)  # [segments_num , max_length, x_dim]
            sample = np.transpose(sample, axes=((2, 0, 1)))  # [ x_dim , segments_num , window_size]
            feature.append(sample)
        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)

    else:
        raise Exception('unknown dataset!')
    print(data_path, feature.shape)
    return feature, label


def data_generator(data_path, window_size, segments_length, batch_size, dataset, is_shuffle=False):
    feature, label = data_transform(data_path, window_size, segments_length, dataset)

    if is_shuffle:
        feature, label = shuffle(feature, label)

    batch_count = 0
    while True:
        if batch_size * batch_count >= len(label):
            feature, label = shuffle(feature, label)
            batch_count = 0

        start_index = batch_count * batch_size
        end_index = min(start_index + batch_size, len(label))
        batch_feature = feature[start_index: end_index]

        batch_label = label[start_index: end_index]
        batch_length = np.array(segments_length * (end_index - start_index))
        batch_count += 1

        yield batch_feature, batch_label, batch_length



def data_generator2(data_path, window_size, segments_length, batch_size, dataset, is_shuffle=False,pred_len=5):
    feature, label = data_transform2(data_path, window_size, segments_length, dataset,pred_len)

    if is_shuffle:
        feature, label = shuffle(feature, label)

    batch_count = 0
    while True:
        if batch_size * batch_count >= len(label):
            feature, label = shuffle(feature, label)
            batch_count = 0

        start_index = batch_count * batch_size
        end_index = min(start_index + batch_size, len(label))
        batch_feature = feature[start_index: end_index]

        batch_label = label[start_index: end_index]
        batch_length = np.array(segments_length * (end_index - start_index))
        batch_count += 1

        yield batch_feature, batch_label, batch_length

def data_transform2(data_path, window_size, segments_length, dataset,pred_len):
    if dataset == 'Boiler':
        data = pd.read_csv(data_path).values
        data = data[:, 2:]  # remove time step
        feature, label = [], []
        for i in range(window_size - 1, len(data)):
            label.append(data[i, -1])

            sample = []
            for length in segments_length:
                a = data[(i - length + 1):(i + 1), :-1]  # [seq_length, x_dim]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)  # [segments_num , max_length, x_dim]
            sample = np.transpose(sample, axes=((2, 0, 1)))  # [ x_dim , segments_num , window_size]

            feature.append(sample)

        feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.int32)
    elif dataset in ['HHAR', 'HAR', 'WISDM','FD','EEG']:
        train_dataset = torch.load(data_path)
        all_data = train_dataset['samples']
        feature, label = [], []
        if len(all_data.shape) < 3:
            all_data = all_data.unsqueeze(2)
        if isinstance(all_data, np.ndarray):
            all_data = torch.from_numpy(all_data)
            # label = torch.from_numpy(label)
        if all_data.shape.index(min(all_data.shape[1], all_data.shape[2])) == 1:  # make sure the Channels in third dim
            all_data = all_data.permute(0, 2, 1)
        for i in range(len(all_data)):
            data = all_data[i,0:100,:] # sample 取前100
            label.append(np.array(all_data[i,100:100+pred_len,:])) # label取后面的5 10 15 20
            sample = []
            for length in segments_length:
                a = data[(100-length):, :]  # [seq_length, x_dim]
                a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                           mode='constant')  # padding to [window_size, x_dim]
                sample.append(a)

            sample = np.array(sample)  # [segments_num , max_length, x_dim]
            sample = np.transpose(sample, axes=((2, 0, 1)))  # [ x_dim , segments_num , window_size]
            feature.append(sample)
        feature = np.array(feature).astype(np.float32)
        label = np.array(label).astype(np.float32)

    else:
        raise Exception('unknown dataset!')
    print(data_path, feature.shape)
    return feature, label

