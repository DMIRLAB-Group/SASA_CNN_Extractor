import os
import math
import pandas
from sklearn.metrics import roc_auc_score

import numpy as np
from data_loader import get_dataset_size, data_generator

import argparse
import torch
from torch import nn

from SASA_Model import SASA,SASA_CNN
from dataset_config import get_dataset_config_class
from hyparams_config import get_hyparams_config_class
import random
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _calc_metrics(pred_labels, true_labels):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)

    accuracy = accuracy_score(true_labels, pred_labels)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



# python main.py -cuda_device 0 -dataset HHAR -batch_size 64 -seed 10 -epochs 40
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-cuda_device', type=str, default='0', help='which gpu to use ')
    parser.add_argument('-dataset', type=str, default='HHAR', help='which dataset ')
    parser.add_argument("-batch_size", type=int, default=512)
    parser.add_argument("-seed", type=int, default=10)
    parser.add_argument('-epochs', type=int, default=40)
    parser.add_argument('-time_interval', type=int, default=2)
    parser.add_argument('-run_time', type=str, default='20230714')

    args = parser.parse_args()

    dataset_config = get_dataset_config_class(args.dataset)()
    hyparams_config = get_hyparams_config_class(args.dataset)()
    args = parser.parse_args()
    device = torch.device("cuda:" + args.cuda_device) if torch.cuda.is_available() else torch.device('cpu')
    setSeed(args.seed)
    root = "logs"
    os.makedirs(root, exist_ok=True)
    record_file = open(os.path.join(root, args.dataset +  "_record_d0.0.txt"), mode="a+")
    for src_id, trg_id in dataset_config.scenarios:
        print(f'source :{src_id}  target:{trg_id}')
        print('data preparing..')

        if args.dataset == 'Boiler':
            src_train_data_path=os.path.join(dataset_config.data_base_path, src_id, 'train.csv')
            tgt_train_data_path=os.path.join(dataset_config.data_base_path, trg_id, 'train.csv')
            tgt_test_data_path=os.path.join(dataset_config.data_base_path, trg_id, 'test.csv')
        else:
            src_train_data_path=os.path.join(dataset_config.data_base_path, "train_" + src_id + '.pt')
            tgt_train_data_path=os.path.join(dataset_config.data_base_path, "train_" + trg_id +'.pt')
            tgt_test_data_path=os.path.join(dataset_config.data_base_path, "test_" + trg_id +'.pt')

        src_train_generator = data_generator(data_path=src_train_data_path,
                                             segments_length=dataset_config.segments_length,
                                             window_size=dataset_config.window_size,
                                             batch_size=args.batch_size, dataset=args.dataset, is_shuffle=True)
        tgt_train_generator = data_generator(data_path=tgt_train_data_path,
                                             segments_length=dataset_config.segments_length,
                                             window_size=dataset_config.window_size,
                                             batch_size=args.batch_size, dataset=args.dataset, is_shuffle=True)

        tgt_test_generator = data_generator(data_path=tgt_test_data_path,
                                            segments_length=dataset_config.segments_length,
                                            window_size=dataset_config.window_size,
                                            batch_size=args.batch_size, dataset=args.dataset, is_shuffle=False)

        tgt_test_set_size = get_dataset_size(tgt_test_data_path,args.dataset, dataset_config.window_size)



        # model = SASA(max_len=dataset_config.window_size, coeff=hyparams_config.coeff,
        #              segments_num=dataset_config.segments_num, input_dim=dataset_config.input_dim,
        #              class_num=dataset_config.class_num,
        #              h_dim=hyparams_config.h_dim, dense_dim=hyparams_config.dense_dim,
        #              drop_prob=hyparams_config.drop_prob,
        #              lstm_layer=hyparams_config.lstm_layer)

        model = SASA_CNN(max_len=dataset_config.window_size, coeff=hyparams_config.coeff,
                        segments_num=dataset_config.segments_num, input_dim=dataset_config.input_dim,
                        class_num=dataset_config.class_num,
                        h_dim=hyparams_config.h_dim, dense_dim=hyparams_config.dense_dim,
                         drop_prob=hyparams_config.drop_prob,
                        lstm_layer=hyparams_config.lstm_layer,
                        kernel_size=dataset_config.kernel_size,
                        stride=dataset_config.stride,
                        mid_channels=dataset_config.mid_channels,
                        dropout=dataset_config.dropout,
                        pred_len=1
                         )

        # darwin：模型参数初始化
        # model.apply(weights_init)

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hyparams_config.learning_rate,
                                     weight_decay=hyparams_config.weight_decay)
        #  darwin：调整学习率
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)

        global_step = 0
        total_train_label_loss = 0.0
        total_train_domain_loss = 0.0

        best_score = 0
        best_accuracy = 0
        best_f1 = 0
        best_step = 0

        while global_step < hyparams_config.training_steps:
            model.train()
            src_train_batch_x, src_train_batch_y, src_train_batch_l = src_train_generator.__next__()

            tgt_train_batch_x, tgt_train_batch_y, tgt_train_batch_l = tgt_train_generator.__next__()

            if src_train_batch_y.shape[0] != tgt_train_batch_y.shape[0] :  #
                continue
            if src_train_batch_x.shape[0] == 1 :
                continue
            src_x = torch.tensor(src_train_batch_x).to(device)
            tgt_x = torch.tensor(tgt_train_batch_x).to(device)
            src_y = torch.tensor(src_train_batch_y).long().to(device)
            batch_y_pred, batch_total_loss = model.forward(src_x=src_x, src_y=src_y, tgt_x=tgt_x)

            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % hyparams_config.test_per_step == 0 and global_step != 0:
                total_tgt_test_label_loss = 0.0
                tgt_test_epoch = int(math.ceil(tgt_test_set_size / float(args.batch_size)))
                tgt_test_y_pred_list = list()
                tgt_test_y_true_list = list()
                for _ in range(tgt_test_epoch):
                    model.eval()
                    with torch.no_grad():
                        test_batch_tgt_x, test_batch_tgt_y, test_batch_tgt_l = tgt_test_generator.__next__()
                        if test_batch_tgt_x.shape[0] == 1:  #
                            continue
                        test_x = torch.tensor(test_batch_tgt_x).to(device)
                        test_y = torch.tensor(test_batch_tgt_y).long().to(device)

                        batch_tgt_y_pred, batch_tgt_total_loss =  model.forward(src_x=test_x, src_y=test_y, tgt_x=torch.clone(test_x))

                        total_tgt_test_label_loss += batch_tgt_total_loss.detach().cpu().numpy()

                        tgt_test_y_pred_list.extend(batch_tgt_y_pred.detach().argmax(dim=1).cpu().numpy())
                        tgt_test_y_true_list.extend(test_y.detach().cpu().numpy())

                mean_tgt_test_label_loss = total_tgt_test_label_loss / tgt_test_epoch
                tgt_test_y_pred_list = np.asarray(tgt_test_y_pred_list)
                tgt_test_y_true_list = np.asarray(tgt_test_y_true_list)

                accuracy,f1 = _calc_metrics(tgt_test_y_pred_list,tgt_test_y_true_list)
                if best_f1 < f1:
                    best_accuracy = accuracy
                    best_f1 = f1

                print("global_steps", global_step, "accuracy", accuracy,"f1", f1)
                print("total loss",mean_tgt_test_label_loss)
                print('\n')

        print("src:%s -- trg:%s , best_result: %g \n\n" % (src_id, trg_id, best_accuracy), file=record_file)
        print("accuracy:%g , f1: %g \n\n" % (best_accuracy, best_f1), file=record_file)
        record_file.flush()

        result_name = 'SASA_CNN_result_' + args.dataset + args.run_time + '.csv'
        if not os.path.exists(result_name):
            df = pandas.DataFrame(
                columns=['source->target','seed',"batch_size",'time_interval',"accuracy","f1"])
            df.to_csv(result_name, index=False)

        results_list = [str(src_id+"->"+trg_id),args.seed,args.batch_size,dataset_config.time_interval,best_accuracy,best_f1]
        save_data = pandas.DataFrame([results_list])
        save_data.to_csv(result_name, mode='a', header=False, index=False)



