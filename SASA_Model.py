import torch
from torch import nn
from torch.nn import LSTM

import torch.nn.functional as F

from sparsemax import Sparsemax



class SASA(nn.Module):

    def __init__(self, max_len, segments_num, input_dim, class_num, h_dim,
                 dense_dim, drop_prob, lstm_layer, coeff):
        super(SASA, self).__init__()

        self.sparse_max = Sparsemax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.max_len = max_len
        self.segments_num = segments_num
        self.feature_dim = input_dim
        self.h_dim = h_dim
        self.dense_dim = dense_dim
        self.drop_prob = drop_prob
        self.lstm_layer = lstm_layer
        self.class_num = class_num
        self.coeff = coeff
        self.base_bone_list = nn.ModuleList(
            [LSTM(input_size=1, hidden_size=self.h_dim, num_layers=self.lstm_layer, batch_first=True)
             for _ in range(0, self.feature_dim)])
        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.ELU()
                                         )

        self.self_attn_K = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),

                                         nn.LeakyReLU()
                                         )
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),

                                         nn.LeakyReLU()
                                         )

        self.classifier = nn.Sequential(nn.BatchNorm1d(self.feature_dim * 2 * self.h_dim),
                                        nn.Linear(self.feature_dim * 2 * self.h_dim, self.dense_dim),
                                        nn.BatchNorm1d(self.dense_dim),
                                        nn.LeakyReLU(),
                                        nn.Dropout(self.drop_prob),
                                        nn.Linear(self.dense_dim, self.class_num))

        # self.predictor = nn.Sequential( nn.Linear(self.feature_dim * 2 * self.h_dim, self.dense_dim),
        #                                 nn.Linear(self.dense_dim, self.pred_len* self.feature_dim))
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, src_x, src_y, tgt_x):
        src_feature, src_intra_aw_list, src_inter_aw_list = self.calculate_feature_alpha_beta(src_x)
        tgt_feature, tgt_intra_aw_list, tgt_inter_aw_list = self.calculate_feature_alpha_beta(tgt_x)
        domain_loss_alpha = []
        domain_loss_beta = []

        y_pred = self.classifier(src_feature)
        y_pred = torch.softmax(y_pred, dim=-1)
        for i in range(self.feature_dim):
            domain_loss_intra = self.mmd_loss(src_struct=src_intra_aw_list[i],
                                              tgt_struct=tgt_intra_aw_list[i], weight=self.coeff)
            domain_loss_inter = self.mmd_loss(src_struct=src_inter_aw_list[i],
                                              tgt_struct=tgt_inter_aw_list[i], weight=self.coeff)
            domain_loss_alpha.append(domain_loss_intra)
            domain_loss_beta.append(domain_loss_inter)

        total_domain_loss_alpha = torch.tensor(domain_loss_alpha).mean()

        total_domain_loss_beta = torch.tensor(domain_loss_beta).mean()

        src_cls_loss = self.cross_entropy(y_pred, src_y)
        total_loss = src_cls_loss + total_domain_loss_beta + total_domain_loss_alpha
        return y_pred, total_loss

    def self_attention(self, Q, K, scale=True, sparse=True, k=3):

        segment_num = Q.shape[1]

        attention_weight = torch.bmm(Q, K.permute(0, 2, 1))

        attention_weight = torch.mean(attention_weight, dim=2, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, segment_num]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1],
                                                                       attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):
        segment_num = Q.shape[1]

        attention_weight = torch.matmul(F.normalize(Q, p=2, dim=-1), F.normalize(K, p=2, dim=-1).permute(0, 1, 3, 2))

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(segment_num, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.segments_num]))

            attention_weight = torch.reshape(attention_weight_sparse, attention_weight.shape)
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def mmd_loss(self, src_struct, tgt_struct, weight):
        # print("================")
        # print(src_struct.shape)
        # print(tgt_struct.shape)
        mean_struct = src_struct - tgt_struct
        # print(mean_struct.shape)
        # print("================")
        delta = torch.mean(mean_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value

    def calculate_feature_alpha_beta(self, x):
        uni_candidate_representation_list = {}

        uni_adaptation_representation = {}

        intra_attn_weight_list = {}
        inter_attn_weight_list = {}

        Hi_list = []

        for i in range(0, self.feature_dim):
            xi = torch.reshape(x[:, i, :, :], shape=[-1, self.max_len, 1])
            _, (candidate_representation_xi, _) = self.base_bone_list[i](xi)

            candidate_representation_xi = torch.reshape(candidate_representation_xi,
                                                        shape=[-1, self.segments_num, self.h_dim])
            # batch_size , segment_num, h_dim
            uni_candidate_representation_list[i] = candidate_representation_xi

            Q_xi = self.self_attn_Q(candidate_representation_xi)
            K_xi = self.self_attn_K(candidate_representation_xi)
            V_xi = self.self_attn_V(candidate_representation_xi)

            intra_attention_weight_xi = self.self_attention(Q=Q_xi, K=K_xi, sparse=True)

            Z_i = torch.bmm(intra_attention_weight_xi.view(intra_attention_weight_xi.shape[0], 1, -1),
                            V_xi)

            intra_attn_weight_list[i]=(torch.squeeze(intra_attention_weight_xi))
            Z_i = F.normalize(Z_i, dim=-1)

            uni_adaptation_representation[i] = Z_i

        for i in range(0, self.feature_dim):
            Z_i = uni_adaptation_representation[i]
            other_candidate_representation_src = torch.stack(
                [uni_candidate_representation_list[j] for j in range(self.feature_dim)], dim=0)

            inter_attention_weight = self.attention_fn(Q=Z_i, K=other_candidate_representation_src, sparse=True)

            U_i_src = torch.mean(torch.matmul(inter_attention_weight, other_candidate_representation_src), dim=0)

            inter_attn_weight_list[i]=(torch.squeeze(inter_attention_weight))
            Hi = torch.squeeze(torch.cat([Z_i, U_i_src], dim=-1), dim=1)
            Hi = F.normalize(Hi, dim=-1)
            Hi_list.append(Hi)
        final_feature = torch.reshape(torch.cat(Hi_list, dim=-1, ),
                                      shape=[x.shape[0], self.feature_dim * 2 * self.h_dim])
        return final_feature, intra_attn_weight_list, inter_attn_weight_list




########## CNN #############################
class CNN(nn.Module):
    def __init__(self, mid_channels, kernel_size, stride, dropout,h_dim):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, mid_channels, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, h_dim, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_in):
        # x = x_in.prmute(0, 2, 1)
        # batch_size seq_len feature_dim
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        # x_flat = x.reshape(x.shape[0], -1)
        return x

class Sasa_Prejection(nn.Module):
    def __init__(self, prej_input_channels,input_channels,pred_len):
        super(Sasa_Prejection, self).__init__()
        self.prej_input_channels = prej_input_channels
        self.input_channels = input_channels
        self.pred_len = pred_len
        # TODO 映射成最后的维度
        self.logits = nn.Linear(self.prej_input_channels, self.input_channels,bias=True)
        # self.configs = configs

    def forward(self, x):
        x1 = x.reshape(x.shape[0],self.pred_len,-1)
        predictions = self.logits(x1)
        return predictions
class SASA_CNN(nn.Module):

    def __init__(self, max_len, segments_num, input_dim, class_num, h_dim,
                 dense_dim, drop_prob, lstm_layer, coeff,mid_channels, kernel_size, stride,dropout,pred_len):
        super(SASA_CNN, self).__init__()

        self.sparse_max = Sparsemax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.max_len = max_len
        self.segments_num = segments_num
        self.feature_dim = input_dim
        self.h_dim = h_dim
        self.dense_dim = dense_dim
        self.drop_prob = drop_prob
        self.lstm_layer = lstm_layer
        self.class_num = class_num
        self.coeff = coeff
        self.pred_len = pred_len
        self.base_bone_list = nn.ModuleList(
            [CNN(mid_channels=mid_channels, kernel_size=kernel_size, stride=stride, dropout=dropout,h_dim=h_dim)
             for _ in range(0, self.feature_dim)])
        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),
                                         nn.ELU()
                                         )
        self.self_attn_K = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),

                                         nn.LeakyReLU()
                                         )
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=self.h_dim, out_features=self.h_dim),

                                         nn.LeakyReLU()
                                         )
        self.classifier = nn.Sequential(nn.BatchNorm1d(self.feature_dim * 2 * self.h_dim),
                                        nn.Linear(self.feature_dim * 2 * self.h_dim, self.dense_dim),
                                        nn.BatchNorm1d(self.dense_dim),
                                        nn.LeakyReLU(),
                                        nn.Dropout(self.drop_prob),
                                        nn.Linear(self.dense_dim, self.class_num))
        prej_input_channels = int(self.h_dim * self.feature_dim * 2 / self.pred_len)
        self.projector = Sasa_Prejection(prej_input_channels,self.feature_dim,self.pred_len)
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, src_x, src_y, tgt_x):
        src_feature, src_intra_aw_list, src_inter_aw_list = self.calculate_feature_alpha_beta(src_x)
        tgt_feature, tgt_intra_aw_list, tgt_inter_aw_list = self.calculate_feature_alpha_beta(tgt_x)
        domain_loss_alpha = []
        domain_loss_beta = []

        y_pred = self.classifier(src_feature)
        y_pred = torch.softmax(y_pred, dim=-1)
        for i in range(self.feature_dim):
            domain_loss_intra = self.mmd_loss(src_struct=src_intra_aw_list[i],
                                              tgt_struct=tgt_intra_aw_list[i], weight=self.coeff)
            domain_loss_inter = self.mmd_loss(src_struct=src_inter_aw_list[i],
                                              tgt_struct=tgt_inter_aw_list[i], weight=self.coeff)
            domain_loss_alpha.append(domain_loss_intra)
            domain_loss_beta.append(domain_loss_inter)

        total_domain_loss_alpha = torch.tensor(domain_loss_alpha).mean()

        total_domain_loss_beta = torch.tensor(domain_loss_beta).mean()

        src_cls_loss = self.cross_entropy(y_pred, src_y)
        total_loss = src_cls_loss + total_domain_loss_beta + total_domain_loss_alpha
        return y_pred, total_loss

    def forecast_forward(self, src_x, src_y, tgt_x):
        src_feature, src_intra_aw_list, src_inter_aw_list = self.calculate_feature_alpha_beta(src_x)
        tgt_feature, tgt_intra_aw_list, tgt_inter_aw_list = self.calculate_feature_alpha_beta(tgt_x)
        domain_loss_alpha = []
        domain_loss_beta = []

        y_pred = self.projector(src_feature)
        for i in range(self.feature_dim):
            domain_loss_intra = self.mmd_loss(src_struct=src_intra_aw_list[i],
                                              tgt_struct=tgt_intra_aw_list[i], weight=self.coeff)
            domain_loss_inter = self.mmd_loss(src_struct=src_inter_aw_list[i],
                                              tgt_struct=tgt_inter_aw_list[i], weight=self.coeff)
            domain_loss_alpha.append(domain_loss_intra)
            domain_loss_beta.append(domain_loss_inter)

        total_domain_loss_alpha = torch.tensor(domain_loss_alpha).mean()

        total_domain_loss_beta = torch.tensor(domain_loss_beta).mean()

        src_cls_loss = self.mse_loss(y_pred, src_y)
        total_loss = src_cls_loss + total_domain_loss_beta + total_domain_loss_alpha
        return y_pred, total_loss

    def self_attention(self, Q, K, scale=True, sparse=True, k=3):

        segment_num = Q.shape[1]

        attention_weight = torch.bmm(Q, K.permute(0, 2, 1))

        attention_weight = torch.mean(attention_weight, dim=2, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, segment_num]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1],
                                                                       attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):
        segment_num = Q.shape[1]

        attention_weight = torch.matmul(F.normalize(Q, p=2, dim=-1), F.normalize(K, p=2, dim=-1).permute(0, 1, 3, 2))

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(segment_num, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, self.segments_num]))

            attention_weight = torch.reshape(attention_weight_sparse, attention_weight.shape)
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def mmd_loss(self, src_struct, tgt_struct, weight):
        mean_struct = src_struct - tgt_struct
        delta = torch.mean(mean_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value

    def calculate_feature_alpha_beta(self, x):
        uni_candidate_representation_list = {}
        uni_adaptation_representation = {}
        intra_attn_weight_list = {}
        inter_attn_weight_list = {}
        Hi_list = []

        for i in range(0, self.feature_dim):
            xi = torch.reshape(x[:, i, :, :], shape=[-1, 1, self.max_len])
            candidate_representation_xi = self.base_bone_list[i](xi)

            candidate_representation_xi = torch.reshape(candidate_representation_xi,
                                                        shape=[-1, self.segments_num, self.h_dim])

            uni_candidate_representation_list[i] = candidate_representation_xi

            Q_xi = self.self_attn_Q(candidate_representation_xi)
            K_xi = self.self_attn_K(candidate_representation_xi)
            V_xi = self.self_attn_V(candidate_representation_xi)

            intra_attention_weight_xi = self.self_attention(Q=Q_xi, K=K_xi, sparse=True)

            Z_i = torch.bmm(intra_attention_weight_xi.view(intra_attention_weight_xi.shape[0], 1, -1),
                            V_xi)

            intra_attn_weight_list[i]=(torch.squeeze(intra_attention_weight_xi))
            Z_i = F.normalize(Z_i, dim=-1)

            uni_adaptation_representation[i] = Z_i

        for i in range(0, self.feature_dim):
            Z_i = uni_adaptation_representation[i]
            other_candidate_representation_src = torch.stack(
                [uni_candidate_representation_list[j] for j in range(self.feature_dim)], dim=0)

            inter_attention_weight = self.attention_fn(Q=Z_i, K=other_candidate_representation_src, sparse=True)

            U_i_src = torch.mean(torch.matmul(inter_attention_weight, other_candidate_representation_src), dim=0)

            inter_attn_weight_list[i]=(torch.squeeze(inter_attention_weight))
            Hi = torch.squeeze(torch.cat([Z_i, U_i_src], dim=-1), dim=1)
            Hi = F.normalize(Hi, dim=-1)
            Hi_list.append(Hi)
        final_feature = torch.reshape(torch.cat(Hi_list, dim=-1, ),
                                      shape=[x.shape[0], self.feature_dim * 2 * self.h_dim])
        return final_feature, intra_attn_weight_list, inter_attn_weight_list



