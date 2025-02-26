import warnings
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import math
import random

warnings.filterwarnings('ignore')

CURRENT_SAMPLE_LABEL = 0

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def cal_ccc(x, y):
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    rhoc = 2 * sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean()) ** 2)
    return rhoc

def cacu_auc(label, prob):
    temp = list(zip(label, prob))
    rank = [val1 for val1, val2 in sorted(temp, key=lambda x: x[1])]
    rank_list = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    M = sum(label)
    N = len(label) - M
    return (sum(rank_list) - M * (M + 1) / 2) / (M * N)

def buildparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--F_lr', type=float, default=0.001)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--fine_epoch', type=int, default=250)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=0)
    parsed_args = parser.parse_args()
    return parsed_args


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, bias=True):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, adj):
        if x is not None:
            x = F.dropout(x, self.dropout)
            xw = torch.matmul(x, self.weight)
        else:
            x = F.dropout(adj, self.dropout)
            xw = torch.matmul(x, self.weight)
        out = torch.bmm(adj, xw)
        if self.bias is not None:
            out += self.bias
        return F.relu(out)

class E2E(nn.Module):
    def __init__(self, in_channel, out_channel, A_dim):
        super().__init__()
        self.d = A_dim
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1xd = nn.Conv2d(in_channel, out_channel, (self.d, 1))
        self.convdx1 = nn.Conv2d(in_channel, out_channel, (1, self.d))
        torch.nn.init.xavier_uniform_(self.conv1xd.weight)
        torch.nn.init.xavier_uniform_(self.convdx1.weight)

    def forward(self, A):
        A = A.view(-1, self.in_channel, self.d, self.d)
        a = self.conv1xd(A)
        b = self.convdx1(A)
        concat1 = torch.cat([a] * self.d, 2)
        concat2 = torch.cat([b] * self.d, 3)
        return concat1 + concat2

class EdgeEncoder(nn.Module):
    def __init__(self, out_dim, A_dim):
        super().__init__()
        self.e2e = nn.Sequential(
            E2E(in_channel=1, out_channel=8, A_dim=A_dim),
            nn.LeakyReLU(0.33),
            E2E(in_channel=8, out_channel=48, A_dim=A_dim),
            nn.LeakyReLU(0.33),
            nn.BatchNorm2d(48)
        )
        self.e2n = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=out_dim, kernel_size=(1, A_dim)),
            nn.LeakyReLU(0.33),
            nn.BatchNorm2d(32)
        )
        self.reset_weigths()

    def reset_weigths(self):
        stdv = 1.0 / math.sqrt(256)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, adj):
        out = self.e2n(self.e2e(adj))
        out = out.squeeze(-1).permute(0, 2, 1)
        out = F.dropout(out, p=0.33, training=self.training)
        return out

class ECN_enc(nn.Module):
    def __init__(self, nfeat, nhid, node_num, dropout, hiddim, mask_ratio=0.2):
        super(ECN_enc, self).__init__()
        self.mask_ratio = mask_ratio
        self.gcnout = 32
        self.ec = EdgeEncoder(self.gcnout, node_num)
        self.GraphEmbDim = 64
        self.GraphN2G = nn.Sequential(
            nn.Conv2d(in_channels=self.gcnout, out_channels=self.GraphEmbDim, kernel_size=(nfeat, 1)),
            nn.LeakyReLU(0.33),
        )
        self.LSTM_hid = hiddim
        self.dropout = 0.6
        self.gcbn = nn.BatchNorm1d(node_num)

    def GCNEmbed(self, adj):
        x = self.gcbn(F.relu(self.ec(adj)))
        return F.dropout(x, self.dropout, training=self.training)

    def encode(self, adj):
        new_emb = self.GCNEmbed(adj)
        new_emb = self.GraphN2G(new_emb.permute(0, 2, 1).unsqueeze(-1)).squeeze()
        return new_emb

    def forward(self, adj):
        return self.encode(adj)

class AggregateLayer(nn.Module):
    def __init__(self, net_schema, in_layer_shape, out_layer_shape):
        super(AggregateLayer, self).__init__()
        self.net_schema = net_schema
        self.att_agg = nn.ModuleDict()
        for k in net_schema:
            self.att_agg[k] = Attention_AggregateLayer(k, in_layer_shape, out_layer_shape, net_schema)

    def forward(self, x_dict):
        ret_x_dict = {}
        for k in self.att_agg.keys():
            ret_x_dict[k] = self.att_agg[k](x_dict, k)
        return ret_x_dict

class Attention_AggregateLayer(nn.Module):
    def __init__(self, curr_k, in_layer_shape, out_shape, net_schema):
        super(Attention_AggregateLayer, self).__init__()
        self.curr_k = curr_k
        self.net_schema = net_schema
        self.type_att_size = 4 * out_shape
        self.w_query = nn.Parameter(torch.FloatTensor(out_shape, self.type_att_size))
        self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, self.type_att_size))
        self.w_att = nn.Parameter(torch.FloatTensor(out_shape, out_shape))
        nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
        nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
        nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

    def forward(self, x_dict, k):
        self_ft = x_dict[self.curr_k]
        self_ft = self_ft.unsqueeze(0) if self_ft.dim() == 1 else self_ft
        nb_ft_list = [self_ft]
        nb_name = [self.curr_k]
        for key in self.net_schema:
            if key != self.curr_k:
                nb_ft = x_dict[key]
                nb_ft = nb_ft.unsqueeze(0) if nb_ft.dim() == 1 else nb_ft
                nb_ft_list.append(nb_ft)
                nb_name.append(key)
        att_query = torch.mm(self_ft.double(), self.w_query.double()).repeat(len(nb_ft_list), 1)
        att_keys = torch.mm(torch.cat(nb_ft_list, 0).float(), self.w_keys)
        dk = att_query.shape[1]
        att_input = torch.mm(att_keys.double(), att_query.T.double()) / math.sqrt(dk)
        att_input = att_input.mean(dim=1)
        e = F.elu(att_input)
        sizes = [nb_ft_list[0].shape[0], nb_ft_list[1].shape[0], nb_ft_list[2].shape[0]]
        e_split = torch.split(e, sizes)
        score = []
        for i in range(len(nb_ft_list)):
            score.append(torch.sum(e_split[i]))
        score = torch.tensor(score)
        att_score = F.softmax(score, dim=0)
        global CURRENT_SAMPLE_LABEL
        if (not self.training) and (CURRENT_SAMPLE_LABEL == 1):
            data = [[self.curr_k, nb_name, att_score[0].item(), att_score[1].item(), att_score[2].item()]]
            with open('function_path_BD.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
        attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0, 1), dim=1)
        agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
        output = torch.mm(agg_nb_ft, self.w_att.double())
        return output

class Individual_subN(nn.Module):
    def __init__(self, node_num_1, node_num_2, node_num_3, nhid, dropout, hiddim):
        super(Individual_subN, self).__init__()
        self.emb1 = ECN_enc(node_num_1, nhid, node_num_1, dropout, hiddim)
        self.emb2 = ECN_enc(node_num_2, nhid, node_num_2, dropout, hiddim)
        self.emb3 = ECN_enc(node_num_3, nhid, node_num_3, dropout, hiddim)
        self.netschema = ['1', '2', '3']
        self.GraphEmbDim = 64
        self.LSTM_hid = hiddim
        self.embT = nn.Sequential(
            nn.Linear(self.GraphEmbDim * 3, self.LSTM_hid, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33)
        )
        self.cla = nn.Sequential(
            nn.Linear(self.LSTM_hid, 2, bias=True),
        )
        self.bn = nn.BatchNorm1d(self.LSTM_hid)
        self.init_weights(self.embT)
        self.init_weights(self.cla)
        self.agg1 = AggregateLayer(self.netschema, in_layer_shape=64, out_layer_shape=64)
        self.agg2 = AggregateLayer(self.netschema, in_layer_shape=64, out_layer_shape=64)
        self.agg3 = AggregateLayer(self.netschema, in_layer_shape=64, out_layer_shape=64)
        self.agg4 = AggregateLayer(self.netschema, in_layer_shape=64, out_layer_shape=64)

    def init_weights(self, layers):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, adj1, adj2, adj3, PreT=False):
        em1 = self.emb1(adj1)
        em2 = self.emb2(adj2)
        em3 = self.emb3(adj3)
        node_list = {'1': em1, '2': em2, '3': em3}
        x_dict = self.agg1(node_list)
        x_dict = self.agg2(x_dict)
        x_dict = self.agg3(x_dict)
        x_dict = self.agg4(x_dict)
        x_emb = torch.cat([x_dict[key] for key in x_dict], 1)
        emb = torch.tanh(x_emb).float()
        if emb.shape[0] == 1:
            cla_v = self.cla(self.embT(emb))
        else:
            cla_v = self.cla(self.bn(self.embT(emb)))
        return cla_v

def sub_extract(choose_sub_network=1):
    subN = {
        1: "DMN",
        2: "CEN",
        3: "SN",
        0: "others"
    }
    file = "../Dataset/AAL_sub_new.txt"
    with open(file, "r") as f:
        content = f.readlines()
    roi_mapping = [list(map(int, ''.join(line.split()))) for line in content]
    ROIs = np.array([choose_sub_network in roi for roi in roi_mapping])
    data = np.load("../Dataset/BD_HC.npy")
    data[np.isinf(data)] = 1
    data[np.isnan(data)] = 0
    newdata = data[:, ROIs, :][:, :, ROIs]
    return newdata, newdata.shape[-1]

def main(opt):
    filename_labels = '../Dataset/BD_HC_label.npy'
    Y_Data = np.load(filename_labels)
    Y_Data[np.isnan(Y_Data)] = 0

    A_Data1, A_dim_1 = sub_extract(choose_sub_network=opt.subN1)
    A_Data2, A_dim_2 = sub_extract(choose_sub_network=opt.subN2)
    A_Data3, A_dim_3 = sub_extract(choose_sub_network=opt.subN3)

    seeds = [3407, 114151, 42, 404, 5]
    setup_seed(seeds[opt.seed])

    batch = opt.batch
    device = opt.device
    n_splits = opt.n_splits

    print("params:", opt)

    AVE_result = {"acc": 0, "auc": 0, "sen": 0, "spec": 0}

    Skf = StratifiedKFold(n_splits=opt.n_splits, shuffle=True)
    split_index = 0
    split_result = []

    for train_index, test_index in Skf.split(A_Data1, Y_Data):
        split_index += 1
        A_train1, A_test1 = torch.FloatTensor(A_Data1[train_index]), torch.FloatTensor(A_Data1[test_index])
        A_train2, A_test2 = torch.FloatTensor(A_Data2[train_index]), torch.FloatTensor(A_Data2[test_index])
        A_train3, A_test3 = torch.FloatTensor(A_Data3[train_index]), torch.FloatTensor(A_Data3[test_index])
        cla_Y_train, cla_Y_test = torch.LongTensor(Y_Data[train_index]), Y_Data[test_index]

        model = Individual_subN(A_dim_1, A_dim_2, A_dim_3, 128, 0.5, hiddim=opt.hiddim).to(device)
        optimizerF = torch.optim.Adam(model.parameters(), lr=opt.F_lr)
        batch_num = math.ceil(len(A_train1) / opt.batch)

        for epoch in range(0, opt.pret_epoch + opt.fine_epoch):
            model.train()
            cla_loss_epoch = 0
            train_acc = 0
            all_sample_num = 0

            for onebatch in range(0, batch_num):
                A_train1_dev = A_train1[onebatch * batch:(onebatch + 1) * batch].to(device)
                A_train2_dev = A_train2[onebatch * batch:(onebatch + 1) * batch].to(device)
                A_train3_dev = A_train3[onebatch * batch:(onebatch + 1) * batch].to(device)
                cla_label = cla_Y_train[onebatch * batch:(onebatch + 1) * batch]
                cla_label_dev = cla_label.to(device)
                sample_num = len(cla_label_dev)
                all_sample_num += sample_num

                cla_v = model(A_train1_dev, A_train2_dev, A_train3_dev, PreT=False)
                optimizerF.zero_grad()
                cla_loss = torch.nn.functional.cross_entropy(cla_v, cla_label_dev) * 10
                loss = cla_loss
                loss.backward()
                optimizerF.step()
                train_acc += torch.sum((torch.max(cla_v, dim=1)[1].cpu()) == cla_label).item()
                cla_loss_epoch += (cla_loss.item() / sample_num)

            train_acc = train_acc / all_sample_num * 100

            model.eval()
            predictions = []
            with torch.no_grad():
                for i in range(len(A_test1)):
                    global CURRENT_SAMPLE_LABEL
                    CURRENT_SAMPLE_LABEL = int(cla_Y_test[i])
                    sample_adj1 = A_test1[i].unsqueeze(0).to(device)
                    sample_adj2 = A_test2[i].unsqueeze(0).to(device)
                    sample_adj3 = A_test3[i].unsqueeze(0).to(device)
                    cla_v = model(sample_adj1, sample_adj2, sample_adj3, PreT=False)
                    pred = torch.max(cla_v, dim=1)[1].item()
                    predictions.append(pred)
                correct_predictions = sum(p == int(y) for p, y in zip(predictions, cla_Y_test))
                acc = (correct_predictions / len(cla_Y_test)) * 100
                auc = cacu_auc(cla_Y_test.astype(int), predictions)
                tn, fp, fn, tp = confusion_matrix(cla_Y_test.astype(int), predictions).ravel()
                sen = tp / float(tp + fn)
                spec = tn / float(tn + fp)

            output_text = f"split {split_index} epoch {epoch}|cla {cla_loss_epoch:.3f} acc {train_acc:.2f}|EVA ACC={acc:.3f} AUC={auc:.3f} SEN={sen:.3f} SPEC={spec:.3f}"
        split_result.append(output_text)
        AVE_result["acc"] += (acc / n_splits)
        AVE_result["auc"] += (auc / n_splits)
        AVE_result["sen"] += (sen / n_splits)
        AVE_result["spec"] += (spec / n_splits)

    print("=" * 20 + " Results " + "=" * 20)
    for one_result in split_result:
        print(one_result)
    print(AVE_result)
    print("meta-path_BD.csv finished")

if __name__ == "__main__":
    opt = buildparse()
    opt.subN1 = 1
    opt.subN2 = 2
    opt.subN3 = 3
    header_data = [["self.curr_k", "nb_name", "att_score[0]", "att_score[1]", "att_score[2]"]]
    with open('function_path_BD.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(header_data)
    opt.pret_epoch = 0
    opt.device = "cuda:0"
    opt.batch = 32
    opt.seed = 3
    opt.fine_epoch = 80
    opt.F_lr = 0.001
    opt.hiddim = 16
    main(opt)
    print("=" * 40)
