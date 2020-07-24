from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, constraints, initializers
from openne import graph, node2vec, hope, lap, grarep
from bionev.GAE.train_model import gae_model
from tensorflow import keras
from bionev.utils import *
import pandas as pd
import numpy as np
import random
import math
import copy
import os

vector_size = 160
estimator = 2000

edge = '/mnt/zcs/lncRNA-miRNA/data/edge/2000n/'
cv_result = '/mnt/zcs/lncRNA-miRNA/cv5/2000n/'

miRNA_sequence = '/mnt/zcs/lncRNA-miRNA/data/mi_seq_L.csv'
lncRNA_seqence = '/mnt/zcs/lncRNA-miRNA/data/lnc_seq_L.csv'
lmi = '/mnt/zcs/lncRNA-miRNA/data/LMI_L.csv'
net_edge = edge + 'L_lncRNA_miRNA_edgelist_L.txt'
gae_embedding = edge + 'GAE_emb_L.txt'
final_result = cv_result + '20_seed.csv'
checkpoint_sigmoid_bais = '/mnt/zcs/lncRNA-miRNA/data/temp/2000n/'


class AttLayer(Layer):
    def __init__(self, output_dim, kernel_constraint=None, **kwargs):
        self.output_dim = output_dim
        self.feature_number = 2
        self.kernel_constraint = constraints.get(kernel_constraint)
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(2, self.feature_number),
                                      initializer=keras.initializers.Ones(),
                                      trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x):
        lnc_feature = self.kernel[0][0] * x[::, :vector_size // 2]
        mi_feature = self.kernel[1][0] * x[::, vector_size // 2:vector_size]
        for i in range(1, self.feature_number):
            lnc_feature += self.kernel[0][i] * x[::, vector_size * i:vector_size * i + vector_size // 2]
            mi_feature += self.kernel[1][i] * x[::, vector_size * i + vector_size // 2:vector_size * (i + 1)]
        merge_feature = K.concatenate((lnc_feature, mi_feature))
        return merge_feature

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def DNN():
    # a deep neural network classifier with attention mechanism
    feature1 = Input(shape=(vector_size,), name='Feature1')
    feature2 = Input(shape=(vector_size,), name='Feature2')

    # train_input = keras.backend.stack([train_input1, train_input2])
    feature = keras.layers.Concatenate()([feature1, feature2])
    attention = AttLayer(vector_size, name='attention')(feature)
    # MLP
    dense1 = Dense(120, activation='relu', kernel_initializer='glorot_normal', name='dense1')(attention)
    dense2 = Dense(60, activation='relu', kernel_initializer='glorot_normal', name='dense2')(dense1)
    probality = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', name='probablity')(dense2)

    model = Model(inputs=[feature1, feature2], outputs=probality)
    model.compile(optimizer=keras.optimizers.Adam(lr=3e-4), loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    return model


def model_fit(feature1, feature2, label, seed, k):
    checkpoint_path = checkpoint_sigmoid_bais + 'seed' + str(seed) + 'round' + str(k) + '.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                            save_best_only=True, mode='min', save_weights_only=True, period=1)
    model = DNN()
    model.fit([feature1, feature2], label, batch_size=128, epochs=50,
              validation_split=0.2, verbose=1, callbacks=[cp_callback])

    clf = DNN()
    clf.load_weights(checkpoint_path)
    return clf


def get_LNS(feature_matrix, neighbor_num):
    feature_matrix = np.matrix(feature_matrix)
    iteration_max = 40  # same as 2018 bibm
    mu = 3  # same as 2018 bibm
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    distance_matrix = np.sqrt(alpha + alpha.T - 2 * X * X.T)
    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(0)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return np.array(W)


def get_link_from_similarity(similarity_matrix, positive_num):
    row_num = similarity_matrix.shape[0]
    sort_index = np.argsort(-similarity_matrix, kind='mergesort')  # sort: large to small
    nearest_neighbor_index = sort_index[:, :positive_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(positive_num), nearest_neighbor_index] = 1
    return nearest_neighbor_matrix


def construct_net(train_LMI, positive_num):  #
    miRNA_seq = np.loadtxt(miRNA_sequence, delimiter=',', dtype=float)
    lncRNA_seq = np.loadtxt(lncRNA_seqence, delimiter=',', dtype=float)

    mi_LNS_sim = get_LNS(miRNA_seq, int(len(miRNA_seq) * 0.8))
    lnc_LNS_sim = get_LNS(lncRNA_seq, int(len(lncRNA_seq) * 0.8))

    mi_intra_link = get_link_from_similarity(mi_LNS_sim, positive_num)
    lnc_intra_link = get_link_from_similarity(lnc_LNS_sim, positive_num)
    # lncRNA_matrix = np.matrix(np.zeros((train_LMI.shape[0], train_LMI.shape[0]), dtype=np.int8))
    # miRNA_matrix = np.matrix(np.zeros((train_LMI.shape[1], train_LMI.shape[1]),dtype=np.int8))
    mat1 = np.hstack((lnc_intra_link, train_LMI))
    mat2 = np.hstack((train_LMI.T, mi_intra_link))
    return np.vstack((mat1, mat2))


def net2edgelist(lncRNA_miRNA_matrix_net):  #
    none_zero_position = np.where(np.triu(lncRNA_miRNA_matrix_net) != 0)  #
    none_zero_row_index = np.mat(none_zero_position[0], dtype=int).T
    none_zero_col_index = np.mat(none_zero_position[1], dtype=int).T
    none_zero_dgelist = np.array(np.hstack((none_zero_row_index, none_zero_col_index)))
    np.savetxt(net_edge, none_zero_dgelist, fmt="%d", delimiter=' ')


###############################################################################################
def get_dw_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = node2vec.Node2vec(graph=graph1, path_length=80, num_paths=30, dim=vector_size//2, dw=True)  # deepwalk
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix


def get_hope_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = hope.HOPE(graph=graph1, d=vector_size//2)
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix


def get_lap_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = lap.LaplacianEigenmaps(graph1, rep_size=vector_size//2)
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix


def get_GraRep_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = grarep.GraRep(graph=graph1, Kstep=1, dim=vector_size//2)  # best parameter
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix


# for GAE
def get_embddeing_from_txt(txt_file):
    lines = []
    with open(txt_file, 'r') as f_wr:
        content = f_wr.readlines()
        for i in range(1, len(content)):
            emb = []
            num = content[i].split()
            node = num[0]
            for col in num[1:]:
                emb.append(float(col))
            line = int(node), np.array(emb)
            lines.append(line)
        emb_dict = dict(lines)
        return emb_dict


# for GAE
def get_GAE_embedding_matrix(lncRNA_miRNA_matrix_net, hidden_unit=512):
    cmd_path1 = 'bionev --input %s --output %s  --method GAE  --dimensions %s  --gae_model_selection gcn_vae  --epochs 500  --hidden %s' % (
        net_edge, gae_embedding, (str(vector_size//2)), (str(hidden_unit)))
    os.system(cmd_path1)
    vec = get_embddeing_from_txt(gae_embedding)

    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix


####################################################################################################


def get_individual_emb(train_lncRNA_miRNA_matrix, positive_num):
    #########构建异质网络
    lncRNA_miRNA_matrix_net = construct_net(train_lncRNA_miRNA_matrix, positive_num)
    net2edgelist(np.mat(lncRNA_miRNA_matrix_net))
    graph1 = graph.Graph()
    graph1.read_edgelist(net_edge)

    # dw_lncRNA_miRNA_emb = get_dw_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    # dw_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    # lncRNA_emb_dw = np.array(dw_lncRNA_miRNA_emb[0:dw_lncRNA_len, 0:])  # 吐槽：蜜汁操作，人家都拼好了，非得截下来
    # miRNA_emb_dw = np.array(dw_lncRNA_miRNA_emb[dw_lncRNA_len::, 0:])

    # hope_lncRNA_miRNA_emb = get_hope_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    # hope_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    # lncRNA_emb_hope = np.array(hope_lncRNA_miRNA_emb[0:hope_lncRNA_len, 0:])
    # miRNA_emb_hope = np.array(hope_lncRNA_miRNA_emb[hope_lncRNA_len::, 0:])
    #
    lap_lncRNA_miRNA_emb = get_lap_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    lap_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    lncRNA_emb_lap = np.array(lap_lncRNA_miRNA_emb[0:lap_lncRNA_len, 0:])
    miRNA_emb_lap = np.array(lap_lncRNA_miRNA_emb[lap_lncRNA_len::, 0:])

    GraRep_lncRNA_miRNA_emb = get_GraRep_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    GraRep_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    lncRNA_emb_GraRep = np.array(GraRep_lncRNA_miRNA_emb[0:GraRep_lncRNA_len, 0:])
    miRNA_emb_GraRep = np.array(GraRep_lncRNA_miRNA_emb[GraRep_lncRNA_len::, 0:])

    # GAE_lncRNA_miRNA_emb = get_GAE_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net),
    #                                                 hidden_unit=512)  # using a different function
    # GAE_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    # GAE_lncRNA_emb = np.array(GAE_lncRNA_miRNA_emb[0:GAE_lncRNA_len, 0:])
    # GAE_miRNA_emb = np.array(GAE_lncRNA_miRNA_emb[GAE_lncRNA_len::, 0:])

    return [lncRNA_emb_lap, miRNA_emb_lap, lncRNA_emb_GraRep, miRNA_emb_GraRep]


def get_train_data(train_lncRNA_miRNA_matrix, train_row, train_col, lnc_feature, mi_feature):
    train_feature = []  #
    train_label = []  #

    for num in range(len(train_row)):
        feature_vector = np.append(lnc_feature[train_row[num], :], mi_feature[train_col[num], :])
        train_feature.append(feature_vector)
        train_label.append(train_lncRNA_miRNA_matrix[train_row[num], train_col[num]])  # cautious, label是填0之后的matrix

    train_feature = np.array(train_feature)
    train_label = np.array(train_label)
    return [train_feature, train_label]


def get_test_data(lncRNA_miRNA_matrix, lnc_feature, mi_feature, testPosition):
    test_feature = []
    test_label = []

    for num in range(len(testPosition)):
        feature_vector = np.append(lnc_feature[testPosition[num][0], :], mi_feature[testPosition[num][1], :])
        test_feature.append(feature_vector)
        test_label.append(lncRNA_miRNA_matrix[testPosition[num][0], testPosition[num][1]])  # cautious, 检查取label的matrix

    test_feature = np.array(test_feature)
    test_label = np.array(test_label)
    return [test_feature, test_label]


def get_Metrics(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    recall = sen
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    max_index = np.argmax(f1)
    max_f1 = f1[0, max_index]
    max_accuracy = accuracy[0, max_index]
    max_recall = recall[0, max_index]
    max_spec = spec[0, max_index]
    max_precision = precision[0, max_index]
    return [max_f1, max_accuracy, max_recall, max_spec, max_precision]


def model_evaluate(real_score, predict_score):
    print("model_evaluate")
    aupr = average_precision_score(real_score, predict_score)
    auc = roc_auc_score(real_score, predict_score)
    [f1, accuracy, recall, spec, precision] = get_Metrics(real_score, predict_score)
    return np.array([aupr, auc, f1, accuracy, recall, spec, precision])


def make_prediction(train_feature_matrix, train_label_vector, test_feature_matrix, seed):
    print("make_predicton")
    clf = RandomForestClassifier(random_state=seed, n_estimators=200, oob_score=True, n_jobs=-1)
    clf.fit(train_feature_matrix, train_label_vector)
    predict_y_proba = np.array(clf.predict_proba(test_feature_matrix)[:, 1])
    return predict_y_proba


##lncRNA_miRNA_matrix--LMI seeds--runs
def cross_validation_experiment(lncRNA_miRNA_matrix, seed, k_folds):
    print("cross_validation_experiment_begin")
    none_zero_position = np.where(lncRNA_miRNA_matrix != 0)  # 返回非零元素的坐标的元组 二维
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(lncRNA_miRNA_matrix == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]

    positive_randomlist = [i for i in range(len(none_zero_row_index))]  # 给正标签做个编号 0--number-1
    random.seed(seed)
    random.shuffle(positive_randomlist)  # 保证20次每次正样本的编号顺序不同

    metric = np.zeros((1, 7))
    metric_csv = []
    size_of_cv = int(len(none_zero_row_index) / k_folds)  # 每折相互关联的LM个数

    print("seed = %d, evaluating lncRNA-miRNA......" % (seed))
    for k in range(k_folds):
        print("------cross validation (round = %d)------" % (k + 1))
        if k != k_folds - 1:
            positive_test = positive_randomlist[k * size_of_cv:(k + 1) * size_of_cv]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))
        else:
            positive_test = positive_randomlist[k * size_of_cv::]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))  ##划分完了每一此所用的正样本编号列表

        # # 训练集的1(all) 和 原始的全部的0
        positive_train_row = none_zero_row_index[positive_train]
        positive_train_col = none_zero_col_index[positive_train]
        train_row = np.append(positive_train_row, zero_row_index)
        train_col = np.append(positive_train_col, zero_col_index)
        index3 = [v for v in range(len(train_row))]
        random.seed(seed)
        random.shuffle(index3)
        train_row_shuffle = [train_row[p] for p in index3]
        train_col_shuffle = [train_col[p] for p in index3]

        test_row = none_zero_row_index[positive_test]  # 仅含有1，没有0
        test_col = none_zero_col_index[positive_test]
        train_lncRNA_miRNA_matrix = np.copy(lncRNA_miRNA_matrix)  # lncRNA_miRNA_matrix : 5/5 matrix
        train_lncRNA_miRNA_matrix[test_row, test_col] = 0  # train_lncRNA_miRNA_matrix : 4/5 matrix

        # testing set： testPosition 测试集的1(all) 和 原始的全部的0
        testPosition = []
        test_position_row = list(test_row) + list(zero_row_index)
        test_position_col = list(test_col) + list(zero_col_index)
        # shuffle
        test_index2 = [m for m in range(len(test_position_row))]
        random.seed(seed)
        random.shuffle(test_index2)
        test_position_row_shuffle = [test_position_row[j] for j in test_index2]
        test_position_col_shuffle = [test_position_col[j] for j in test_index2]
        # 组成坐标
        if len(test_position_row) == len(test_position_col):
            for i in range(len(test_position_row)):
                testPosition.append([test_position_row_shuffle[i], test_position_col_shuffle[i]])

        # 注意，网络的结构不应该变，还是取 4/5的矩阵 得到embedding
        [lncRNA_emb_lap, miRNA_emb_lap, lncRNA_emb_GraRep, miRNA_emb_GraRep] = get_individual_emb(
            copy.deepcopy(train_lncRNA_miRNA_matrix), 10)

        [lap_train_feature, lap_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle,
                                                              lncRNA_emb_lap, miRNA_emb_lap)
        [lap_test_feature, lap_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_lap, miRNA_emb_lap,
                                                           testPosition)

        [GraRep_train_feature, GraRep_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle,
                                                                    train_col_shuffle, lncRNA_emb_GraRep,
                                                                    miRNA_emb_GraRep)
        [GraRep_test_feature, GraRep_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_GraRep,
                                                                 miRNA_emb_GraRep, testPosition)

        train_label = GraRep_train_label
        test_label = GraRep_test_label
        train_feature1 = GraRep_train_feature
        test_feature1 = GraRep_test_feature
        train_feature2 = lap_train_feature
        test_feature2 = lap_test_feature
        print(train_feature1.shape)
        print(train_feature2.shape)
        # weight feature
        clf = model_fit(train_feature1, train_feature2, train_label, seed, k)
        # attention
        attention_network = Model(inputs=clf.input, outputs=clf.get_layer('attention').output)
        train_feature = attention_network.predict([train_feature1, train_feature2])
        test_feature = attention_network.predict([test_feature1, test_feature2])

        # train model
        model = RandomForestClassifier(random_state=seed, n_estimators=estimator, oob_score=True, n_jobs=-1)
        model.fit(train_feature, train_label)

        probabibity = np.array(model.predict_proba(test_feature)[:, 1])
        result = model_evaluate(test_label, probabibity)
        print([round(i, 4) for i in result])

        metric_csv.append(result)
        metric += result

    metric = metric / k_folds
    print([round(i, 4) for i in metric[0]])  #
    metric_csv.append(metric[0])

    df = pd.DataFrame(np.array(metric_csv),
                      index=['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5', 'Average'],
                      columns=['AUPR', 'AUC', 'F1', 'ACC', 'REC', 'SPEC', 'PRE'])
    df.to_csv(cv_result + 'seed' + str(seed) + '.csv')

    return metric


if __name__ == "__main__":

    k_folds = 5
    result_csv = []
    l_m_matrix = np.loadtxt(lmi, delimiter=',', dtype=int)
    for seed in range(1, 21):
        # seq similarity top 10 convert to 1
        temp_result = cross_validation_experiment(l_m_matrix, seed, k_folds)
        result_csv.append(temp_result[0])
    df = pd.DataFrame(np.array(result_csv),
                      columns=['AUPR', 'AUC', 'F1', 'ACC', 'REC', 'SPEC', 'PRE'])
    df.to_csv(final_result)

