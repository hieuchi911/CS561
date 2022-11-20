import argparse
from math import sqrt
import numpy as np
from random import shuffle
from tqdm import tqdm

class DenseLayer():
    def __init__(self, inp_len, outp_len, lrate):
        self.inp_len = inp_len
        self.outp_len = outp_len
        self.lrate = lrate
        self.weights = np.random.normal(loc=0.0,
                            scale=sqrt(2/(inp_len+outp_len)),
                            size=(inp_len,outp_len))
        
        self.biases = np.zeros(outp_len)
    
    def forward(self, prev_outp):
        # prev_outp: shape(1 or batch_size, inp_len), self.weights of shape(inp_len, outp_len)
        # print("prev_outp is", prev_outp.shape)
        # print("self.weights is", self.weights.shape)
        return np.dot(prev_outp, self.weights) + self.biases

    def backward(self, prev_o, future_dz):
        # prev_o shape (batch, inp_len)
        # future_dz shape (batch, future_outp_len=outp_len)
        # self.weights shape (inp_len, outp_len)
        df_do = np.dot(future_dz, self.weights.T)    # future_z(batch * 1, outp_len) x W(inp_len, outp_len).T ==> df_do(batch * 1,inp_len)

        df_dw = np.dot(prev_o.T, future_dz)  # prev_o(batch * 1,inp_len).T x future_z(batch * 1, outp_len) ==> df_dw(inp_len, outp_len)
        df_db = future_dz.mean(axis=0) * prev_o.shape[0]

        self.biases = self.biases - self.lrate * df_db
        self.weights = self.weights - self.lrate * df_dw
        return df_do    # shape df_do(batch * 1, inp_len)

class SigmoidLayer():
    
    def forward(self, prev_z):
        return 1 / (1 + np.exp(-prev_z))

    def backward(self, prev_z, future_do):
        # future_do shape (batch, outp_len of prev dense)
        # prev_z shape (batch * 1, outp_len of prev dense)
        do_dz = 1 / (1 + np.exp(-prev_z)) * (1 - 1 / (1 + np.exp(-prev_z)))
        df_dz = future_do * do_dz # future_do(batch, outp_len of prev dense) * do_dz(batch * 1, outp_len of prev dense) ==> df_dz(batch * 1, outp_len of prev dense)
        return df_dz # shape df_dz(batch * 1, outp_len of prev dense)


class ANNAgent():
    def __init__(self, layers, inp_len, outp_len, lrate) -> None:    #layers [128,128,64]
        self.layers = []
        self.layers.append(DenseLayer(inp_len, layers[0], lrate))
        # print("Added layer 1 shape ", self.layers[0].weights.shape)
        self.layers.append(SigmoidLayer())
        for i in range(len(layers)-1):
            self.layers.append(DenseLayer(layers[i], layers[i+1], lrate))
            # print(f"Added layer {i+1+2} shape ", self.layers[i].weights.shape)
            self.layers.append(SigmoidLayer())
        self.layers.append(DenseLayer(layers[-1], outp_len, lrate))
        # print(f"Added layer {i+1+2} shape ", self.layers[-1].weights.shape)
        self.layers.append(SigmoidLayer())
    
    def train(self, x, y):
        # print("x is", x.shape)
        layer_outs = []
        inp = x
        for l in self.layers:
            inp = l.forward(inp)
            layer_outs.append(inp)

        layer_inps = [x] + layer_outs
        logit = layer_outs[-1]
        
        loss = crossentropy(logit, y)
        grads = crossentropy_grad(logit, y)

        for l in range(len(self.layers))[::-1]:
            grads = self.layers[l].backward(layer_inps[l], grads)
        
        return loss
    
    def forward(self, x):
        layer_outs = []
        inp = x
        for l in self.layers:
            inp = l.forward(inp)
            layer_outs.append(inp)
        logit = layer_outs[-1]
        ret = logit > 0.5
        return ret * 1

def crossentropy(logits, labels):
    # logits of shape (batch, 1)
    # labels of shape (batch, 1)
    
    return -np.mean(labels * np.log(logits) + (1-labels) * np.log(1-logits)) # -np.mean()

def crossentropy_grad(logits, labels):
    # logits of shape (batch, 1)
    # labels of shape (batch, 1)
    
    return (logits - labels) / (logits * (1 - logits))

def split_data(dataset, labels, folds=False, rate=0.2):
    if not folds:
        tied = list(zip(dataset, labels))
        shuffle(tied)
        return tied[:int(len(tied) * (1-rate))], tied[int(len(tied) * (1-rate)):]
    return

def make_batches():
    return

def train(learner, dataset, batch_size=200, epochs=1000):
    batches = []
    for i in range(int(len(dataset)/batch_size)+1):
        batches.append(np.array(dataset[i*batch_size:min((i+1)*batch_size, len(dataset))]))
    for epoch in range(epochs):
        # for step, batch in enumerate(tqdm(batches, desc="Iteration")):
        for step, batch in enumerate(batches):
            learner.train(np.array([b[0] for b in batch]), np.array([[b[1]] for b in batch]))

def predict(learner, data):
    return learner.forward(np.array([d for d in data]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train',type=str)
    parser.add_argument('train_la',type=str)
    parser.add_argument('test',type=str)

    args = parser.parse_args()

    train_set_corpus = np.loadtxt(args.train, delimiter=",")
    train_labels_corpus = np.loadtxt(args.train_la, delimiter=",")
    
    test_set_corpus = np.loadtxt(args.test, delimiter=",")

    train_set = list(zip(train_set_corpus, train_labels_corpus))
    batch_size = 128

    # trn_dt, val_dt = split_data(train_set_corpus, train_labels_corpus)
    # x_train, y_train = [i[0] for i in trn_dt], [i[1] for i in trn_dt]    
    # x_val, y_val = [i[0] for i in val_dt], [i[1] for i in val_dt]

    layers = [128, 64]

    learner = ANNAgent(layers, 2, 1, lrate=0.01)

    train(learner, train_set)
    preds = predict(learner, test_set_corpus)

    with open("test_predictions.csv", "w") as outp:
        for i in preds:
            outp.write(str(i[0]) + "\n")
    
    # test_labels_corpus = np.loadtxt("resource/asnlib/public/spiral_test_label.csv", delimiter=",")
    # count = 0
    # for i, el in enumerate(preds):
    #     if el[0] == int(test_labels_corpus[i]):
    #         count += 1
    #     # print(el[0], " ---- ", int(train_labels_corpus[i]))
    # print("acc: ", count/len(preds))