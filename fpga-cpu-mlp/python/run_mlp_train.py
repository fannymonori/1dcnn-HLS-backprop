import numpy as np
import torch
import os
import random
from sklearn.utils import shuffle

from vergara_drift_batches_dataset import get_drift_channels

np.random.seed(seed=0)
random.seed(0)
torch.manual_seed(0)

def init_weights(shape):
    _weight = torch.empty(shape)
    torch.nn.init.xavier_normal_(_weight)
    _weight = _weight.detach().numpy().astype(np.double)
    return _weight

class MLPNet(torch.nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()

        self.dense = torch.nn.Linear(in_features=128, out_features=128, bias=True)
        self.d1_w = init_weights((128, 128))
        self.d1_b = np.zeros((128)).astype(np.double)

        self.dense2 = torch.nn.Linear(in_features=128, out_features=6, bias=True)
        self.d2_w = init_weights((6, 128))
        self.d2_b = np.zeros((6)).astype(np.double)

        self.dense.weight = torch.nn.Parameter(torch.from_numpy(self.d1_w))
        self.dense.bias = torch.nn.Parameter(torch.from_numpy(self.d1_b))

        self.dense2.weight = torch.nn.Parameter(torch.from_numpy(self.d2_w))
        self.dense2.bias = torch.nn.Parameter(torch.from_numpy(self.d2_b))

        self.h1 = None
        self.out = None

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        self.h1 = self.dense(x)
        self.h1.retain_grad()

        self.r1 = self.tanh(self.h1)
        self.r1.retain_grad()

        self.out = self.dense2(self.r1)
        self.out.retain_grad()

        return self.out


np.set_printoptions(precision=8)

net = MLPNet()
print(net)
loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0, weight_decay=0, dampening=0, nesterov=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = get_drift_channels()

X_train = dataset["X_train"]
num_of_samples = X_train.shape[0]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
Y_train = dataset["Y_train"]

print("Dataset shape", X_train.shape, Y_train.shape)

softm = torch.nn.Softmax(dim=1)

folder_path = "./saved_data/mlp/"
if not os.path.exists(folder_path):
     os.makedirs(folder_path)

#net.load_state_dict(torch.load("./saved_model_new.pt", weights_only=True))

acc_list = []
loss_list = []
num_of_epochs = 50
for e in range(0, num_of_epochs):
    print("EPOCH: ", e)

    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

    online_batch_size = 32
    num_of_splits = int(X_train.shape[0] / online_batch_size) + 1
    num_of_splits_y = int(Y_train.shape[0] / online_batch_size) + 1
    batches_X = np.array_split(X_train, num_of_splits, axis=0)
    batches_Y = np.array_split(Y_train, num_of_splits, axis=0)

    acc = 0
    count = 0
    num_of_samples_batch = 0
    epoch_loss = 0.0
    for batch_X, batch_Y in zip(batches_X, batches_Y):

        optimizer.zero_grad()

        if(batch_X.shape[0] < online_batch_size):
            rand_batch = random.randint(0, count-1)
            rand_index = random.randint(0, batches_X[rand_batch].shape[0] - 1)
            rand_element_x = np.expand_dims(batches_X[rand_batch][rand_index], axis=0)
            rand_element_y = np.expand_dims(batches_Y[rand_batch][rand_index], axis=0)
            batch_X = np.concatenate([batch_X, rand_element_x], axis=0)
            batch_Y = np.concatenate([batch_Y, rand_element_y], axis=0)

        num_of_samples_batch += batch_X.shape[0]

        x = batch_X
        y = torch.from_numpy(batch_Y)

        x_tensor = torch.from_numpy(x)
        y_tensor = y
        x_tensor.requires_grad = True

        optimizer.zero_grad()

        out_net = net(x_tensor)
        output = loss(out_net, y_tensor)

        predicted = softm(out_net.data)
        _, predicted = torch.max(predicted, 1)
        y_argmax = np.argmax(batch_Y, axis=1)
        acc_array = np.asarray((predicted.detach().numpy() == y_argmax)*1)
        acc += acc_array.astype(float).sum()

        output.retain_grad()
        output.backward()

        if count == 0:
            np.savez("gas_mlp_data.npz",
                 input=x.flatten("C"),
                 bias=net.dense.bias.detach().numpy(),
                 ograd=net.h1.grad.detach().numpy().flatten("C"),
                 w=np.swapaxes(net.dense.weight.detach().numpy(), 0, 1).flatten(),
                 w2=np.swapaxes(net.dense2.weight.detach().numpy(), 0, 1).flatten(),
                 bias2=net.dense2.bias.detach().numpy(),
                 y=batch_Y.flatten("C"))

        epoch_loss += float(output)

        optimizer.step()
        count = count + 1

    accuracy = 100 * acc / float(num_of_samples_batch)
    print("Accuracy after %d number of epochs is %f" % (e+1, accuracy))
    print("Average loss is:", epoch_loss / 32.0)
    acc_list.append(accuracy)
    loss_list.append(epoch_loss / 32.0)

test_batches = dataset["test_batches"]
running_acc = 0
for k,v in test_batches.items():
        x_test = v[0]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        y_test = v[1]
        test_eval = torch.nn.Softmax()(net(torch.from_numpy(x_test)))

        _, predicted = torch.max(test_eval.data, 1)
        y_argmax = np.argmax(y_test, axis=1)
        acc_array = np.asarray((predicted.detach().numpy() == y_argmax)*1)
        acc = acc_array.astype(float).sum() / x_test.shape[0]

        print('Accuracy on test dataset # {} is {:.4%}'.format(k, acc))
        running_acc += acc

for k, v in test_batches.items():
        x_test = v[0]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        y_test = v[1]

        online_batch_size = 32
        num_of_splits = int(x_test.shape[0] / online_batch_size) + 1
        num_of_splits_y = int(y_test.shape[0] / online_batch_size) + 1
        batches_X = np.array_split(x_test, num_of_splits, axis=0)
        batches_Y = np.array_split(y_test, num_of_splits, axis=0)

        folder_path = "./saved_data/mlp/test_" + str(k)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        count = 0
        acc = 0
        num_of_samples_batch = 0
        for batch_X, batch_Y in zip(batches_X, batches_Y):

            if (batch_X.shape[0] < online_batch_size):
                rand_batch = random.randint(0, len(batches_X) - 1)
                rand_index = random.randint(0, batches_X[rand_batch].shape[0] - 1)
                rand_element_x = np.expand_dims(batches_X[rand_batch][rand_index], axis=0)
                rand_element_y = np.expand_dims(batches_Y[rand_batch][rand_index], axis=0)
                batch_X = np.concatenate([batch_X, rand_element_x], axis=0)
                batch_Y = np.concatenate([batch_Y, rand_element_y], axis=0)

            num_of_samples_batch += batch_X.shape[0]

            file_name = folder_path + "/" + str(count) + ".npz"
            np.savez(file_name,
                     input=batch_X.flatten("C"),
                     y=batch_Y.flatten("C"))

            test_eval = torch.nn.Softmax()(net(torch.from_numpy(batch_X)))

            _, predicted = torch.max(test_eval.data, 1)
            y_argmax = np.argmax(batch_Y, axis=1)
            acc_array = np.asarray((predicted.detach().numpy() == y_argmax) * 1)
            acc += acc_array.astype(float).sum()

            count = count + 1

        accuracy = 100 * acc / float(num_of_samples_batch)
        print("Accuracy on test batch number %d is %f" % (k, accuracy))
