import numpy as np
import tensorflow as tf
import torch
import random
import os
import numpy as np
from sklearn.utils import shuffle

from utils.wine_dataset import get_wine_spoilage
from utils.utils import add_bias

np.random.seed(seed=0)
random.seed(0)
torch.manual_seed(0)

class CNN1D(torch.nn.Module):

    def __init__(self, num_of_channels, num_of_classes, signal_length):
        super(CNN1D, self).__init__()

        self.conv1_features = 32
        self.conv_1d_1 = torch.nn.Conv1d(
            in_channels=num_of_channels,
            out_channels=self.conv1_features,
            kernel_size=2,
            bias=True,
            stride=1,
            #padding='same'
            padding=0
        )
        self.activation1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.signal_length1 = 128

        self.conv2_features = 16
        self.conv_1d_2 = torch.nn.Conv1d(
            in_channels=self.conv1_features,
            out_channels=self.conv2_features,
            kernel_size=2,
            bias=True,
            stride=1,
            #padding='same'
            padding=0
        )
        self.activation2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.signal_length2 = 64

        self.conv3_features = 16
        self.conv_1d_3 = torch.nn.Conv1d(
            in_channels=self.conv2_features,
            out_channels=self.conv3_features,
            kernel_size=2,
            bias=True,
            stride=1,
            #padding='same'
            padding=0
        )
        self.activation3 = torch.nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=2)
        self.signal_length3 = 32

        self.conv4_features = 16
        self.conv_1d_4 = torch.nn.Conv1d(
            in_channels=self.conv3_features,
            out_channels=self.conv4_features,
            kernel_size=2,
            bias=True,
            stride=1,
            #padding='same'
            padding=0
        )
        self.activation4 = torch.nn.ReLU()
        self.signal_length4 = 32

        self.conv5_features = 16
        self.conv_1d_5 = torch.nn.Conv1d(
            in_channels=self.conv5_features,
            out_channels=self.conv5_features,
            kernel_size=2,
            bias=True,
            stride=1,
            #padding='same'
            padding=0
        )
        self.activation5 = torch.nn.ReLU()
        self.signal_length5 = 32

        self.linear = torch.nn.LazyLinear(num_of_classes)
        self.softmax = torch.nn.Softmax()

        self.signal_length = signal_length

        #self.activation4 = torch.nn.ReLU()

        self.z1 = None
        self.z2 = None
        self.z3 = None

        self.a1 = None
        self.a2 = None
        self.a3 = None

        self.mp1 = None
        self.mp2 = None
        self.mp3 = None

        self.dense1 = None
        #self.dense2 = None

        self.count = 0

    def run_conv_im2col_fw(self, x_, b_, w_im2col, out_channels, signal_length, layername):
        x_torch_im2col = torch.from_numpy(np.expand_dims(x_, axis=-1))
        unfold = torch.nn.Unfold(kernel_size=(3, 1))
        x_torch_im2col = unfold(x_torch_im2col).detach().numpy()
        z_im2col_manual = np.matmul(w_im2col, x_torch_im2col[0])
        z_im2col_manual = add_bias(z_im2col_manual, b_, out_channels, signal_length)
        np.savetxt('logs/full_training/fw/manual/manual_z_' + layername + '.txt', z_im2col_manual, fmt='%1.8f')

    def run_conv_im2col_dX(self, zgrad, w_im2col, signal_length, k_size, layername):
        w_im2col_transposed = w_im2col.swapaxes(0, 1).copy()
        x_grad_manual = np.matmul(w_im2col_transposed, zgrad[0])
        np.savetxt('logs/full_training/bw/manual/manual_conv_dx_unfolded' + layername + '.txt', x_grad_manual,
                   fmt='%1.8f')
        fold = torch.nn.Fold(output_size=(signal_length, 1), kernel_size=(k_size, 1))
        x_grad_manual_col2im = fold(torch.from_numpy(x_grad_manual)).detach().numpy()
        np.savetxt('logs/full_training/bw/manual/manual_conv_dx_' + layername + '.txt', x_grad_manual_col2im[:, :, 0], fmt='%1.8f')

    def run_conv_im2col_dW(self, x, z_grad, k_size, layername):
        x_torch_im2col = torch.from_numpy(np.expand_dims(x, axis=-1))
        unfold = torch.nn.Unfold(kernel_size=(k_size, 1))
        x_im2col = unfold(x_torch_im2col).detach().numpy()
        x_tensor_im2col_tr = np.swapaxes(x_im2col, 1, 2)
        dw2_manual = np.matmul(z_grad, x_tensor_im2col_tr)
        np.savetxt('logs/full_training/bw/manual/manual_conv_dw_' + layername + '.txt', dw2_manual[0], fmt='%1.8f')


    def forward(self, x):
        self.z1 = self.conv_1d_1(x)
        self.z1.retain_grad()
        self.a1 = self.activation1(self.z1)
        self.a1.retain_grad()
        self.mp1 = self.maxpool1(self.a1)
        self.mp1.retain_grad()

        self.z2 = self.conv_1d_2(self.mp1)
        self.z2.retain_grad()
        self.a2 = self.activation2(self.z2)
        self.a2.retain_grad()
        self.mp2 = self.maxpool2(self.a2)
        self.mp2.retain_grad()

        self.z3 = self.conv_1d_3(self.mp2)
        self.z3.retain_grad()
        self.a3 = self.activation3(self.z3)
        self.a3.retain_grad()

        self.z4 = self.conv_1d_4(self.a3)
        self.z4.retain_grad()
        self.a4 = self.activation4(self.z4)
        self.a4.retain_grad()

        self.z5 = self.conv_1d_5(self.a4)
        self.z5.retain_grad()
        self.a5 = self.activation5(self.z5)
        self.a5.retain_grad()

        self.flat = torch.nn.Flatten()(self.a5)
        self.flat.retain_grad()

        self.dense1 = self.linear(self.flat)
        self.dense1.retain_grad()
        self.output = self.dense1

        return self.output


def read_data(filename):
    fd = open(filename, 'r')
    lines = fd.readlines()

    count = 0
    labels = list()
    samples = list()
    for l in lines:
        if count % 2 == 0:
            label = np.asarray(l.split()).astype(float)
            labels.append(label)
        else:
            sample = np.asarray(l.split()).astype(float)
            samples.append(sample)

        count = count + 1

    signal_length = 100
    num_of_sensors = 6

    X= np.asarray(samples)
    Y = np.asarray(labels)

    X = X.reshape((X.shape[0], signal_length, num_of_sensors))
    X = np.pad(X, ((0, 0), (0, 128 - 100), (0, 0)), 'constant')

    return X, Y

def save_layer_im2col(layer_orig, kernel_size, file_path, pad_to=16):
    layer_torch_im2col = torch.from_numpy(np.expand_dims(layer_orig, axis=-1))
    unfold = torch.nn.Unfold(kernel_size=(kernel_size, 1))
    layer_torch_im2col = unfold(layer_torch_im2col).detach().numpy()
    padding = pad_to - (layer_torch_im2col.shape[1] % pad_to)
    layer_torch_im2col = np.pad(layer_torch_im2col, ((0, 0), (0, padding)), 'constant')
    np.savetxt(file_path, layer_torch_im2col, fmt='%1.8f')
    append_string(file_path, str(layer_torch_im2col.shape))
    return layer_torch_im2col

def save_layer_pad(layer, file_path, pad_to=16):
    padding = pad_to - (layer.shape[1] % pad_to)
    layer_padded = np.pad(layer, ((0, 0), (0, padding)), 'constant')
    np.savetxt(file_path, layer_padded, fmt='%1.7f')
    return layer_padded

def append_string(filename, string_to_append):
    with open(filename, 'a') as f:
        f.write(''.join(str(string_to_append)))
        f.write("\n")

def save_numpy(filename, matrix):
    np.savetxt(filename, matrix, fmt='%1.8f')
    append_string(filename, str(matrix.shape))

def run():
    if not os.path.exists("logs/full_training_2/fw/pytorch/"):
        os.makedirs("logs/full_training_2/fw/pytorch/")

    if not os.path.exists("logs/full_training_2/fw/im2col/"):
        os.makedirs("logs/full_training_2/fw/im2col/")

    if not os.path.exists("logs/full_training_2/bw/pytorch/"):
        os.makedirs("logs/full_training_2/bw/pytorch/")

    tf.random.set_seed(0)

    wine_dataset = get_wine_spoilage()

    X_train = wine_dataset["X_train"]
    y_train = wine_dataset["y_train"]
    X_valid = wine_dataset["X_valid"]
    y_valid = wine_dataset["y_valid"]
    X_test = wine_dataset["X_test"]
    y_test = wine_dataset["y_test"]

    X_train = np.concatenate((X_train, X_valid))
    y_train = np.concatenate((y_train, y_valid))

    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    print("Wine dataset train shape: ", X_train.shape, y_train.shape)
    print("Wine dataset test shape: ", X_test.shape, y_test.shape)

    original_signal_length = 100
    signal_length = 128
    num_of_sensors = 6
    num_of_classes = 3

    X_train = np.pad(X_train, ((0, 0), (0, 0), (0, signal_length - original_signal_length)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (0, 0), (0, signal_length - original_signal_length)), 'constant')

    print("Wine dataset train shape after padding: ", X_train.shape, y_train.shape)
    print("Wine dataset test shape after padding: ", X_test.shape, y_test.shape)

    print(" Shape of X = ", X_train.shape)
    print(" Shape of Y = ", y_train.shape)

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    small_cnn_model = CNN1D(num_of_sensors, num_of_classes, signal_length)
    print(small_cnn_model)

    num_of_samples = X_train.shape[0]

    loss = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(small_cnn_model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(small_cnn_model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    softm = torch.nn.Softmax(dim=1)
    small_cnn_model.train()

    np.set_printoptions(precision=8)
    for e in range(0, 20):
        X_train, y_train = shuffle(X_train, y_train, random_state=0)

        acc = 0
        for i in range(0, X_train.shape[0]):
            optimizer.zero_grad()
            x = np.expand_dims(X_train[i], axis=0)
            x_tensor = torch.from_numpy(x)
            y = np.expand_dims(y_train[i], axis=0)
            y_tensor = torch.from_numpy(y)

            out_net = small_cnn_model(x_tensor.float())
            output = loss(out_net, y_tensor)

            output.backward()

            if e == 0 and i == 0:
                print("Save reference files")
                ############################################
                #### SAVING GROUND TRUTH PYTORCH VALUES ####

                ### INPUT and LABEL
                save_numpy('logs/full_training_2/fw/pytorch/x.txt', np.squeeze(x, axis=0))
                save_numpy('logs/full_training_2/fw/pytorch/y.txt', np.squeeze(y, axis=0))

                ### OUTPUT
                save_numpy('logs/full_training_2/fw/pytorch/out.txt', np.squeeze(softm(out_net).detach().numpy(), axis=0))

                ### BIASES
                b1 = small_cnn_model.conv_1d_1.bias.detach().numpy()
                b2 = small_cnn_model.conv_1d_2.bias.detach().numpy()
                b3 = small_cnn_model.conv_1d_3.bias.detach().numpy()
                b4 = small_cnn_model.conv_1d_4.bias.detach().numpy()
                b5 = small_cnn_model.conv_1d_5.bias.detach().numpy()
                dense_b1 = small_cnn_model.linear.bias.detach().numpy()
                save_numpy('logs/full_training_2/fw/pytorch/b1.txt', b1)
                save_numpy('logs/full_training_2/fw/pytorch/b2.txt', b2)
                save_numpy('logs/full_training_2/fw/pytorch/b3.txt', b3)
                save_numpy('logs/full_training_2/fw/pytorch/b4.txt', b4)
                save_numpy('logs/full_training_2/fw/pytorch/b5.txt', b5)

                ### WEIGHTS
                w1 = small_cnn_model.conv_1d_1.weight.detach().numpy()
                w1_im2col = np.reshape(w1, (w1.shape[0], w1.shape[1] * w1.shape[2]))
                save_numpy('logs/full_training_2/fw/pytorch/w1.txt', w1_im2col.copy())
                w1_im2col_transposed = w1_im2col.swapaxes(0, 1).copy()
                save_numpy('logs/full_training_2/fw/pytorch/w1_tr.txt', w1_im2col_transposed.copy())
                w2 = small_cnn_model.conv_1d_2.weight.detach().numpy()
                w2_im2col = np.reshape(w2, (w2.shape[0], w2.shape[1] * w2.shape[2]))
                save_numpy('logs/full_training_2/fw/pytorch/w2.txt', w2_im2col.copy())
                w2_im2col_transposed = w2_im2col.swapaxes(0, 1).copy()
                save_numpy('logs/full_training_2/fw/pytorch/w2_tr.txt', w2_im2col_transposed.copy())
                w3 = small_cnn_model.conv_1d_3.weight.detach().numpy()
                w3_im2col = np.reshape(w3, (w3.shape[0], w3.shape[1] * w3.shape[2]))
                save_numpy('logs/full_training_2/fw/pytorch/w3.txt', w3_im2col.copy())
                w3_im2col_transposed = w3_im2col.swapaxes(0, 1).copy()
                save_numpy('logs/full_training_2/fw/pytorch/w3_tr.txt', w3_im2col_transposed.copy())
                w4 = small_cnn_model.conv_1d_4.weight.detach().numpy()
                w4_im2col = np.reshape(w4, (w4.shape[0], w4.shape[1] * w4.shape[2]))
                save_numpy('logs/full_training_2/fw/pytorch/w4.txt', w4_im2col.copy())
                w4_im2col_transposed = w4_im2col.swapaxes(0, 1).copy()
                save_numpy('logs/full_training_2/fw/pytorch/w4_tr.txt', w4_im2col_transposed.copy())
                w5 = small_cnn_model.conv_1d_5.weight.detach().numpy()
                w5_im2col = np.reshape(w5, (w5.shape[0], w5.shape[1] * w5.shape[2]))
                save_numpy('logs/full_training_2/fw/pytorch/w5.txt', w5_im2col.copy())
                w5_im2col_transposed = w5_im2col.swapaxes(0, 1).copy()
                save_numpy('logs/full_training_2/fw/pytorch/w5_tr.txt', w5_im2col_transposed.copy())

                dense_w1 = small_cnn_model.linear.weight.detach().numpy().transpose()
                save_numpy('logs/full_training_2/fw/pytorch/dense_w1.txt', dense_w1)

                # PAD dense weight to a factor of 2 to adhere to tiled layout
                pad = 16 - (int(dense_w1.shape[0] / 16) - 16)
                padding = ((0, 0), (0, pad), (0, 0))
                dense_w1_reshaped = np.reshape(dense_w1, (16, int(dense_w1.shape[0] / 16), dense_w1.shape[1]))
                dense_w1_reshaped_padded = np.pad(dense_w1_reshaped, padding, mode='constant')
                dense_w1_reshaped_padded = np.reshape(dense_w1_reshaped_padded, (16 * 32, dense_w1.shape[1]))

                save_numpy('logs/full_training_2/fw/pytorch/dense_w1_padded.txt', dense_w1_reshaped_padded)
                save_numpy('logs/full_training_2/fw/pytorch/dense_b1.txt', dense_b1)

                ### LAYER OUTPUTS
                save_numpy('logs/full_training_2/fw/pytorch/l1_1_a1_pytorch.txt', small_cnn_model.a1.detach().numpy()[0]) #conv
                save_numpy('logs/full_training_2/fw/pytorch/l1_2_z1_pytorch.txt', small_cnn_model.z1.detach().numpy()[0]) #relu
                save_numpy('logs/full_training_2/fw/pytorch/l1_3_mp1_pytorch.txt', small_cnn_model.mp1.detach().numpy()[0]) #mp1

                save_numpy('logs/full_training_2/fw/pytorch/l2_1_a2_pytorch.txt', small_cnn_model.a2.detach().numpy()[0]) #conv
                save_numpy('logs/full_training_2/fw/pytorch/l2_2_z2_pytorch.txt', small_cnn_model.z2.detach().numpy()[0]) #relu
                save_numpy('logs/full_training_2/fw/pytorch/l2_3_mp2_pytorch.txt', small_cnn_model.mp2.detach().numpy()[0]) #mp1

                save_numpy('logs/full_training_2/fw/pytorch/l3_1_a3_pytorch.txt', small_cnn_model.a3.detach().numpy()[0]) #conv
                save_numpy('logs/full_training_2/fw/pytorch/l3_2_z3_pytorch.txt', small_cnn_model.z3.detach().numpy()[0]) #relu

                save_numpy('logs/full_training_2/fw/pytorch/l4_1_a4_pytorch.txt', small_cnn_model.a4.detach().numpy()[0]) #conv
                save_numpy('logs/full_training_2/fw/pytorch/l4_2_z4_pytorch.txt', small_cnn_model.z4.detach().numpy()[0]) #relu

                save_numpy('logs/full_training_2/fw/pytorch/l5_1_a5_pytorch.txt', small_cnn_model.a5.detach().numpy()[0]) #conv
                save_numpy('logs/full_training_2/fw/pytorch/l5_2_z5_pytorch.txt', small_cnn_model.z5.detach().numpy()[0]) #relu

                save_numpy('logs/full_training_2/fw/pytorch/l6_flat_pytorch.txt', small_cnn_model.flat.detach().numpy()[0])

                save_numpy('logs/full_training_2/fw/pytorch/l7_dense1_pytorch.txt', small_cnn_model.dense1.detach().numpy()[0])

                save_numpy('logs/full_training_2/bw/pytorch/d_dense_w1.txt', small_cnn_model.linear.weight.grad.detach().numpy().transpose())

                ### Ground truth BW
                # WEIGHT GRADS
                dw1 = small_cnn_model.conv_1d_1.weight.grad.detach().numpy()
                dw1 = np.reshape(dw1, (dw1.shape[0], dw1.shape[1] * dw1.shape[2]))
                dw2 = small_cnn_model.conv_1d_2.weight.grad.detach().numpy()
                dw2 = np.reshape(dw2, (dw2.shape[0], dw2.shape[1] * dw2.shape[2]))
                dw3 = small_cnn_model.conv_1d_3.weight.grad.detach().numpy()
                dw3 = np.reshape(dw3, (dw3.shape[0], dw3.shape[1] * dw3.shape[2]))
                save_numpy('logs/full_training_2/bw/pytorch/l1_dw_pytorch.txt', dw1)
                save_numpy('logs/full_training_2/bw/pytorch/l2_dw_pytorch.txt', dw2)
                save_numpy('logs/full_training_2/bw/pytorch/l3_dw_pytorch.txt', dw3)

                save_numpy('logs/full_training_2/bw/pytorch/l1_dw_pytorch_tr.txt', dw1.transpose().copy())
                save_numpy('logs/full_training_2/bw/pytorch/l2_dw_pytorch_tr.txt', dw2.transpose().copy())
                save_numpy('logs/full_training_2/bw/pytorch/l3_dw_pytorch_tr.txt', dw3.transpose().copy())

                ### DENSE layers
                save_numpy('logs/full_training_2/bw/pytorch/l5_dense1_dx_pytorch.txt', small_cnn_model.dense1.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l4_flat_dx_pytorch.txt',small_cnn_model.flat.grad.detach().numpy()[0])

                ### ReLU BW
                save_numpy('logs/full_training_2/bw/pytorch/l1_1_dz1_pytorch.txt', small_cnn_model.z1.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l2_1_dz2_pytorch.txt', small_cnn_model.z2.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l3_1_dz3_pytorch.txt', small_cnn_model.z3.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l4_1_dz4_pytorch.txt', small_cnn_model.z4.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l5_1_dz5_pytorch.txt', small_cnn_model.z5.grad.detach().numpy()[0])

                ### MP BW
                save_numpy('logs/full_training_2/bw/pytorch/l1_2_da1_pytorch.txt', small_cnn_model.a1.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l2_2_da2_pytorch.txt', small_cnn_model.a2.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l3_2_da3_pytorch.txt', small_cnn_model.a3.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l4_2_da4_pytorch.txt', small_cnn_model.a4.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l5_2_da5_pytorch.txt', small_cnn_model.a5.grad.detach().numpy()[0])

                ### CONV BW
                save_numpy('logs/full_training_2/bw/pytorch/l1_3_dmp1_pytorch.txt', small_cnn_model.mp1.grad.detach().numpy()[0])
                save_numpy('logs/full_training_2/bw/pytorch/l2_3_dmp2_pytorch.txt', small_cnn_model.mp2.grad.detach().numpy()[0])

                # CONV DX
                z1_grad = save_layer_pad(small_cnn_model.z1.grad.detach().numpy()[0], 'logs/full_training_2/bw/pytorch/l1_2_dz1_pytorch_padded.txt', 16)
                z2_grad = save_layer_pad(small_cnn_model.z2.grad.detach().numpy()[0], 'logs/full_training_2/bw/pytorch/l2_2_dz2_pytorch_padded.txt', 16)
                z3_grad = save_layer_pad(small_cnn_model.z3.grad.detach().numpy()[0], 'logs/full_training_2/bw/pytorch/l3_2_dz3_pytorch_padded.txt', 16)
                z4_grad = save_layer_pad(small_cnn_model.z4.grad.detach().numpy()[0], 'logs/full_training_2/bw/pytorch/l4_2_dz3_pytorch_padded.txt', 16)
                z5_grad = save_layer_pad(small_cnn_model.z5.grad.detach().numpy()[0],'logs/full_training_2/bw/pytorch/l5_2_dz3_pytorch_padded.txt', 16)

                #### SAVING im2col MANUAL RESULTS ####

                ### convert X to im2col format
                w1_im2col = np.pad(w1_im2col, ((0, 0), (0, 4)), 'constant') #pad w1
                save_numpy('logs/full_training_2/fw/im2col/w1_im2col_padded.txt', w1_im2col.copy())
                x_torch_im2col = save_layer_im2col(x[0], 2, 'logs/full_training_2/fw/im2col/x_im2col_padded.txt', 16)
                x_torch_im2col = np.pad(x_torch_im2col, ((0, 4), (0, 0)), 'constant')

                #save_layer_im2col(small_cnn_model.z1.detach().numpy()[0], 2, 'logs/full_training_2/fw/im2col/l1_1_z1_im2col.txt', 16)
                #save_layer_im2col(small_cnn_model.a1.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l1_2_a1_im2col.txt', 16)
                #mp1_im2col = save_layer_im2col(small_cnn_model.mp1.detach().numpy()[0], 2,"logs/full_training_2/fw/im2col/l1_3_mp1_im2col.txt",16)

                #z2_im2col = save_layer_im2col(small_cnn_model.z2.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l2_1_z2_im2col.txt', 16)
                #a2_im2col = save_layer_im2col(small_cnn_model.a2.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l2_2_a2_im2col.txt', 16)
                #mp2_im2col = save_layer_im2col(small_cnn_model.mp2.detach().numpy()[0], 2, "logs/full_training_2/fw/im2col/l2_3_mp2_im2col.txt", 16)

                #z3_im2col = save_layer_im2col(small_cnn_model.z3.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l3_1_z3_im2col.txt', 16)
                #a3_im2col = save_layer_im2col(small_cnn_model.a3.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l3_2_a3_im2col.txt', 16)

                #z4_im2col = save_layer_im2col(small_cnn_model.z4.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l4_1_z4_im2col.txt', 16)
                #a4_im2col = save_layer_im2col(small_cnn_model.a4.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l4_2_a4_im2col.txt', 16)

                #z5_im2col = save_layer_im2col(small_cnn_model.z5.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l5_1_z5_im2col.txt', 16)
                #a5_im2col = save_layer_im2col(small_cnn_model.a5.detach().numpy()[0], 2,'logs/full_training_2/fw/im2col/l5_2_a5_im2col.txt', 16)

                np.savez("im2col_1dcnn_data.npz",
                         x=x_torch_im2col.astype(float).copy(),
                         y=y.astype(float).copy(),
                         w1=w1_im2col.astype(float).copy(),
                         w2=w2_im2col.copy().astype(float).copy(),
                         w3=w3_im2col.copy().astype(float).copy(),
                         w4=w4_im2col.copy().astype(float).copy(),
                         w5=w5_im2col.copy().astype(float).copy(),
                         b1=b1.astype(float).copy(),
                         b2=b2.astype(float).copy(),
                         b3=b3.astype(float).copy(),
                         b4=b4.astype(float).copy(),
                         b5=b5.astype(float).copy(),
                         fcw1=dense_w1.copy().astype(float).copy(),
                         fcw1_padded=dense_w1_reshaped_padded.copy().astype(float).copy(),
                         fcb1=dense_b1.copy().astype(float).copy(),
                         )


            _, predicted = torch.max(out_net.data, 1)
            y_argmax = np.argmax(y, axis=1)
            acc_array = np.asarray((predicted.detach().numpy() == y_argmax)*1)
            acc += acc_array.astype(float).sum()

            optimizer.step()

        accuracy = 100 * acc / num_of_samples
        print("Accuracy after %d number of epochs is %f" % (e+1, accuracy))


    #Evaluate on test data
    x_test_tensor = torch.from_numpy(X_test)
    pred = small_cnn_model(x_test_tensor.float())
    _, predicted = torch.max(pred.data, 1)
    y_argmax = np.argmax(y_test, axis=1)
    acc_array = np.asarray((predicted.detach().numpy() == y_argmax)*1)
    acc = acc_array.astype(float).sum()
    accuracy = 100 * acc / y_test.shape[0]
    print("Accuracy on test dataset: ", accuracy)


run()
