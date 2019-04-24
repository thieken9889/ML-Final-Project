import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data_wrangler import data_wrangler
import time
import matplotlib.pyplot as plt


print("torch.cuda.is_available()   =", torch.cuda.is_available())
print("torch.cuda.device_count()   =", torch.cuda.device_count())
print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
print("torch.cuda.current_device() =", torch.cuda.current_device())

# If your laptop has a cuda device then use it, otherwise, just do everything on the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# load the data
datadir = 'chest_xray'
train_dl, test_dl = data_wrangler(datadir, 64, batch_size=100)

# load pre-trained AlexNet
model = torchvision.models.resnet152(pretrained=True)
# print(str(model))

# freeze the gradients for the pre-trained network
for param in model.parameters():
    param.requires_grad = False

# rearrange the classifier
model.fc = nn.Sequential(nn.Dropout(p=0.5),
                                 nn.Linear(in_features=2048, out_features=256, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=256, out_features=1, bias=True),
                                 nn.Sigmoid())
print(str(model))

# # print the requires_grad parameter to make sure we are training the last few layers
# for param in model.parameters():
#     print(param.requires_grad)

# send the model to the device we are using
model.to(device)

# only need binary cross-entropy loss because we only have 2 class
criterion = nn.BCELoss()

# choose ADAM because its the best?
lr = 1e-3
opt = optim.Adam(model.parameters(), lr=lr)


# train and test the model over several epochs
num_epoch = 5

a_tr_loss = np.zeros([num_epoch])
a_tr_accuracy = np.zeros([num_epoch])
a_ts_loss = np.zeros([num_epoch])
a_ts_accuracy = np.zeros([num_epoch])

print_intvl = 1

for epoch in range(num_epoch):

    model.train()  # put model in training mode
    correct = 0  # initialize error counter
    total = 0  # initialize total counter
    batch_loss = []

    # save the start time of epoch (in seconds)
    t0 = time.time()

    # iterate over training set
    for train_iter, data in enumerate(train_dl):
        # get the batches from the dataloader data
        x_batch, y_batch = data
        y_batch = y_batch.type(torch.float)
        y_batch = y_batch.view(y_batch.size(0), 1)

        # send the data to the device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        out = model(x_batch)
        # print(out)
        # print(y_batch)

        # Compute Loss
        loss = criterion(out, y_batch)
        batch_loss.append(loss.item())

        # print(loss)

        # Compute gradients using back propagation
        opt.zero_grad()
        loss.backward()

        # Take an optimization 'step'
        opt.step()

        # Do hard classification
        predicted = out.round()
        # print(predicted)

        # Compute number of decision errors
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        # print(total)
        # print(correct)

    tr_accuracy = 100 * correct / total  # Compute accuracy over epoch
    a_tr_loss[epoch] = np.mean(batch_loss)  # Compute average loss over epoch
    a_tr_accuracy[epoch] = 100 * correct / total

    model.eval()  # put model in evaluation mode
    correct = 0  # initialize error counter
    total = 0  # initialize total counter
    batch_loss = []

    with torch.no_grad():
        for data in test_dl:
            xts_batch, yts_batch = data
            yts_batch = yts_batch.type(torch.float)
            #print(yts_batch)
            yts_batch = yts_batch.view(yts_batch.size(0), 1)

            xts_batch, yts_batch = xts_batch.to(device), yts_batch.to(device)

            outputs = model(xts_batch)
            batch_loss.append(criterion(outputs, yts_batch).item())

            predicted = outputs.round()

            total += yts_batch.size(0)
            correct += (predicted == yts_batch).sum().item()

    a_ts_loss[epoch] = np.mean(batch_loss)
    a_ts_accuracy[epoch] = 100 * correct / total

    # get end time and calculate duration in minutes
    t1 = time.time()
    epoch_time = (t1-t0)/60

    # Print details every print_mod epoch
    print('Epoch: {0:2d}   Train Loss: {1:.3f}   '.format(epoch + 1, a_tr_loss[epoch])
          + 'Train Accuracy: {0:.2f}    Test Loss: {1:.3f}   '.format(a_tr_accuracy[epoch], a_ts_loss[epoch])
          + 'Test Accuracy: {0:.2f}    '.format(a_ts_accuracy[epoch])
          + 'Duration: {0:.2f}'.format(epoch_time))

plt.plot(a_tr_accuracy)
plt.plot(a_ts_accuracy)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['training accuracy', 'test accuracy'])
plt.show()