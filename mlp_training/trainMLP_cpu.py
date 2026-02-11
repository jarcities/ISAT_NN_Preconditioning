# this script trains the MLP for the IP-ISAT and SEP-ISAT preconditioning

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from sklearn.cluster import KMeans
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Training settings
parser = argparse.ArgumentParser(description="Preconditioned ISAT")
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20000,
    metavar="N",  # 6000
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.8,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
)
########################## ADDED ##########################
parser.add_argument(
    "--bit", type=int, default=64, metavar="M", help="32 or 64 bit to train model on "
)
parser.add_argument(
    "--use-cpu",
    action="store_true",
    default=False,
    help="enables parallelized CPU training",
)
parser.add_argument(
    "--num-cpus",
    type=int,
    default=os.cpu_count(),
    metavar="N",
    help="number of CPUs to use for data loading",
)
###########################################################
parser.add_argument(
    "--use-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--use-mps", action="store_true", default=False, help="enables macOS GPU training"
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)
parser.add_argument(
    "--seed", type=int, default=4, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=True,
    help="For Saving the current Model",
)
args = parser.parse_args()

torch.manual_seed(args.seed)

########################## ADDED ##########################
# parallelize with CPU/GPU?
# USE_CPUS = args.use_cpu # parallelize on cpu
# USE_CUDA = args.use_cuda # nvidia
# USE_MPS = args.use_mps # apple
NUM_CPUS = args.num_cpus  # number of cpus to load data
if args.bit == 32:
    BIT = torch.float32
elif args.bit == 64:
    BIT = torch.float64

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(BIT)
device = torch.device("cpu")

N = 100  # number of neurons in the hidden layers
IP_ISAT = 1  # whether to perform IP-ISAT training
SEP_ISAT = 0  # whether to perform SEP-ISAT training


class Net(nn.Module):  # define the network, 6 MLP layers with N neurons each
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(54, N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, N)
        self.fc4 = nn.Linear(N, N)
        self.fc5 = nn.Linear(N, N)
        self.fc6 = nn.Linear(N, N)
        self.fc7 = nn.Linear(N, N)
        self.fc8 = nn.Linear(N, N)
        self.fc9 = nn.Linear(N, 54)

    def forward(self, x):

        x = self.fc1(x)
        x = F.mish(x)

        x = self.fc2(x)
        x = F.mish(x)

        x = self.fc3(x)
        x = F.mish(x)

        x = self.fc4(x)
        x = F.mish(x)

        x = self.fc5(x)
        x = F.mish(x)  # Mish is used as the activation function for all hidden layers

        x = self.fc6(x)
        x = F.mish(x)  # Mish is used as the activation function for all hidden layers

        x = self.fc7(x)
        x = F.mish(x)  # Mish is used as the activation function for all hidden layers

        x = self.fc8(x)
        x = F.mish(x)  # Mish is used as the activation function for all hidden layers

        x = self.fc9(x)

        return x


def my_loss_H(
    output, target, X, nClusters, nKmeans
):  # custom loss function for SEP-ISAT preconditioning

    output1 = output - target  # calculate the residual

    for ii in range(nClusters):

        output2 = output1[ii * nKmeans : (ii + 1) * nKmeans, :] - torch.mean(
            output1[ii * nKmeans : (ii + 1) * nKmeans, :], 0, True
        )
        X2 = X[ii * nKmeans : (ii + 1) * nKmeans, :] - torch.mean(
            X[ii * nKmeans : (ii + 1) * nKmeans, :], 0, True
        )

        grad = torch.linalg.lstsq(X2, output2 ,driver = 'gelsd').solution
        # grad = torch.linalg.lstsq(X2.cpu(), output2.cpu(), driver="gelsd").solution.to(
        #     X.device
        # )

        output1[ii * nKmeans : (ii + 1) * nKmeans, :] = output1[
            ii * nKmeans : (ii + 1) * nKmeans, :
        ] - (
            torch.matmul(X2, grad)
            + torch.mean(output1[ii * nKmeans : (ii + 1) * nKmeans, :], 0, True)
        )
    # for each K-Means cluster, subtract a least squares fit from the residual

    loss = torch.mean((output1) ** 2)  # overall loss
    return loss


def my_loss(output, target, X):  # standard RMS loss
    loss = torch.mean((output - target) ** 2)
    return loss


def train(
    args, model, device, train_loader, optimizer, epoch
):  # training for IP-ISAT networks
    model.train()
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = my_loss(output, Y, X)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.dry_run:
                break


def test(model, device, test_loader):  # testing for IP-ISAT networks
    model.eval()
    test_loss = 0
    nLoss = 0
    correct = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += my_loss(output, Y, X)  # sum up batch loss
            nLoss += 1

    test_loss /= nLoss

    print("\nTest set: Average loss: \n", test_loss)


def train2(
    args, model, device, train_loader, optimizer, epoch, nClusters, nKmeans
):  # training for SEP-ISAT networks
    model.train()
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = my_loss_H(output, Y, X, nClusters, nKmeans)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.dry_run:
                break


def test2(
    model, device, test_loader, nClusters, nKmeans
):  # testing for SEP-ISAT networks
    model.eval()
    test_loss = 0
    correct = 0
    nLoss = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += my_loss_H(output, Y, X, nClusters, nKmeans)
            nLoss += 1

    test_loss /= nLoss

    print("\nTest set: Average loss: \n", test_loss)


def PPPcluster(X, F, nClusters):
    # even cluster splitting
    stages = np.round(np.log2(nClusters)).astype(int)

    S = X.shape
    N = S[0]
    dim = S[1]

    ind = np.arange(N)

    clusters = np.zeros(N)

    for ii in range(stages):  # 0,1,2...
        nBatch = int(N / np.power(2, ii))
        for jj in range(np.power(2, ii)):  # 1,2,4...
            indLoc = ind[np.arange(nBatch) + jj * nBatch]
            Xloc = X[indLoc, :]
            Floc = F[indLoc, :]
            Xmean = np.mean(Xloc, axis=0)
            Fmean = np.mean(Floc, axis=0)

            steps = np.cumsum(
                np.sum(np.power(Xloc - Xmean, 2.0), axis=1)
                * np.sum(np.power(Floc - Fmean, 10.0), axis=1)
            )

            count = steps[-1]

            a = np.random.rand() * count

            I = np.min(np.where(a < steps))

            V = Xloc[I, :] - Xmean

            R = np.matmul(Xloc, V)

            order = np.argsort(R)

            ind[indLoc] = ind[indLoc[order]]

    clusterSize = np.round(N / nClusters).astype(int)

    for ii in range(nClusters):
        clusters[ind[np.arange(clusterSize) + ii * clusterSize]] = ii

    return clusters


def KmeansDataset(
    data1, n1, K1, K2, idim, device
):  # generate the training and testing datasets from numpy data
    # arranging the raw data via K-Means clustering on the inputs

    X = data1[:n1, :idim]  # training input (from data.csv)
    Y = data1[:n1, idim:]  # training output (from data.csv)

    # kmeans = KMeans(n_clusters=K1)
    # kmeans.fit(X) # perform K-Means clustering on the input
    # cl = kmeans.labels_

    cl = PPPcluster(X, Y, K1)

    print(np.count_nonzero(cl == 3))

    d = np.column_stack((cl, X, Y))

    d = d[d[:, 0].argsort()]  # arrange the training data by clusters

    X = d[:, 1 : idim + 1]
    Y = d[:, idim + 1 :]  # extract the cluster-arranged inputs and outputs

    X = torch.from_numpy(X).to(BIT)
    Y = torch.from_numpy(Y).to(BIT)  # convert to tensors

    X, Y = X.to(device), Y.to(device)  # send to device (currently, only CPU)

    X2 = data1[n1:, :idim]
    Y2 = data1[n1:, idim:]  # testing data

    # kmeans = KMeans(n_clusters=K2)
    # kmeans.fit(X2) # perform K-Means clustering on the input
    # cl = kmeans.labels_

    cl = PPPcluster(X2, Y2, K2)

    print(np.count_nonzero(cl == 3))

    d = np.column_stack((cl, X2, Y2))

    d = d[d[:, 0].argsort()]  # arrange the testing data by clusters

    X2 = d[:, 1 : idim + 1]
    Y2 = d[:, idim + 1 :]  # extract the cluster-arranged inputs and outputs

    X2 = torch.from_numpy(X2).to(BIT)
    Y2 = torch.from_numpy(Y2).to(BIT)  # convert to tensors

    X2, Y2 = X2.to(device), Y2.to(device)  # send to device (currently, only CPU)

    trainDataset = TensorDataset(X, Y)
    testDataset = TensorDataset(X2, Y2)  # generate the pytorch datasets

    return trainDataset, testDataset


# def main():
#     ########################## ADDED ##########################
#     if USE_CUDA and torch.cuda.is_available():
#         device = torch.device("cuda:0")
#         train_kwargs = {'batch_size': args.batch_size, 'num_workers': NUM_CPUS, 'pin_memory': True}
#         test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': NUM_CPUS, 'pin_memory': True}
#     elif USE_MPS and torch.backends.mps.is_available():
#         device = torch.device("mps")
#         train_kwargs = {'batch_size': args.batch_size, 'num_workers': 0}
#         test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 0}
#     else:
#         device = torch.device("cpu")
#         train_kwargs = {'batch_size': args.batch_size, 'num_workers': NUM_CPUS}
#         test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': NUM_CPUS}
#     ###########################################################

#     Nsam = 262144 # number of samples in data.csv

#     idim = 54 # dimension of the input/output

#     nKmeans = 128 # number of samples in each K-Means cluster

#     nClusters = int (args.batch_size / nKmeans) # number of overall K-Means clusters

#     K1 = int((2*Nsam/4)/nKmeans) # number of training K-Means clusters
#     K2 = int((2*Nsam/4)/nKmeans) # number of testing K-Means clusters

#     n1 = int(2*Nsam/4)

#     data1 = np.genfromtxt('data.csv',delimiter=',') # get data from data.csv

#     trainDataset, testDataset = KmeansDataset(data1,n1,K1,K2,idim,device) # format into PyTorch datasets

#     train_loader = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=True)
#     test_loader = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=True)
# 	# loaders for IP-ISAT training (shuffling enabled)

#     train_loader2 = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=False)
#     test_loader2 = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=False)
# 	# loaders for SEP-ISAT training (shuffling disabled)

#     ########################## ADDED ##########################
#     ## https://docs.pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html ##
#     if USE_CUDA and torch.cuda.is_available():
#         if torch.cuda.device_count() > 1:
#             print("USING -> ", torch.cuda.device_count(), " GPUS")
#             model = nn.DataParallel(Net())
#         else:
#             model = Net()
#         model = model.to(device)
#     else:
#         model = Net().to(device)
#     ###########################################################
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     optimizer2 = optim.Adam(model.parameters(), lr=args.lr)

#     start_time = time.time()

#     scheduler = StepLR(optimizer, step_size=1000, gamma=args.gamma)
#     for epoch in range(1, IP_ISAT*args.epochs + 1):  # IP-ISAT training
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)
#         scheduler.step()

#     shuffleEpoch = 500

#     trainDataset, testDataset = KmeansDataset(data1,n1,K1,K2,idim,device)
#     train_loader2 = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=False)
#     test_loader2 = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=False)

#     scheduler2 = StepLR(optimizer, step_size=1, gamma=0.99975)

#     for epoch in range(1, SEP_ISAT*args.epochs + 1): # SEP-ISAT training

#         print(epoch)

#         if (np.mod(epoch,shuffleEpoch)==1): # rearrange training/testing data into new K-Means clusters every
#             # "shuffleEpoch" training epochs

#             trainDataset, testDataset = KmeansDataset(data1,n1,K1,K2,idim,device)
#             train_loader2 = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=False)
#             test_loader2 = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=False)
#             scheduler2.step()

#         train2(args, model, device, train_loader2, optimizer2, epoch, nClusters, nKmeans)

#         if (np.mod(epoch,10)==1):
#             test2(model, device, test_loader2, nClusters, nKmeans)

#     end_time = time.time()
#     print(f"Elapsed time: {end_time - start_time} seconds")

#     ########################## ADDED ##########################
#     m = model.module if isinstance(model, nn.DataParallel) else model
#     ###########################################################

#     dd = m.fc1.weight.detach().cpu().numpy()
#     dd.tofile('fc1w.csv', sep = ',')

#     dd = m.fc2.weight.detach().cpu().numpy()
#     dd.tofile('fc2w.csv', sep = ',')

#     dd = m.fc3.weight.detach().cpu().numpy()
#     dd.tofile('fc3w.csv', sep = ',')

#     dd = m.fc4.weight.detach().cpu().numpy()
#     dd.tofile('fc4w.csv', sep = ',')

#     dd = m.fc5.weight.detach().cpu().numpy()
#     dd.tofile('fc5w.csv', sep = ',')

#     dd = m.fc6.weight.detach().cpu().numpy()
#     dd.tofile('fc6w.csv', sep = ',')

#     dd = m.fc7.weight.detach().cpu().numpy()
#     dd.tofile('fc7w.csv', sep = ',')

#     dd = m.fc8.weight.detach().cpu().numpy()
#     dd.tofile('fc8w.csv', sep = ',')

#     dd = m.fc9.weight.detach().cpu().numpy()
#     dd.tofile('fc9w.csv', sep = ',')

#     dd = m.fc1.bias.detach().cpu().numpy()
#     dd.tofile('fc1b.csv', sep = ',')

#     dd = m.fc2.bias.detach().cpu().numpy()
#     dd.tofile('fc2b.csv', sep = ',')

#     dd = m.fc3.bias.detach().cpu().numpy()
#     dd.tofile('fc3b.csv', sep = ',')

#     dd = m.fc4.bias.detach().cpu().numpy()
#     dd.tofile('fc4b.csv', sep = ',')

#     dd = m.fc5.bias.detach().cpu().numpy()
#     dd.tofile('fc5b.csv', sep = ',')

#     dd = m.fc6.bias.detach().cpu().numpy()
#     dd.tofile('fc6b.csv', sep = ',') # output the network weights into ASCII format

#     dd = m.fc7.bias.detach().cpu().numpy()
#     dd.tofile('fc7b.csv', sep = ',') # output the network weights into ASCII format

#     dd = m.fc8.bias.detach().cpu().numpy()
#     dd.tofile('fc8b.csv', sep = ',') # output the network weights into ASCII format

#     dd = m.fc9.bias.detach().cpu().numpy()
#     dd.tofile('fc9b.csv', sep = ',') # output the network weights into ASCII format

# if __name__ == '__main__':
#     main()


def runTraining(rank, world_size, args, data1, n1, K1, K2, idim):
    ## https://www.codegenes.net/blog/dataparallel-pytorch-cpu/#google_vignette ##
    print(f"Running DDP on rank {rank}.")

    # setup distributed env
    ## https://medium.com/@nishantbhansali80/data-parallel-with-pytorch-on-cpus-3e89312db6c0 ##
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = torch.device("cpu")

    # calculate effective batch size per process
    effective_batch_size = args.batch_size // world_size
    train_kwargs = {"batch_size": effective_batch_size, "num_workers": 0}
    test_kwargs = {"batch_size": args.test_batch_size // world_size, "num_workers": 0}

    # load data
    ## https://docs.pytorch.org/docs/stable/data.html ##
    trainDataset, testDataset = KmeansDataset(data1, n1, K1, K2, idim, device)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainDataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testDataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        trainDataset, sampler=train_sampler, **train_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        testDataset, sampler=test_sampler, **test_kwargs
    )

    train_sampler2 = torch.utils.data.distributed.DistributedSampler(
        trainDataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader2 = torch.utils.data.DataLoader(
        trainDataset, sampler=train_sampler2, **train_kwargs
    )
    test_loader2 = torch.utils.data.DataLoader(
        testDataset, sampler=test_sampler, **test_kwargs
    )

    # create model and wrap with DDP
    model = Net().to(device)
    ddp_model = DDP(model)

    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(ddp_model.parameters(), lr=args.lr)

    # IP-ISAT training
    scheduler = StepLR(optimizer, step_size=1000, gamma=args.gamma)
    for epoch in range(1, IP_ISAT * args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train(args, ddp_model, device, train_loader, optimizer, epoch)
        if rank == 0:
            test(ddp_model, device, test_loader)
        scheduler.step()

    # SEP-ISAT training
    nClusters = int(args.batch_size / 128)
    nKmeans = 128
    shuffleEpoch = 500

    trainDataset, testDataset = KmeansDataset(data1, n1, K1, K2, idim, device)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(
        trainDataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader2 = torch.utils.data.DataLoader(
        trainDataset, sampler=train_sampler2, **train_kwargs
    )
    test_sampler2 = torch.utils.data.distributed.DistributedSampler(
        testDataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_loader2 = torch.utils.data.DataLoader(
        testDataset, sampler=test_sampler2, **test_kwargs
    )

    scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.99975)

    for epoch in range(1, SEP_ISAT * args.epochs + 1):
        if rank == 0:
            print(epoch)

        if np.mod(epoch, shuffleEpoch) == 1:
            trainDataset, testDataset = KmeansDataset(data1, n1, K1, K2, idim, device)
            train_sampler2 = torch.utils.data.distributed.DistributedSampler(
                trainDataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            train_loader2 = torch.utils.data.DataLoader(
                trainDataset, sampler=train_sampler2, **train_kwargs
            )
            test_sampler2 = torch.utils.data.distributed.DistributedSampler(
                testDataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            test_loader2 = torch.utils.data.DataLoader(
                testDataset, sampler=test_sampler2, **test_kwargs
            )
            scheduler2.step()

        train2(
            args,
            ddp_model,
            device,
            train_loader2,
            optimizer2,
            epoch,
            nClusters,
            nKmeans,
        )

        if (np.mod(epoch, 10) == 1) and rank == 0:
            test2(ddp_model, device, test_loader2, nClusters, nKmeans)

    if rank == 0:

        # save model weights
        m = ddp_model.module
        dd = m.fc1.weight.detach().cpu().numpy()
        dd.tofile("fc1w.csv", sep=",")

        dd = m.fc2.weight.detach().cpu().numpy()
        dd.tofile("fc2w.csv", sep=",")

        dd = m.fc3.weight.detach().cpu().numpy()
        dd.tofile("fc3w.csv", sep=",")

        dd = m.fc4.weight.detach().cpu().numpy()
        dd.tofile("fc4w.csv", sep=",")

        dd = m.fc5.weight.detach().cpu().numpy()
        dd.tofile("fc5w.csv", sep=",")

        dd = m.fc6.weight.detach().cpu().numpy()
        dd.tofile("fc6w.csv", sep=",")

        dd = m.fc7.weight.detach().cpu().numpy()
        dd.tofile("fc7w.csv", sep=",")

        dd = m.fc8.weight.detach().cpu().numpy()
        dd.tofile("fc8w.csv", sep=",")

        dd = m.fc9.weight.detach().cpu().numpy()
        dd.tofile("fc9w.csv", sep=",")

        dd = m.fc1.bias.detach().cpu().numpy()
        dd.tofile("fc1b.csv", sep=",")

        dd = m.fc2.bias.detach().cpu().numpy()
        dd.tofile("fc2b.csv", sep=",")

        dd = m.fc3.bias.detach().cpu().numpy()
        dd.tofile("fc3b.csv", sep=",")

        dd = m.fc4.bias.detach().cpu().numpy()
        dd.tofile("fc4b.csv", sep=",")

        dd = m.fc5.bias.detach().cpu().numpy()
        dd.tofile("fc5b.csv", sep=",")

        dd = m.fc6.bias.detach().cpu().numpy()
        dd.tofile("fc6b.csv", sep=",")

        dd = m.fc7.bias.detach().cpu().numpy()
        dd.tofile("fc7b.csv", sep=",")

        dd = m.fc8.bias.detach().cpu().numpy()
        dd.tofile("fc8b.csv", sep=",")

        dd = m.fc9.bias.detach().cpu().numpy()
        dd.tofile("fc9b.csv", sep=",")

    dist.destroy_process_group()


def main():

    start_time = time.time()

    Nsam = 262144  # number of samples in data.csv
    idim = 54  # dimension of the input/output
    nKmeans = 128  # number of samples in each K-Means cluster

    K1 = int((2 * Nsam / 4) / nKmeans)  # number of training K-Means clusters
    K2 = int((2 * Nsam / 4) / nKmeans)  # number of testing K-Means clusters
    n1 = int(2 * Nsam / 4)

    data1 = np.genfromtxt("data.csv", delimiter=",")  # get data from data.csv

    # compute world size for data distro
    world_size = max(NUM_CPUS, os.cpu_count())
    print(f"\n# of processes -> {world_size}\n")

    # parallelize training
    mp.spawn(
        runTraining,
        args=(world_size, args, data1, n1, K1, K2, idim),
        nprocs=world_size,
        join=True,
    )

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
