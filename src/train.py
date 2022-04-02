import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pymysql
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

torch.cuda.empty_cache()

# NASDAQ 100 stocks
url = "https://fmpcloud.io/api/v3/nasdaq_constituent?apikey=<your_api_key>"
stock_list = pd.read_json(url)["symbol"].tolist()

SQL_CONNECTION_STR = "mysql+pymysql://root:@127.0.0.1:3306/us_stock"
MAX_HOLDING_PERIOD = 20
UPPER_BOUND = 0.1
LOWER_BOUND = 0.1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 384
NUM_EPOCHS = 1000
LOAD_MODEL = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOWN_SAMPLING_WINDOW_SIZES = [2,3,4,5,6,7,8]

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def get_triple_barrier_labels(seq):
    seq = seq.values[::-1]
    if any(seq >= seq[0]*(1+UPPER_BOUND)): return(2)
    if any(seq <= seq[0]*(1-LOWER_BOUND)): return(0)
    else: return(1)

class CustomDatasetTripleBarrier(Dataset):
    def __init__(self, symbol = None, start_date = None, end_date = None, max_len = None):
        super().__init__()
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.max_len = max_len

        conn = pymysql.connect(host = "127.0.0.1", user = "root", db = "us_stock")
        data_sql = "SELECT adj_close, date, open, high, low, close, volume FROM `daily_stock` WHERE symbol = '%s'"

        if start_date is not None:
            start_date = (datetime.strptime(start_date, "%Y-%m-%d") - pd.DateOffset(days = 512)).strftime("%Y-%m-%d")
            data_sql += f" and date >= '{start_date}'"
        data_sql += " ORDER BY date DESC"
        if max_len is not None:
            data_sql += f" LIMIT {max_len+512}"
        if symbol is None:
            symbol = [i[0] for i in conn.cursor().execute("select distinct symbol from `daily_stock`").fetchall()]

        self.data = {}
        self.label = {}
        self.max_sample_size = OrderedDict()
        for sym in tqdm(symbol, desc = "Loading data"):
            tem = pd.read_sql(data_sql%sym, conn)
            tem["date"] = tem["date"].map(str)
            tem = tem.set_index("date").sort_index(ascending=True)
            tem["label"] = tem[::-1]["adj_close"].rolling(MAX_HOLDING_PERIOD).apply(get_triple_barrier_labels)[::-1]
            if self.end_date is not None:
                tem = tem[:end_date]
            adj_factor = tem["adj_close"] / tem["close"]
            close_mean, close_std = tem["adj_close"].mean(), tem["adj_close"].std()
            for col in ["open", "high", "low", "close"]:
                tem[col] *= adj_factor
                tem[col] = (tem[col] - close_mean) / close_std
            tem["volume"] = tem["volume"].rolling(20).apply(lambda x: ((x-x.mean())/x.std()).iloc[-1])
            tem = tem.dropna()
            if tem.shape[0] < 512:
                continue
            stock_price = torch.from_numpy(tem[["open", "high", "low", "close", "volume"]].values).float()
            self.data[sym] = stock_price
            self.label[sym] = torch.from_numpy(tem["label"].values).long()
            self.max_sample_size[sym] = stock_price.shape[0] - 512 + 1
        self.symbol = list(self.data.keys())

    def __len__(self):
        return sum(self.max_sample_size.values())

    def __getitem__(self, index):
        prev_sym = None
        prev_count = 0
        for sym, size in self.max_sample_size.items():
            if prev_sym is None and size > index:
                out = self.data[sym][index:index+512, :], self.label[sym][index+512-1]
                return(out)
            elif prev_count + size > index and index >= prev_count:
                out = self.data[sym][index-prev_count:index-prev_count+512, :], self.label[sym][index-prev_count+512-1]
                return(out)
            else:
                prev_sym = sym
                prev_count += size
        raise Exception(f"Index {index} exceeding our limit {sum(self.max_sample_size.values())}")

train_dataset = CustomDatasetTripleBarrier(symbol = stock_list, start_date = "2009-01-01", end_date = "2018-12-31")
test_dataset = CustomDatasetTripleBarrier(symbol = stock_list, start_date = "2019-06-01", end_date = "2020-06-30")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=1)#, sampler=RandomSampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=1)#, sampler=RandomSampler)

class DownSampling(nn.Module):
    def __init__(self, window_sizes, step_sizes, method = "mean", device = DEVICE):
        super().__init__()
        if type(window_sizes) == int:
            window_sizes = [window_sizes]
        if type(step_sizes) == int:
            step_sizes = [step_sizes]
        assert len(step_sizes) == len(window_sizes), "window_sizes and step_sizes should have the same length"
        self.window_sizes = window_sizes
        self.step_sizes = step_sizes
        self.n_window = len(self.window_sizes)
        self.device = device

    def transform_one_sample(self, x):
        n = x.shape[0]
        out = []
        for idx in range(self.n_window):
            x = x.to(self.device)
            x_open, x_high, x_low, x_close, x_vol = x[0,:], x[1,:], x[2,:], x[3,:], x[4,:]
            x_open = x_open.unfold(0,self.window_sizes[idx],self.step_sizes[idx]).to(torch.float)[:,0]
            x_high = x_high.unfold(0,self.window_sizes[idx],self.step_sizes[idx]).to(torch.float).max(axis = 1)[0]
            x_low = x_low.unfold(0,self.window_sizes[idx],self.step_sizes[idx]).to(torch.float).min(axis = 1)[0]
            x_close = x_close.unfold(0,self.window_sizes[idx],self.step_sizes[idx]).to(torch.float)[:,-1]
            x_vol = x_vol.unfold(0,self.window_sizes[idx],self.step_sizes[idx]).to(torch.float).max(axis = 1)[0]
            out.append(torch.stack((x_open, x_high, x_low, x_close, x_vol)).permute((1,0)).to(self.device))
        return(out)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view((1,*x.shape))
        assert len(x.shape) == 3, "Input data must be 2 or 3 dimensional"
        batch_size = x.shape[0]
        out_dim = [i.shape[0] for i in self.transform_one_sample(x[0,:,:])]
        out = [torch.zeros((batch_size, 5, dim), device=self.device) for dim in out_dim]
        for idx in range(batch_size):
            result = self.transform_one_sample(x[idx,:])
            for i, r in enumerate(result):
                out[i][idx,:,:] = r.permute((1,0))
        out = [i.unsqueeze(dim=1) for i in out]
        return(out)


class DownSamplingConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            SamePaddingConv(1, 256, (5,3), padding='valid'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            SamePaddingConv(256, 512, 3, padding='valid'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
            SamePaddingConv(512, 512, 3, padding='valid'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            SamePaddingConv(512, 256, 3, padding='valid'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2),
        )

    def forward(self, x):
        return(self.model(x))

class AttentionBlock(nn.Module):
    def __init__(self, input_channels, num_heads, num_attention_layers):
        super().__init__()
        attention_layers = []
        for i in range(num_attention_layers):
            attention_layers.append(nn.MultiheadAttention(input_channels, num_heads, batch_first=True, dropout=0.2))
        self.attentions = nn.ModuleList(attention_layers)

    def forward(self, x):
        for attention in self.attentions:
            x = attention(x, x, x, need_weights=False)[0]
        return x

class SamePaddingConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if type(args[2]) == int:
            self.dim = 1
        else:
            self.dim = 2
        if self.dim == 1:
            total_pad_len = args[2] - 1
        else:
            total_pad_len = args[2][1] - 1
        left_pad_len = np.ceil(total_pad_len/2).astype(int)
        right_pad_len = np.floor(total_pad_len/2).astype(int)
        if self.dim == 1:
            self.pad = nn.ReflectionPad1d((left_pad_len, right_pad_len))
            self.Conv = nn.Conv1d(*args, **kwargs)
        else:
            self.pad = nn.ReflectionPad2d((left_pad_len, right_pad_len, 0, 0))
            self.Conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        if len(x.shape) == 4 and self.dim == 1:
            x = x.squeeze(dim=2)
        x = self.pad(x)
        x = self.Conv(x)
        return(x)


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_sampling = DownSampling(window_sizes = DOWN_SAMPLING_WINDOW_SIZES, step_sizes = DOWN_SAMPLING_WINDOW_SIZES)
        self.downsampling_block = DownSamplingConvBlock()

        self.batch_norm6 = nn.BatchNorm1d(1792)
        self.conv1 = SamePaddingConv(1792, 512, 5, padding='valid')
        self.conv2 = SamePaddingConv(512, 128, 5, padding='valid', bias = False)

        self.batch_norm5 = nn.BatchNorm1d(128)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.positional_embeddings = nn.Embedding(64, 128)
        self.attentionblock = AttentionBlock(128, 4, 3)
        self.batch_norm7 = nn.BatchNorm1d(64)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        batch_size = x.shape[0]

        x_downsampling = self.down_sampling(x)
        out = x_downsampling.copy()
        for idx, i in enumerate(x_downsampling):
            i = self.downsampling_block(i)
            i = F.upsample(i, size=(128))
            out[idx] = i
        x = torch.stack(out, dim = 1).to(DEVICE)
        x = x.view((batch_size, -1, 128))
        x = self.batch_norm6(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm5(x)
        x = self.max_pool1(x)
        x = x.permute(0,2,1)
        positions = torch.arange(0, x.shape[1]).expand(x.shape[0], x.shape[1]).to(DEVICE)
        x += self.positional_embeddings(positions)
        x = self.attentionblock(x)
        x = self.batch_norm7(x)
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return(x)


mcnn = nn.DataParallel(MCNN())
mcnn.to(device=DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mcnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=2e-5, verbose=False)

if LOAD_MODEL:
    load_checkpoint(torch.load("../runs11/MCNN_checkpoint_epoch41.pth.tar", map_location=DEVICE), mcnn, optimizer)
else:
    with torch.no_grad():
        mcnn(torch.rand((1,5,512)).to(DEVICE))
    initialize_weights(mcnn)

print(sum(p.numel() for p in mcnn.parameters() if p.requires_grad))

writer = SummaryWriter(f"../runs/14")

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure, ax = plt.subplots(1, 1, figsize = (6,6), tight_layout = True)
    cm_plot = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    ax.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")

    threshold = cm_plot.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_plot[i, j] > threshold else "black"
        ax.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return figure

def inference(model, test_loader, max_batch = 50, plot = False):
    accuracy = 0
    model.eval()
    cm = np.zeros((3,3))
    softmax = nn.Softmax(dim=1)
    
    for idx, (data, label) in enumerate(test_loader):
        data = data.permute((0,2,1)).to(DEVICE)
        label = label.to(DEVICE)
        with torch.no_grad():
            pred_label = model(data)
            pred_label = softmax(pred_label).argmax(dim=1)
            acc = label == pred_label
            accuracy += acc.sum()
            cm += confusion_matrix(label.cpu(), pred_label.cpu(), labels=[0,1,2])
        if idx >= max_batch:
            break
    model.train()
    
    if plot:
        figure = plot_confusion_matrix(cm, class_names=["0", "1", "2"])
        return accuracy/(idx+1)/BATCH_SIZE, figure
    else:
        return accuracy/(idx+1)/BATCH_SIZE


step = 0
mcnn.train()
softmax = nn.Softmax(dim=1)

for epoch in range(NUM_EPOCHS):
    losses = np.zeros((len(train_loader)))
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    
    counter = 0
    for batch_idx, (data, label) in loop:
        data = data.permute((0,2,1)).to(device=DEVICE)
        label = label.to(device=DEVICE)

        # forward
        pred_label = mcnn(data)
        loss = criterion(pred_label, label)
        losses[counter] = loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # update progress bar
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss = losses[:counter+1].mean())    

        # update step
        step += 1
        counter += 1

        # LR scheduler
        scheduler.step()

    #accuracy
    with torch.no_grad():
        if epoch % 1 == 0:
            train_accuracy, train_cm = inference(mcnn, train_loader, plot = True)
            test_accuracy, test_cm = inference(mcnn, test_loader, plot = True)
        else:
            train_accuracy = inference(mcnn, train_loader)
            test_accuracy = inference(mcnn, test_loader)

    # tensorboard
    writer.add_scalar('Training loss', sum(losses)/len(losses), global_step=epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, global_step=epoch)
    writer.add_scalar('Testing Accuracy', test_accuracy, global_step=epoch)
    if epoch % 1 == 0:
        writer.add_figure('Training confusion matrix', train_cm, global_step=epoch)
        writer.add_figure('Testing confusion matrix', test_cm, global_step=epoch)
    
    # save model
    checkpoint = {
        "state_dict": mcnn.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    if (epoch % 1 == 0) & (epoch != 0):
        save_checkpoint(checkpoint, filename=f"runs14_MCNN_checkpoint_epoch{epoch}.pth.tar")
