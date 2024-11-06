import streamlit as st
st.run_on_save()
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import stqdm
plt.rcParams['axes.facecolor'] = '#0b0e12' 
plt.style.use('dark_background')


st.title('Devil\'s Toy')

def generate_data(num_per_class = 300, classes = 3, dim = 2):

    X = np.zeros((num_per_class * classes, dim))
    y = np.zeros(num_per_class * classes, dtype='uint8')

    for j in range(classes):
        ix = range(num_per_class * j,num_per_class * (j + 1))
        r = np.linspace(0.0, 1, num_per_class)
        t = np.linspace(j * 4, (j + 1) * 4,num_per_class) + np.random.randn(num_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y


st.subheader("Generate data")
n_dots = st.number_input("Number of dots per class", min_value=9, value=300)
cls = st.slider("Number of classes?", 1, 5, 3)

X, y = generate_data(num_per_class=n_dots, classes=cls)


def plot_devil_toys(X, y):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Отображаем точки
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma')
    ax.tick_params(axis='both', which='major', labelsize=12, color='white')

    # Устанавливаем заголовок
    ax.set_title('Devil\'s Toys', fontsize=15)
    
    return fig

fig = plot_devil_toys(X, y)

# Отображаем график в Streamlit
st.pyplot(fig)


# Data splitting
st.subheader("Data Split")
test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, test_size=test_size)

if type(X_train) != torch.Tensor:

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    y_train = torch.FloatTensor(y_train).view(-1, 1)
    np_y_test = y_test.copy()
    y_test = torch.FloatTensor(y_test).view(-1, 1)


st.subheader("Data loading")
bsize = st.number_input("Batch size", min_value=1, max_value=100, value=70, step=10)
train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=bsize, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=bsize, shuffle=False)


class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X = self.layers(X)
        return X


st.subheader("Model hyperparameters")

DevilNet = Net(cls)

lr = st.slider("Learning rate (0.001 - 0.1)", 0.001, 0.1, 0.01, 0.001)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(DevilNet.parameters(), lr=lr)

num_epoch = st.slider("Number of epochs", 1, 1000, 100, 1)
def train(model, loader, loss, optimizer, num_epoch):
    total_loss = []

    for epoch in stqdm.stqdm(range(num_epoch)):
        epoch_loss = []

        for X, y in loader:
            y = y.reshape(loader.batch_size).long()
            y_pred = model(X)

            loss_value = loss(y_pred, y)
            epoch_loss.append(loss_value.item())

            # Вычисление производных весов
            loss_value.backward()

            # Шаг изменения весов
            optimizer.step()

            # Обнуление производных, сохраненных в оптимизаторе
            optimizer.zero_grad()

        total_loss.append(np.mean(epoch_loss))

    st.success('Training completed!')
    return total_loss



def train_loss_plot(total_loss):
    plt.figure(figsize=(10, 8))
    plt.title('Training Loss Over Epochs', fontsize=15)
    plt.plot(range(1, len(total_loss) + 1), total_loss, 'r')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.tight_layout()
    return plt


st.subheader("Training Progress")
loss_plot = train_loss_plot(train(DevilNet, train_loader, criterion, optimizer, num_epoch))
st.pyplot(loss_plot)

with torch.no_grad():   
    test_pred = DevilNet(X_test)
fn_test_pred = test_pred.numpy().argmax(axis=1)
fn_y = y_test.reshape(y_test.shape[0]).long()

st.subheader("Test Results")
fn_loss = criterion(test_pred, fn_y).item()
st.markdown(f'Loss:  {fn_loss}')


test_plot = plot_devil_toys(X_test, fn_test_pred)
st.pyplot(test_plot)