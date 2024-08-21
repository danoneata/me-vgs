import pdb

import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.datasets import make_circles

import torch
import torch.nn.functional as F

from torch import nn


torch.manual_seed(0)


def transform_geometrical(x):
    θ = np.pi / 8
    σ = 0.999
    ε = 0.0001
    τ = [0.0, 0.0]

    x2 = x.copy()
    x2 = x + np.random.normal(0, ε, x.shape)
    x2 = np.dot(x2, np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]]))
    x2 = σ * x2
    x2 = x2 + np.array(τ)
    x3 = x2.copy()
    x3[:, 0] = x2[:, 0] ** 2 - x2[:, 1] ** 2
    x3[:, 1] = 2 * x2[:, 0] * x2[:, 1]

    return x3


def transform_model(x):
    x2 = x.copy()
    model = get_model()
    x2 = torch.tensor(x2, dtype=torch.float32)
    x2 = model(x2).detach().numpy()
    x3 = x2.copy()
    return x3    


def transform_identity(x):
    return x


TRANSFORM_FUNCS = {
    "geometrical": transform_geometrical,
    "model": transform_model,
    "identity": transform_identity,
}


def generate_data(transform_func_type):
    n_samples = 500

    x1, Y1 = make_circles(
        n_samples=2 * n_samples,
        noise=0.1,
        factor=0.2,
        random_state=1,
    )
    x1 = x1[Y1 == 0]

    transform_func = TRANSFORM_FUNCS[transform_func_type]
    x2 = transform_func(x1)

    return x1, x2


def plot(x1, x2):
    fig, ax = plt.subplots()

    n_samples, n_dim = x1.shape
    assert n_dim == 2

    for i in range(0, n_samples, 10):
        ax.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], color="gray")

    ax.scatter(x1[:, 0], x1[:, 1])
    ax.scatter(x2[:, 0], x2[:, 1])

    return fig, ax


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True)


def stdnorm(x):
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)


def barlip_loss(z1, z2, λ=0.1):
    N, D = z1.size()
    # z1 = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
    # z2 = (z2 - z2.mean(dim=0)) / z2.std(dim=0)
    corr = torch.einsum("bi, bj -> ij", z1, z2) / N
    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= λ
    loss = cdif.sum()
    return loss


def clip_loss(z1, z2):
    N, D = z1.size()
    # z1 = l2norm(z1)
    # z2 = l2norm(z2)
    logits1 = z1 @ z2.t()
    logits2 = z2 @ z1.t()
    labels = torch.arange(N, device=logits1.device)
    loss1 = F.cross_entropy(logits1, labels)
    loss2 = F.cross_entropy(logits2, labels)
    return (loss1 + loss2) / 2


def get_model():
    return nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        # nn.Linear(10, 10),
        # nn.ReLU(),
        nn.Linear(10, 2),
    )


def accuracy(z1, z2):
    N, D = z1.size()
    dists = torch.cdist(z1, z2)
    labels = torch.arange(N, device=dists.device)
    acc = (dists.argmin(dim=1) == labels).float().mean().item()
    return 100 * acc


LOSSES = {
    "clip": clip_loss,
    "barlip": barlip_loss,
}

NORMS = {
    "clip": l2norm,
    "barlip": stdnorm,
}


def train(x1, x2, loss_type, to_run):
    loss_fn = LOSSES[loss_type]
    norm_fn = NORMS[loss_type]

    n_epochs = 2000

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)

    xn1 = norm_fn(x1)
    xn2 = norm_fn(x2)
    loss = loss_fn(xn1, xn2)

    acc = accuracy(xn1, xn2)
    title = f"Epoch:      0 · Loss: {loss.item():.3f} · Acc: {acc:.1f}%"
    fig, ax = plot(xn1.numpy(), xn2.numpy())
    ax.set_title(title)
    st.pyplot(fig)
    print(title)

    if not to_run:
        return

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        z1 = model(x1)
        z2 = x2

        zn1 = norm_fn(z1)
        zn2 = norm_fn(z2)
        loss = loss_fn(zn1, zn2)

        if epoch % 100 == 0:
            acc = accuracy(zn1, zn2)
            title = f"Epoch: {epoch:6d} · Loss: {loss.item():.3f} · Acc: {acc:.1f}%"
            fig, ax = plot(zn1.detach().numpy(), zn2.detach().numpy())
            ax.set_title(title)
            st.pyplot(fig)
            print(title)

        loss.backward()
        optimizer.step()



if __name__ == "__main__":
    with st.sidebar:
        transform_func_type = st.selectbox("Transform function", ["geometrical", "model", "identity"])
        loss_type = st.selectbox("Loss type", ["clip", "barlip"])
        to_run = st.button("Run")

    x1, x2 = generate_data(transform_func_type)
    fig, ax = plot(x1, x2)
    ax.set_title("Original data")
    st.pyplot(fig)
    st.markdown("---")
    train(x1, x2, loss_type, to_run)
