import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam

import numpy as np
from tqdm import tqdm

from src.net import Net


class VAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net(512, 1024)
        # self.net = Net(8192, 8192)

    @property
    def device(self):
        return next(self.parameters()).device

    def ul_loss(self, x):
        with torch.no_grad():
            pred_poba = F.softmax(self.net(x), dim=1)

        r = 10 * torch.randn(x.shape).to(self.device)
        r.requires_grad_()

        noised_log_proba = F.log_softmax(self.net(x + r), dim=1)
        adv_distance = F.kl_div(noised_log_proba, pred_poba, reduction="batchmean")

        adv_grad = torch.autograd.grad(adv_distance, r)[0]
        r_vadv = F.normalize(adv_grad, dim=(1, 2, 3))
        self.net.zero_grad()

        log_proba = F.log_softmax(self.net(x + r_vadv), dim=1)
        loss = F.kl_div(log_proba, pred_poba, reduction="batchmean")
        return loss

    def __loss(self, imgs, labels):
        labeled = (labels != -1)

        ul_loss = self.ul_loss(imgs)
        l_loss = F.cross_entropy(self.net(imgs[labeled]), labels[labeled])
        return ul_loss + l_loss

    def fit(self, trainloader, testloader, epochs, lr):
        optim = Adam(self.parameters(), lr=lr)

        losses = []
        accuracies = [self._test(testloader)]

        for epoch in range(epochs):
            pbar = tqdm(trainloader, desc='Fitting..', postfix={'acc': 0.0})
            for imgs, labels in pbar:
                pbar.set_postfix({'epoch': epoch})
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                loss = self.__loss(imgs, labels)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.detach().cpu().numpy())

            acc = self._test(testloader)
            print(f'acc {acc}')
            pbar.update()
            accuracies.append(acc)

        return np.array(losses), np.array(accuracies)

    @torch.no_grad()
    def _test(self, testloader):
        self.eval()
        acc = 0
        for imgs, labels in testloader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds = torch.argmax(F.softmax(self.net(imgs), dim=1), dim=1)
            acc += (preds == labels).sum().item()

        self.train()
        return acc / len(testloader)
