# -*- coding = utf-8 -*-
# @Time: 2025/10/21 20:11
# @Author: Zhihang Yi
# @File: model.py
# @Software: PyCharm

import torch
import logging

logger = logging.getLogger(__name__)

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dims, hidden_dims)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dims, input_dims)
        self.identity = torch.nn.Identity()

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x))) + self.identity(x)


class HousePriceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(74, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            ResidualBlock(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            ResidualBlock(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            ResidualBlock(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        pred = self.model(x)

        if torch.isnan(pred).any():
            logger.error("NaN values found in predictions")

        return pred  # (N,)
