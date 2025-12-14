# -*- coding = utf-8 -*-
# @Time: 2025/10/22 13:27
# @Author: Zhihang Yi
# @File: run.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
from model import HousePriceModel
from dataset import HousePriceDataset
import logging
import yaml
import matplotlib.pyplot as plt


def train():
    logger.info('loading hyperparameters and other configurations...')
    with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)
        config = file['config']

    hyperparameters = config['hyperparameters']
    other = config['other']
    batch_size = hyperparameters['batch_size']
    learning_rate = float(hyperparameters['learning_rate'])
    epochs = hyperparameters['epochs']
    training_path = other['training_path']
    test_path = other['test_path']
    device = torch.device(other['device'])
    logger.info('hyperparameters and other configurations loaded.')

    logger.info('loading dataset...')
    dataset = HousePriceDataset(training_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info('dataset loaded')

    logger.info('initializing model, criterion, and optimizer...')
    model = HousePriceModel().to(device)
    # model.load_state_dict(torch.load(other['trained_weights_path'], map_location=device))
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=float(hyperparameters['weight_decay']))
    logger.info('model, criterion, and optimizer initialized.')

    loss_history = []

    logger.info('starting training...')
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            loss = criterion(torch.log1p(pred), torch.log1p(y))
            loss = torch.sqrt(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(dataloader)
        loss_history.append(avg_loss)

        logger.info(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")

    logger.info('training completed.')

    logger.info('saving the trained weights...')
    torch.save(model.state_dict(), other['trained_weights_path'])
    logger.info('trained weights saved.')

    logger.info('plotting the training loss curve...')
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(other['loss_curve_path'])
    logger.info('training loss curve plotted and saved.')


def validation():
    logger.info('loading hyperparameters and other configurations...')
    with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)
        config = file['config']

    hyperparameters = config['hyperparameters']
    other = config['other']
    logger.info('hyperparameters and other configurations loaded.')

    logger.info('loading the trained model...')
    model = HousePriceModel()
    model.load_state_dict(torch.load(other['trained_weights_path'], map_location=other['device']))
    model.eval()
    logger.info('trained model loaded.')

    dataset = HousePriceDataset(other['training_path'])
    actual_prices = []
    predicted_prices = []

    for features, label in dataset:
        with torch.no_grad():
            prediction = model(features)
            logger.info(f'Predicted Price: {prediction.item():.2f}, Actual Price: {label.item():.2f}')

        actual_prices.append(label.item())
        predicted_prices.append(prediction.item())

    logger.info('plotting actual vs predicted prices...')
    plt.figure()
    plt.plot(range(len(actual_prices)), actual_prices, label='Actual Prices', color='blue')
    plt.plot(range(len(predicted_prices)), predicted_prices, label='Predicted Prices', color='red')
    plt.xlabel('Index')
    plt.ylabel('House Price')
    plt.legend()
    plt.savefig(other['comparison_path'])
    logger.info('actual vs predicted prices plotted and saved.')


def test():
    logger.info('loading hyperparameters and other configurations...')
    with open('config.yaml', 'r') as f:
        file = yaml.safe_load(f)
        config = file['config']

    hyperparameters = config['hyperparameters']
    other = config['other']
    batch_size = hyperparameters['batch_size']
    device = torch.device(other['device'])
    logger.info('hyperparameters and other configurations loaded.')

    logger.info('loading test dataset...')
    dataset = HousePriceDataset(other['test_path'], split='test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logger.info('test dataset loaded.')

    logger.info('loading the trained model...')
    model = HousePriceModel().to(device)
    model.load_state_dict(torch.load(other['trained_weights_path'], map_location=device))
    model.eval()
    logger.info('trained model loaded.')

    predictions = []

    logger.info('starting testing...')
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            pred = model(x)
            predictions.extend(pred.cpu().numpy().flatten().tolist())

    logger.info('testing completed.')

    ID = 1461

    logger.info('saving test predictions...')
    with open(other['test_predictions_path'], 'w') as f:
        f.write("Id,SalePrice\n")
        for price in predictions:
            f.write(f"{ID,price}\n")
            ID += 1
    logger.info('test predictions saved.')


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("train.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    train()
    validation()
    test()
