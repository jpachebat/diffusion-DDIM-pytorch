from dataset import create_dataset
from model.UNet import UNet
from model.mlp import ScoreNetwork
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, load_yaml
import torch
from utils.callbacks import ModelCheckpoint


def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    print(config)

    if config['Dataset']['dataset'] == 'gaussian':
        loader = create_dataset(**config)

    else:
        loader = create_dataset(**config["Dataset"])
    start_epoch = 1

    if config["Dataset"]['dataset'] == "gaussian":
        model = ScoreNetwork(**config['V_func'])
    else:
        model = UNet(**config["Model"]['unet']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])

    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch)
        model_checkpoint.step(loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)
