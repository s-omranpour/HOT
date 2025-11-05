import argparse

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.ts_data import TSDataModule
from src.models import HOTForTimeseriesForecasting

L.seed_everything(43)


def main():
    parser = argparse.ArgumentParser(
         "HOT-timeseries", add_help=False
    )
    parser.add_argument(
        "--name", default='weather', type=str, help="dataset name"
    )
    parser.add_argument(
        "--data_path", default='data/timeseries', type=str, help="dataset path"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout"
    )
    parser.add_argument(
        "--weight_decay", default=1e-2, type=float, help="weight decay"
    )
    parser.add_argument(
        "--d_hidden", default=128, type=int, help="hidden dimension"
    )
    parser.add_argument(
        "--d_mlp", default=512, type=int, help="MLP dimension"
    )
    parser.add_argument(
        "--num_blocks", default=4, type=int, help="number of blocks"
    )
    parser.add_argument(
        "--num_heads", default=8, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--patch_size", default=4, type=int, help="patch size"
    )
    parser.add_argument(
        "--attention_type", default='kronecker_product', type=str, help="attention type"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="max number of epochs"
    )
    args = parser.parse_args()

    datamod = TSDataModule(
        data_path=args.data_path,
        name=args.name,
        split_sizes=[0.7, 0.1, 0.2],
        context_length=96, 
        batch_size=128, 
        prediction_length=720, 
        normalize=True,
        num_workers=2
    )

    model = HOTForTimeseriesForecasting(
        d_hidden=args.d_hidden,
        d_mlp=args.d_mlp,
        n_blocks=args.num_blocks, 
        n_head=args.num_heads, 
        patch_size=args.patch_size,
        attention_type=args.attention_type,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    ## Trainer
    wandb_logger = WandbLogger(
        project='HOT-Timeseries',
    )
    early_stop_callback = EarlyStopping(
        monitor="val-avg-mae", min_delta=0.005, patience=10, verbose=False, mode="min"
    )
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        devices=1,
        accelerator="gpu", 
        num_nodes=1,
        logger=wandb_logger,
        callbacks=[early_stop_callback],
        accumulate_grad_batches=1,
        gradient_clip_val=1.,
        enable_progress_bar=False
    )
    trainer.fit(model, datamod.train_dataloader(), datamod.val_dataloader()) 
    trainer.test(model, datamod.test_dataloader())

if __name__ == '__main__':
    main()