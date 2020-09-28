import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset import NerDataset
from net import NerBertModel


def get_dataloader(data_path, preprocessor, batch_size):
    dataset = NerDataset(data_path, preprocessor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def main(args):
    preprocessor = Preprocessor(args.bert_model, args.max_len)
    train_dataloader = get_dataloader(
        args.train_data_path, preprocessor, args.train_batch_size
    )
    val_dataloader = get_dataloader(
        args.val_data_path, preprocessor, args.train_batch_size
    )
    test_dataloader = get_dataloader(
        args.test_data_path, preprocessor, args.eval_batch_size
    )

    bert_finetuner = NerBertModel(
        args, train_dataloader, val_dataloader, test_dataloader
    )

    logger = TensorBoardLogger(save_dir=args.log_path, version=1, name=args.task)

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/{epoch}_{val_acc:3f}",
        verbose=True,
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        prefix="",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode="max",
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        # distributed_backend="",
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
    )

    trainer.fit(bert_finetuner)
    trainer.test()


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)