from preprocessing.preprocess import FakeNewsDataset
from torch.utils.data import DataLoader


def build_data_pipeline(batch_size: int, num_workers: int, tokenizer, **kwargs) -> (DataLoader, DataLoader):
    train_dataset = FakeNewsDataset('train', tokenizer, **kwargs)
    test_dataset = FakeNewsDataset('test', tokenizer, **kwargs)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=num_workers,
                                   pin_memory=True if num_workers > 0 else False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=num_workers,
                                  pin_memory=True if num_workers > 0 else False)

    return train_data_loader, test_data_loader
