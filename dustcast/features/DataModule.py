import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dustcast.features.utils import generateDatasets, mkdir_if_not_exists

class SeviriDataModule(pl.LightningDataModule):
    def __init__(
        self,
        generate_datasets: bool = False,
        save_datasets: bool = False,
        load_datasets: bool = False,
        onetmstp_files = [],
        ds_train_file: str = "./train_ds.pt",
        ds_valid_file: str = "./valid_ds.pt",
        num_workers: int = 1,
        batch_size: int = 16,
        img_size: int = 64,
        frac_train: float = 0.8,
        len_block: int = 1000,
        normlims: tuple = False,
        load_into_memory: bool = False
    ):
        super().__init__()
        self.generate_datasets = generate_datasets
        self.save_datasets = save_datasets
        self.load_datasets = load_datasets
        self.onetmstp_files = onetmstp_files
        self.ds_train_file = ds_train_file
        self.ds_valid_file = ds_valid_file
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_size = img_size
        self.frac_train = frac_train
        self.len_block = len_block
        self.normlims = normlims
        self.load_into_memory = True if save_datasets else load_into_memory

    def prepare_data(self):
        if self.generate_datasets:
            print("Generating datasets")
            self.train_ds, self.valid_ds = generateDatasets(files=self.onetmstp_files,
                                                            frac_train=self.frac_train,
                                                            len_block=self.len_block,
                                                            img_size=self.img_size,
                                                            normlims=self.normlims,
                                                            load_into_memory=self.load_into_memory,
                                                            stack_chfr=True)
            if self.save_datasets:
                mkdir_if_not_exists(os.path.dirname(self.ds_train_file))
                torch.save(self.train_ds, self.ds_train_file)
                torch.save(self.valid_ds, self.ds_valid_file)
        elif self.load_datasets:
            print("Loading datasets")
            self.train_ds = torch.load(self.ds_train_file, weights_only=False)
            self.valid_ds = torch.load(self.ds_valid_file, weights_only=False)
        else:
            raise ValueError(
                "Either generate_datasets or load_datasets must be set to True."
            )

    def train_dataloader(self):
        print(f'num_workers: {self.num_workers}')
        return DataLoader(self.train_ds, pin_memory=True, persistent_workers=True, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, pin_memory=True, persistent_workers=True, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)
