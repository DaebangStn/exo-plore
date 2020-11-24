from python.surrogate.preprocess import gt_path_from_exp_name, gt_from_exp_name
from python.surrogate.util import *
import polars as pl
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold


class GaitDataset(Dataset):
    def __init__(self, ds: Union[pl.LazyFrame, pl.DataFrame], in_lbl: List[str], out_lbl: List[str]):
        if isinstance(ds, pl.LazyFrame):
            df = ds.collect()
        else:
            df = ds
        inputs = df[in_lbl].to_numpy().astype('float32')
        targets = df[out_lbl].to_numpy().astype('float32')
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class MaskedTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        self.x = x
        self.y = y
        self.mask = mask

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]


class GaitData:
    def __init__(self, name: str, in_lbl: List[str], out_lbl: List[str], batch_size: int=16384, fold_idx: int=0,
                 num_folds: Optional[int]=None, transform: Union[str, dict]='mean_per_cycle', 
                 tensor_ds: bool=False):
        self._name = name
        self._num_folds = num_folds
        if num_folds is not None:
            assert 0 <= fold_idx < num_folds, f"Invalid fold idx({fold_idx}) for {num_folds} folds."
            print(f"[Data] Using fold {fold_idx + 1} of {num_folds}")
        self._fold_idx = fold_idx

        self._in_lbl = in_lbl
        self._out_lbl = out_lbl
        self._batch_size = batch_size
        self._tf = transform
        self._use_tensor_dataset = tensor_ds

        self._full_ds = None
        self._train_ds = None
        self._val_ds = None

        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        gt_from_exp_name(self._name, self._in_lbl, self._out_lbl, self._tf)
        self._processed_path = gt_path_from_exp_name(self._name, self._in_lbl, self._out_lbl, is_ckpt=False)
        
    def setup(self) -> None:
        lf = pl.scan_parquet(self._processed_path)
        df = lf.select(self._in_lbl + self._out_lbl).collect()
        df_size = df.shape[0]
        self._batch_size = min(self._batch_size, df_size + 1)
        
        if self._use_tensor_dataset:
            print('[TRAIN] Load data into TensorDataset')
            inputs = df[self._in_lbl].to_numpy().astype('float32')
            targets = df[self._out_lbl].to_numpy().astype('float32')
            self._input_tensor = torch.from_numpy(inputs).cuda()
            self._target_tensor = torch.from_numpy(targets).cuda()
            self._input_tensor, self._target_tensor, self._mask = self._pad_and_mask_batchsize(self._input_tensor, self._target_tensor)
            self._full_ds = MaskedTensorDataset(self._input_tensor, self._target_tensor, self._mask)            
        else:
            print('[TRAIN] Load data into GaitDataset')
            self._full_ds = GaitDataset(df, self._in_lbl, self._out_lbl)
        del df    

        if self._num_folds is None or self._num_folds == 1:
            self._train_ds = self._full_ds
            self._val_ds = self._full_ds
        else:
            kf = KFold(n_splits=self._num_folds, shuffle=True, random_state=42)
            indices = list(kf.split(self._full_ds))
            train_idx, val_idx = indices[self._fold_idx]
            self._train_ds = Subset(self._full_ds, train_idx)
            self._val_ds = Subset(self._full_ds, val_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size if not self._use_tensor_dataset else 1,
            shuffle=False,
            num_workers=0 if self._use_tensor_dataset else 4,
            pin_memory=True if not self._use_tensor_dataset else False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size if not self._use_tensor_dataset else 1,
            shuffle=False,
            num_workers=0 if self._use_tensor_dataset else 4,
            pin_memory=True if not self._use_tensor_dataset else False
        )

    def _pad_and_mask_batchsize(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x.shape[0]
        device = x.device
        remainder = N % self._batch_size
        pad_len = self._batch_size - remainder
        mask = torch.ones(N, dtype=torch.bool, device=device)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, *x.shape[1:], dtype=x.dtype, device=device)], dim=0)
            y = torch.cat([y, torch.zeros(pad_len, *y.shape[1:], dtype=y.dtype, device=device)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.bool, device=device)], dim=0)
        x = x.view(-1, self._batch_size, *x.shape[1:])
        y = y.view(-1, self._batch_size, *y.shape[1:])
        mask = mask.view(-1, self._batch_size)
        return x, y, mask

    @property
    def input_col(self) -> List[str]:
        return self._in_lbl

    @property
    def target_col(self) -> List[str]:
        return self._out_lbl

    @property
    def size(self) -> int:
        return len(self._full_ds)

    @property
    def input_tensor(self) -> torch.Tensor:
        return self._input_tensor

    @property
    def target_tensor(self) -> torch.Tensor:
        return self._target_tensor