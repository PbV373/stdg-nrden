import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import controldiffeq
import torch.nn.functional as F
from typing import Optional


_LOGSIG_BACKEND: Optional[str] = None


def _get_logsig_backend():
    """
    Returns:
        backend: "signatory" or "iisignature"
        logsig_channels_fn: (in_channels:int, depth:int) -> int
        logsig_stream_fn: (paths_bn: torch.Tensor [B*N,T,C]) -> torch.Tensor [B*N,T,C_logsig]
    """
    global _LOGSIG_BACKEND
    if _LOGSIG_BACKEND == "signatory":
        import signatory  # type: ignore

        def channels_fn(in_channels: int, depth: int) -> int:
            return int(signatory.logsignature_channels(in_channels, depth))

        def stream_fn(paths_bn: torch.Tensor, depth: int) -> torch.Tensor:
            return signatory.logsignature(paths_bn, depth=depth, stream=True)

        return "signatory", channels_fn, stream_fn

    if _LOGSIG_BACKEND == "iisignature":
        import iisignature  # type: ignore

        def channels_fn(in_channels: int, depth: int) -> int:
            return int(iisignature.logsiglength(in_channels, depth))

        def stream_fn(paths_bn: torch.Tensor, depth: int) -> torch.Tensor:
            # iisignature has no stream=True; build logsignatures on prefixes (ok for moderate lag).
            paths_np = paths_bn.detach().cpu().numpy()  # [B*N, T, C]
            bn, T, C_in = paths_np.shape

            # iisignature needs a prepared object; API differs by version
            prepared = None
            try:
                prepared = iisignature.prepare(C_in, depth, "logsig")
            except Exception:
                try:
                    prepared = iisignature.prepare(C_in, depth)
                except Exception as e:
                    raise RuntimeError(
                        f"iisignature.prepare failed. C_in={C_in}, depth={depth}. "
                        "Verify iisignature is installed correctly."
                    ) from e

            outs = []
            for t in range(T):
                prefix = paths_np[:, : t + 1, :]
                out_t = iisignature.logsig(prefix, prepared)  # [B*N, C_logsig]
                outs.append(torch.from_numpy(out_t).float())
            return torch.stack(outs, dim=1)  # [B*N, T, C_logsig]

        return "iisignature", channels_fn, stream_fn

    # auto-detect
    try:
        import signatory  # type: ignore  # noqa: F401
        _LOGSIG_BACKEND = "signatory"
        return _get_logsig_backend()
    except Exception:
        _LOGSIG_BACKEND = "iisignature"
        return _get_logsig_backend()


def _dense_path_from_spline(path_bn: torch.Tensor, T: int, substeps: int) -> torch.Tensor:
    """
    Natural cubic spline, then dense samples on [0, T-1].
    path_bn: [B*N, T, C_in] at discrete times 0..T-1.
    Returns path_dense: [B*N, T_dense, C_in]
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    times = torch.linspace(0, T - 1, T, dtype=path_bn.dtype, device=path_bn.device)
    coeffs = controldiffeq.natural_cubic_spline_coeffs(times, path_bn)
    spline = controldiffeq.NaturalCubicSpline(times, coeffs)
    T_dense = (T - 1) * substeps + 1
    times_dense = torch.linspace(0, T - 1, T_dense, dtype=path_bn.dtype, device=path_bn.device)
    pieces = []
    for ti in times_dense:
        pieces.append(spline.evaluate(ti))
    return torch.stack(pieces, dim=1)


def _build_augmented_path(x_np: np.ndarray) -> torch.Tensor:
    """
    Build augmented raw path [time, features] with normalized time in [0, 1].
    Input: x_np [B, T, N, D_raw]
    Output: torch.Tensor [B, T, N, 1 + D_raw]
    """
    B, T, N, D_raw = x_np.shape
    x = torch.from_numpy(x_np).float()  # CPU
    times = torch.linspace(0.0, 1.0, T, dtype=x.dtype, device=x.device)
    time_ch = times.view(1, T, 1, 1).expand(B, T, N, 1)
    return torch.cat([time_ch, x], dim=-1)


def _logsig_stream_encode(x_np, depth, spline_substeps: int = 4):
    """
    Pipeline: discrete observations -> natural cubic spline X(t) -> dense path
    -> LogSignature stream -> align to length T.

    Args:
        x_np: numpy array, shape [B, T, N, D_raw]
        depth: LogSignature truncation depth
        spline_substeps: segments between knot intervals (>=1); 1 keeps knots only.

    Returns:
        torch.FloatTensor on CPU, shape [B, T, N, C_logsig]
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not isinstance(x_np, np.ndarray):
        x_np = np.asarray(x_np)
    if x_np.ndim != 4:
        raise ValueError(f"Expected x_np with shape [B,T,N,D], got {x_np.shape}")

    B, T, N, D_raw = x_np.shape
    path = _build_augmented_path(x_np)  # [B, T, N, 1 + D_raw]

    # Merge batch and node dims; spline then LogSig
    path_bn = path.permute(0, 2, 1, 3).reshape(B * N, T, 1 + D_raw)  # [B*N, T, C_in]
    backend, _, stream_fn = _get_logsig_backend()

    substeps = max(1, int(spline_substeps))
    if backend == "iisignature" and substeps > 2:
        substeps = 2
    path_for_logsig = _dense_path_from_spline(path_bn, T, substeps)  # [B*N, T_dense, C_in]
    T_dense = path_for_logsig.size(1)

    logsig_bn = stream_fn(path_for_logsig, depth=depth)

    # signatory(stream=True) often length T_dense-1
    if logsig_bn.dim() == 3 and logsig_bn.size(1) == T_dense - 1:
        pad = torch.zeros(logsig_bn.size(0), 1, logsig_bn.size(2), dtype=logsig_bn.dtype, device=logsig_bn.device)
        logsig_bn = torch.cat([pad, logsig_bn], dim=1)

    # Resize to lag=T to match CDE time grid
    if logsig_bn.size(1) != T:
        logsig_bn = F.interpolate(
            logsig_bn.transpose(1, 2),
            size=T,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    C_logsig = logsig_bn.size(-1)
    logsig = logsig_bn.reshape(B, N, T, C_logsig).permute(0, 2, 1, 3).contiguous()  # [B, T, N, C_logsig]
    return logsig


def _standardize_by_train(train_x: torch.Tensor, val_x: torch.Tensor, test_x: torch.Tensor):
    """
    Channel-wise z-score using train split statistics.
    Tensors are [B, T, N, C] on CPU.
    """
    mean = train_x.mean(dim=(0, 1, 2), keepdim=True)
    std = train_x.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def data_loader_cde(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    data = torch.utils.data.TensorDataset(*X, torch.tensor(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    actual_nodes = data.shape[1]
    if hasattr(args, 'num_nodes') and args.num_nodes != actual_nodes:
        print(f"Warning: config num_nodes ({args.num_nodes}) != data ({actual_nodes}); using data.")
    args.num_nodes = actual_nodes
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    assert x_tra.shape[2] == args.num_nodes
    assert x_val.shape[2] == args.num_nodes
    assert x_test.shape[2] == args.num_nodes
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_dataloader_cde(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    data = load_st_dataset(args.dataset)
    actual_nodes = data.shape[1]
    if hasattr(args, 'num_nodes') and args.num_nodes != actual_nodes:
        print(f"Warning: config num_nodes ({args.num_nodes}) != actual nodes ({actual_nodes})")
        print(f"Using actual node count: {actual_nodes}")
    args.num_nodes = actual_nodes
    print(f"num_nodes set to: {args.num_nodes}")

    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    assert x_tra.shape[2] == args.num_nodes
    assert x_val.shape[2] == args.num_nodes
    assert x_test.shape[2] == args.num_nodes
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    # TODO: make argument for missing data
    if args.missing_test == True:
        generator = torch.Generator().manual_seed(56789)
        xs = np.concatenate([x_tra, x_val, x_test])
        for xi in xs:
            removed_points_seq = torch.randperm(xs.shape[1], generator=generator)[:int(xs.shape[1] * args.missing_rate)].sort().values
            removed_points_node = torch.randperm(xs.shape[2], generator=generator)[:int(xs.shape[2] * args.missing_rate)].sort().values

            for seq in removed_points_seq:
                for node in removed_points_node:
                    xi[seq,node] = float('nan')
        x_tra = xs[:x_tra.shape[0],...]
        x_val = xs[x_tra.shape[0]:x_tra.shape[0]+x_val.shape[0],...]
        x_test = xs[-x_test.shape[0]:,...]
    # ==== LogSignature encode (spline then LogSig) ====
    depth = max(1, int(getattr(args, 'signature_depth', 1)))
    spline_substeps = max(1, int(getattr(args, "logsig_spline_substeps", 4)))
    print(f"[LogSig] spline-first: substeps={spline_substeps} (iisignature may auto-cap), depth={depth}")
    x_tra_logsig = _logsig_stream_encode(x_tra, depth=depth, spline_substeps=spline_substeps)
    x_val_logsig = _logsig_stream_encode(x_val, depth=depth, spline_substeps=spline_substeps)
    x_test_logsig = _logsig_stream_encode(x_test, depth=depth, spline_substeps=spline_substeps)

    # Standardize logsignature channels using train statistics
    x_tra_logsig, x_val_logsig, x_test_logsig = _standardize_by_train(
        x_tra_logsig, x_val_logsig, x_test_logsig
    )

    # Optionally concat raw augmented path [time, raw] with logsignature features
    use_concat_raw = bool(getattr(args, 'logsig_concat_raw', True))
    if use_concat_raw:
        x_tra_raw = _build_augmented_path(x_tra)
        x_val_raw = _build_augmented_path(x_val)
        x_test_raw = _build_augmented_path(x_test)
        x_tra_ctrl = torch.cat([x_tra_raw, x_tra_logsig], dim=-1)
        x_val_ctrl = torch.cat([x_val_raw, x_val_logsig], dim=-1)
        x_test_ctrl = torch.cat([x_test_raw, x_test_logsig], dim=-1)
    else:
        x_tra_ctrl, x_val_ctrl, x_test_ctrl = x_tra_logsig, x_val_logsig, x_test_logsig

    # Time grid for spline and CDE solver (length = lag)
    times = torch.linspace(0, args.lag - 1, args.lag, dtype=x_tra_ctrl.dtype, device=x_tra_ctrl.device)

    # Natural cubic spline coeffs; inputs are [B, N, T, C]
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra_ctrl.transpose(1, 2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val_ctrl.transpose(1, 2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test_ctrl.transpose(1, 2))
    ##############get dataloader######################
    train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)

    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler, times

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Maritime dataloader self-check')
    parser.add_argument('--dataset', default='nanhai', choices=['nanhai', 'bohai'])
    parser.add_argument('--num_nodes', default=265, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--missing_test', default=False, type=bool)
    parser.add_argument('--missing_rate', default=0.1, type=float)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler, times = get_dataloader_cde(
        args, normalizer='std', tod=False, dow=False, weather=False, single=True)
    print('cde dataloader OK, times:', times.shape)
