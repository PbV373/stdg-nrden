import math
import os
import sys
from pathlib import Path

if sys.stdout.encoding != 'UTF-8':
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
import time

from model.BasicTrainer_cde import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader_cde
from lib.TrainInits import print_model_parameters
from lib.load_dataset import MARITIME_DATASETS, maritime_npz_path
from lib.paths import rel_to_root
from model.Make_model import make_model

# SummaryWriter is imported lazily: old tensorboard/tensorflow stacks can break on numpy>=2 at import time


class CombinedLoss(nn.Module):
    """
    MAE + MSE + temporal difference (trend) loss over prediction steps.
    Optional linear horizon weighting: later steps weigh more to reduce long-horizon smoothing.
    """

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, horizon_weight_end=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.horizon_weight_end = horizon_weight_end

    def _h_weight(self, pred):
        """Shape (1, H, 1, ...) broadcast to pred; later steps weigh more if horizon_weight_end>1."""
        H = pred.shape[1]
        if H <= 1 or self.horizon_weight_end == 1.0:
            return 1.0
        w = torch.linspace(
            1.0, float(self.horizon_weight_end), H,
            device=pred.device, dtype=pred.dtype,
        )
        shape = (1, H) + (1,) * (pred.ndim - 2)
        return w.view(shape)

    def forward(self, pred, target):
        w = self._h_weight(pred)
        loss_mae = (torch.abs(pred - target) * w).mean()
        loss_mse = (((pred - target) ** 2) * w).mean()

        if pred.shape[1] > 1:
            pred_grad = pred[:, 1:, ...] - pred[:, :-1, ...]
            target_grad = target[:, 1:, ...] - target[:, :-1, ...]
            Hm = pred_grad.shape[1]
            if isinstance(w, torch.Tensor) and self.horizon_weight_end != 1.0:
                w_mid = 0.5 * (w[:, :Hm, ...] + w[:, 1: Hm + 1, ...])
            else:
                w_mid = 1.0
            loss_grad = (torch.abs(pred_grad - target_grad) * w_mid).mean()
        else:
            loss_grad = torch.tensor(0.0, device=pred.device)

        total_loss = self.alpha * loss_mae + self.beta * loss_mse + self.gamma * loss_grad

        return total_loss


class WarmupCosineScheduler:
    """Linear warmup then cosine decay learning-rate schedule."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # linear warmup
            lr_scale = self.current_epoch / self.warmup_epochs
            lr = self.base_lr * lr_scale
        else:
            # cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr_ratio * self.base_lr + 0.5 * (1 - self.min_lr_ratio) * self.base_lr * (
                        1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

# *************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'nanhai'  # nanhai / bohai; data at data/<name>.npz; logs under runs/<name>/; weights pre-trained/<name>.pth
MODEL = 'NRDEN'  # config model/<dataset>_<MODEL>.conf

# Parse --dataset / --model early so config path and log dirs stay consistent
for i, arg in enumerate(sys.argv):
    if arg == '--dataset' and i + 1 < len(sys.argv):
        DATASET = sys.argv[i + 1]
    if arg == '--model' and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]

if DATASET not in MARITIME_DATASETS:
    print(f'Maritime datasets only {sorted(MARITIME_DATASETS)}; got: {DATASET}')
    sys.exit(1)


def read_config_file(config_file):
    config = configparser.ConfigParser()
    try:
        # try multiple text encodings
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']
        for encoding in encodings:
            try:
                with open(config_file, 'r', encoding=encoding) as f:
                    config.read_file(f)
                    print(f"Read config with encoding {encoding}")
                    return config
            except UnicodeDecodeError:
                continue

        with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
            config.read_file(f)
            print("Read config with utf-8 and errors='ignore'")
            return config

    except FileNotFoundError:
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to read config: {e}")
        sys.exit(1)


config_file = rel_to_root('model', f'{DATASET}_{MODEL}.conf')
print('Read configuration file: %s' % (config_file))
config = read_config_file(config_file)

from lib.metrics import MAE_torch


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


# parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=0, type=int, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
args.add_argument('--comment', default='', type=str)


# data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
# model
args.add_argument('--model_type', default=config['model']['type'], type=str)
args.add_argument('--g_type', default=config['model']['g_type'], type=str)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--hid_dim', default=config['model']['hid_dim'], type=int)
args.add_argument('--hid_hid_dim', default=config['model']['hid_hid_dim'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
args.add_argument('--signature_depth', default=config['model'].get('signature_depth', 1), type=int,
                 help='truncation depth for true logsignature encoding')
args.add_argument('--logsig_concat_raw', default=config['model'].get('logsig_concat_raw', True), type=eval,
                 help='concatenate raw [time, x] path with logsig features')
args.add_argument('--logsig_spline_substeps', default=config['model'].get('logsig_spline_substeps', 4), type=int,
                 help='dense samples per knot interval after natural cubic spline (paper: spline then LogSig)')
args.add_argument('--solver', default='rk4', type=str)
# seasonal / spatial weights (chlorophyll-oriented graph variant)
args.add_argument('--seasonal_weight', default=0.4, type=float,
                 help='seasonal weight for chlorophyll data')
args.add_argument('--spatial_weight', default=0.6, type=float,
                 help='spatial weight for chlorophyll data')

# LR warmup epochs (used when lr_decay is enabled)
args.add_argument('--warmup_epochs', default=config['train'].get('warmup_epochs', 10), type=int,
                 help='warmup epochs for learning rate scheduling')
# graph construction method
args.add_argument('--graph_method', default=config['model'].get('graph_method', 'optimized'), type=str,
                 help='graph construction: optimized, attention, hybrid_attention, hybrid, chlorophyll_optimized; '
                      'enhanced_attention uses VectorField_g_enhanced via make_model_enhanced; other names fall back to VectorField_g_dynamic')

args.add_argument('--use_adaptive_sparse', default=False, type=eval,
                 help='use adaptive graph sparsification')
args.add_argument('--sparsify_method', default='simple', type=str,
                 choices=['topk', 'importance', 'hybrid', 'simple'],
                 help='method for graph sparsification')
args.add_argument('--min_topk', default=5, type=int,
                 help='minimum number of neighbors in adaptive sparsification')
args.add_argument('--max_topk', default=30, type=int,
                 help='maximum number of neighbors in adaptive sparsification')

args.add_argument('--graph_dropout', default=config['model'].get('graph_dropout', 0.2), type=float,
                 help='dropout rate for graph layers')
args.add_argument('--use_residual', default=config['model'].get('use_residual', True), type=eval,
                 help='use residual connections')
args.add_argument('--use_layer_norm', default=config['model'].get('use_layer_norm', True), type=eval,
                 help='use layer normalization')

args.add_argument('--n_heads', default=config['model'].get('n_heads', 4), type=int)
args.add_argument('--attention_dropout', default=config['model'].get('attention_dropout', 0.1), type=float)
# dynamic graph controls
args.add_argument('--use_dynamic_graph', default=config['model'].get('use_dynamic_graph', False), type=eval)
args.add_argument('--graph_alpha', default=config['model'].get('graph_alpha', 3), type=float)
args.add_argument('--graph_topk', default=config['model'].get('graph_topk', 20), type=int)
# train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
# args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval,
                  help='use real value for loss calculation')

args.add_argument('--missing_test', default=False, type=bool)
args.add_argument('--missing_rate', default=0.1, type=float)

# test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args.add_argument('--model_path', default='', type=str)
# log
args.add_argument('--log_dir', default='../runs', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--tensorboard', action='store_true', help='tensorboard')
args.add_argument('--deterministic', action='store_true',
                  help='more reproducible (slower); default off for cudnn.benchmark speed')

args = args.parse_args()
if args.dataset not in MARITIME_DATASETS:
    print(f'Dataset must be one of {sorted(MARITIME_DATASETS)}; got: {args.dataset}')
    sys.exit(1)
init_seed(args.seed, cudnn_benchmark=not args.deterministic)

GPU_NUM = args.device
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

print(args)

# ===== Pre-compute LogSignature control dimension before model build =====
# - signature_depth sets logsignature truncation order
# - discrete controls are mapped to a logsignature stream; channel count depends on the backend
# - model is built before the dataloader, so args.input_dim must be updated here
args.raw_input_dim = int(args.input_dim)
args.signature_depth = max(1, int(getattr(args, 'signature_depth', 1)))
raw_dim_for_logsig = args.raw_input_dim
# Raw feature dim may differ from config (e.g. config says 2 but NPZ has 1 channel)
if args.dataset in MARITIME_DATASETS:
    try:
        import numpy as _np
        _data = _np.load(maritime_npz_path(args.dataset))['data']
        if _data.ndim >= 3:
            raw_dim_for_logsig = int(_data.shape[2])
    except Exception as e:
        print(
            f"[LogSignature] could not read raw feature dim from {args.dataset} NPZ; "
            f"keeping input_dim={raw_dim_for_logsig}. Reason: {e}"
        )

logsig_in_channels = raw_dim_for_logsig + 1  # [time] + raw features
backend = None
try:
    import signatory  # type: ignore

    logsig_channels = int(signatory.logsignature_channels(logsig_in_channels, args.signature_depth))
    backend = "signatory"
except Exception:
    try:
        import iisignature  # type: ignore

        logsig_channels = int(iisignature.logsiglength(logsig_in_channels, args.signature_depth))
        backend = "iisignature"
    except Exception as e:
        raise ImportError(
            "Could not import a log-signature backend. Install one of:\n"
            "- pip install signatory (may be awkward on some Windows setups)\n"
            "- pip install iisignature (often easier on Windows)"
        ) from e

if bool(getattr(args, 'logsig_concat_raw', True)):
    args.input_dim = logsig_in_channels + logsig_channels
else:
    args.input_dim = logsig_channels

print(f"[LogSignature:{backend}] depth={args.signature_depth}, raw_dim={raw_dim_for_logsig}, in_channels={logsig_in_channels}, logsig_channels={logsig_channels}, concat_raw={args.logsig_concat_raw}, model_input_dim={args.input_dim}")

base_dir = Path(__file__).parent.parent  # repository root
save_name = time.strftime(
    "%m-%d-%Hh%Mm") + args.comment + "_" + args.dataset + "_" + args.model + "_" + args.model_type + "_" + "embed{" + str(
    args.embed_dim) + "}" + "hid{" + str(args.hid_dim) + "}" + "hidhid{" + str(args.hid_hid_dim) + "}" + "lyrs{" + str(
    args.num_layers) + "}" + "lr{" + str(args.lr_init) + "}" + "wd{" + str(args.weight_decay) + "}"

log_base = base_dir / "runs" / args.dataset
log_dir = log_base / save_name

try:
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory ready: {log_dir}")

    if not log_dir.is_dir():
        print(f"Log path is not a directory: {log_dir}")
        sys.exit(1)

    # writable check
    test_file = log_dir / "test_write.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        print(f"Log directory is not writable: {e}")
        sys.exit(1)

except Exception as e:
    print(f"Failed to create log directory: {e}")
    sys.exit(1)

args.log_dir = str(log_dir)

# Lazy SummaryWriter; disable tensorboard on failure so Trainer never writes to None
w = None
if args.tensorboard:
    try:
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(log_dir)
        print(f"SummaryWriter: {log_dir}")
    except Exception as e:
        print(f"TensorBoard disabled (training continues): {e}")
        print("Try: omit --tensorboard; or pip install 'tensorboard>=2.14'; or pip install 'numpy<2' for old tensorflow stacks.")
        args.tensorboard = False
        w = None

# init model
if args.graph_method == 'enhanced_attention':
    from model.Make_model import make_model_enhanced
    model, vector_field_f, vector_field_g = make_model_enhanced(args)
elif args.model_type == 'type1':
    model, vector_field_f, vector_field_g = make_model(args)
elif args.model_type == 'type1_temporal_only':
    model, vector_field_f, vector_field_g = make_model(args)
elif args.model_type == 'type1_spatial_only':
    model, vector_field_f, vector_field_g = make_model(args)
elif args.model_type == 'type1_spatial':
    model, vector_field_f, vector_field_g = make_model(args)
else:
    raise ValueError(f"Unsupported model_type: {args.model_type}")

model = model.to(args.device)

# place vector fields on the training device per model_type
if args.model_type == 'type1_temporal_only':
    # temporal NRDE only: vector_field_f only
    if vector_field_f is not None:
        vector_field_f = vector_field_f.to(args.device)
    vector_field_g = None
elif args.model_type == 'type1_spatial_only':
    # spatial NRDE only: vector_field_g only
    vector_field_f = None
    if vector_field_g is not None:
        vector_field_g = vector_field_g.to(args.device)
elif args.model_type == 'type1_temporal':
    # temporal-only branch
    if vector_field_f is not None:
        vector_field_f = vector_field_f.to(args.device)
    vector_field_g = None
elif args.model_type == 'type1_spatial':
    # spatial-only branch
    vector_field_f = None
    if vector_field_g is not None:
        vector_field_g = vector_field_g.to(args.device)
else:
    # full type1: both fields
    if vector_field_f is not None:
        vector_field_f = vector_field_f.to(args.device)
    if vector_field_g is not None:
        vector_field_g = vector_field_g.to(args.device)

print(model)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

# load dataset
train_loader, val_loader, test_loader, scaler, times = get_dataloader_cde(args,
                                                                          normalizer=args.normalizer,
                                                                          tod=args.tod, dow=False,
                                                                          weather=False, single=False)

# init loss function, optimizer
if args.loss_func == 'combined':
    # slightly higher beta/gamma + horizon weights for long steps; cost similar to pure MAE
    loss = CombinedLoss(
        alpha=0.45, beta=0.35, gamma=0.28, horizon_weight_end=1.42
    ).to(args.device)
elif args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'huber_loss':
    loss = torch.nn.HuberLoss(delta=1.0).to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,
                             weight_decay=args.weight_decay)

# learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Using warmup + cosine LR schedule...')
    lr_scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_ratio=0.01
    )
else:
    lr_scheduler = None



# start training
trainer = Trainer(model, vector_field_f, vector_field_g, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler, args.device, times,
                  w)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load(rel_to_root('pre-trained', f'{args.dataset}.pth')))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger, times)
else:
    raise ValueError