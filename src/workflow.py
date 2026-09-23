"""Configuration loading and command-line workflows for NMGAT."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .constants import FEATURES


def validate_inputs(folder):
    frames = {s: pd.read_csv(Path(folder) / f'{s}.csv') for s in ('train', 'test', 'unlabeled')}
    seen = set()
    for split, frame in frames.items():
        if frame.empty:
            raise ValueError(f'{split}: empty dataset')
        if frame.duplicated(['Z', 'N']).any():
            raise ValueError(f'{split}: duplicate (Z,N) keys')
        coords = frame[['Z', 'N', 'A']].to_numpy()
        if not np.isfinite(coords).all() or not np.equal(coords, np.round(coords)).all():
            raise ValueError(f'{split}: noninteger nuclear coordinates')
        if not (frame.A == frame.Z + frame.N).all():
            raise ValueError(f'{split}: A != Z + N')
        keys = set(zip(frame.Z, frame.N))
        if seen & keys:
            raise ValueError(f'{split}: overlap with another split')
        seen |= keys
        if not np.isfinite(frame[list(FEATURES)].to_numpy()).all():
            raise ValueError(f'{split}: nonfinite features')
        if split != 'unlabeled':
            if not np.isfinite(frame[['Eexp', 'residual']].to_numpy()).all():
                raise ValueError(f'{split}: nonfinite targets')
            if not np.allclose(frame.residual, frame.Eth - frame.Eexp, rtol=0, atol=1e-8):
                raise ValueError(f'{split}: residual must equal Eth - Eexp')
    if not np.isfinite(frames['train'].uncertainty).all() or (frames['train'].uncertainty < 0).any():
        raise ValueError('train: invalid experimental uncertainty')
    return frames


ROOT = Path(__file__).resolve().parents[1]


def expand_graph(module, cfg, data, path):
    """Append unlabelled global nodes while preserving original order and scaler set."""
    frame = pd.read_csv(path)
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Global feature file missing columns: {sorted(missing)}')
    if frame.duplicated(['Z', 'N']).any():
        raise ValueError('Global feature file contains duplicate nuclides')
    if not np.isfinite(frame[list(FEATURES)].to_numpy()).all():
        raise ValueError('Global features must be finite')
    coords = frame[['Z', 'N', 'A']].to_numpy()
    if not np.equal(coords, np.round(coords)).all() or not (frame.A == frame.Z + frame.N).all():
        raise ValueError('Invalid global nuclear coordinates')
    indexed = frame.set_index(['Z', 'N'])
    absent = set(data.coords) - set(indexed.index)
    if absent:
        raise ValueError(f'Global graph must include all original nodes; missing {len(absent)}')
    overlap = frame.set_index(['Z','N'], drop=False).loc[data.coords, list(FEATURES)].to_numpy(dtype=np.float32)
    if not np.array_equal(overlap, data.x.cpu().numpy()):
        raise ValueError('Global features disagree with original-graph features at float32 precision')
    existing = set(data.coords)
    extra = frame.loc[[(int(z),int(n)) not in existing for z,n in zip(frame.Z,frame.N)]]
    n = len(extra)
    if n == 0:
        raise ValueError('Global graph has no additional nodes')
    all_coords = data.coords + list(zip(extra.Z.astype(int), extra.N.astype(int)))
    nan = torch.full((n,), float('nan'), dtype=torch.float32)
    fields = dict(coords=all_coords, a_vals=data.a_vals+extra.A.astype(int).tolist(),
                  x=torch.cat([data.x, torch.tensor(extra[list(FEATURES)].to_numpy(),dtype=torch.float32)]),
                  eth=torch.cat([data.eth, torch.tensor(extra.Eth.to_numpy(),dtype=torch.float32)]),
                  eexp=torch.cat([data.eexp,nan]), residual=torch.cat([data.residual,nan]),
                  edge_index=module.build_edge_index(all_coords,cfg.neighbor_offsets),
                  unlabeled_idx=data.unlabeled_idx+list(range(len(data.coords),len(all_coords))))
    if hasattr(data,'dataset_split'):
        fields.update(dataset_split=data.dataset_split+['unlabeled']*n,
                      participates_in_loss=data.participates_in_loss+[False]*n,
                      train_uncertainty=data.train_uncertainty+[float('nan')]*n)
    return replace(data, **fields)


def load_experiment(config_path=None, output=None, device='cpu', epochs=None,
                    *, model='WS4', variant='A', graph=None, global_features=None):
    path = Path(config_path or ROOT/'configs/aligned.json').resolve()
    settings = json.loads(path.read_text(encoding='utf8'))
    root = next((p for p in path.parents if (p/'configs/aligned.json').is_file()), ROOT)
    aligned = settings.get('protocol') in ('aligned', 'pretrained')
    if aligned:
        settings.update(name=f'{model}-{variant}',model=model,variant=variant,
                        data_path=f'data/processed/{model}',
                        uncertainty_trim_ratio=settings['variant_trim_ratios'][variant])
    from . import training as module
    data_path = root / settings['data_path']
    validate_inputs(data_path)
    accepted = module.TrainConfig.__dataclass_fields__
    kwargs = {k: v for k, v in settings.items() if k in accepted}
    kwargs.update(data_path=data_path, train_csv='train.csv', test_csv='test.csv',
                  unlabeled_csv='unlabeled.csv', device=device,
                  neighbor_offsets=module.make_default_offsets(settings['radius'], True))
    if output is not None:
        kwargs.update(save_dir=Path(output) / 'artifacts', plot_dir=Path(output) / 'plots')
    if epochs is not None:
        kwargs['epochs'] = epochs
    cfg = module.TrainConfig(**kwargs)
    data = module.prepare_data(cfg, feature_cols=FEATURES)
    settings['graph'] = graph or settings.get('graph','small')
    if settings['graph'] == 'global':
        global_path = Path(global_features) if global_features else root/'data/global'/f'{settings["model"]}.csv'
        if not global_path.exists():
            raise FileNotFoundError(f'Global inputs not found: {global_path}. Supply --global-features with all 17 features; see README.md.')
        data = expand_graph(module,cfg,data,global_path)
        settings['global_features_path'] = str(global_path.resolve())
        settings['global_features_sha256'] = hashlib.sha256(global_path.read_bytes()).hexdigest()
    return module, cfg, data, settings


def predict_checkpoint(module, cfg, data, checkpoint):
    """Load an EMA or plain checkpoint and evaluate it on its training graph."""
    state = torch.load(checkpoint, map_location=data.x.device, weights_only=True)
    ema_updates = int(state['n_averaged']) if 'n_averaged' in state else 0
    if 'n_averaged' in state:
        state = {k.removeprefix('module.'): v for k, v in state.items() if k != 'n_averaged'}
    model = module.MassNet(in_dim=len(FEATURES), hidden_dim=cfg.hidden_dim,
                          heads=cfg.heads, dropout=cfg.dropout, concat=cfg.concat, norm=cfg.norm).to(data.x.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    standardized, _, _ = module.standardize_by_idx(data.x, data.train_idx)
    with torch.inference_mode():
        pred = model(standardized, data.edge_index).cpu().numpy()
    return pred, ema_updates


def train_main():
    p = argparse.ArgumentParser(description='Train a documented NMGAT experiment.')
    p.add_argument('--config', help='Optional configuration; default is the shared A/B protocol')
    p.add_argument('--model', choices=['WS4','DZ','LDM'], default='WS4')
    p.add_argument('--variant', choices=['A','B'], default='A')
    p.add_argument('--graph', choices=['small','global'], default=None)
    p.add_argument('--global-features', type=Path)
    p.add_argument('--output', required=True, help='New directory; must not already exist')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int)
    p.add_argument('--num-seeds', type=int)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    module, cfg, cpu_data, settings = load_experiment(args.config, output, args.device, args.epochs,
        model=args.model,variant=args.variant,graph=args.graph,global_features=args.global_features)
    count = settings['num_seeds'] if args.num_seeds is None else args.num_seeds
    if count <= 0 or cfg.epochs <= 0:
        p.error('epochs and num-seeds must be positive')
    settings.update(epochs=cfg.epochs, num_seeds=count)
    output.mkdir(parents=True, exist_ok=False)
    data = cpu_data.to(module.select_device(args.device))
    metadata = {'experiment': settings,
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
                'num_seeds': count, 'base_seed': settings['base_seed'], 'features': list(FEATURES),
                'python': sys.version, 'platform': platform.platform(),
                'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torch-geometric', 'numpy', 'pandas']},
                'input_sha256': {s: hashlib.sha256((cfg.data_path / f'{s}.csv').read_bytes()).hexdigest() for s in ['train', 'test', 'unlabeled']}}
    _, mean, std = module.standardize_by_idx(data.x, data.train_idx)
    metadata.update(scaler_mean=mean.cpu().tolist(), scaler_std=std.cpu().tolist(),
                    coordinates=data.coords, train_idx=data.train_idx, test_idx=data.test_idx,
                    unlabeled_idx=data.unlabeled_idx,
                    loss_train_idx=getattr(data, 'loss_train_idx', data.train_idx))
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    rows, predictions = [], []
    for seed in range(settings['base_seed'], settings['base_seed'] + count):
        out = module.train_one_seed(data, cfg, seed)
        module.save_history_csv(out['history'], cfg.plot_dir / f'seed_{seed}' / 'history.csv')
        rows.append({k: out[k] for k in ['seed', 'final_train_rmsd', 'final_test_rmsd']})
        predictions.append(out['pred_all'])
    pd.DataFrame(rows).to_csv(output / 'seed_summary.csv', index=False)
    stack = np.stack(predictions).astype(np.float64)
    result = pd.DataFrame(data.coords, columns=['Z', 'N'])
    result['pred_residual'] = stack.mean(0)
    result['pred_binding_energy'] = data.eth.cpu().numpy() - stack.mean(0)
    result['prediction_std_ddof0_MeV'] = stack.std(0, ddof=0)
    result['split'] = ['train' if i in data.train_idx else 'test' if i in data.test_idx else 'unlabeled' for i in range(len(result))]
    result['participates_in_loss'] = [i in metadata['loss_train_idx'] for i in range(len(result))]
    result.to_csv(output / 'ensemble_predictions.csv', index=False)
    metrics = {'seeds':count,'graph':settings['graph'],'nodes':len(data.coords)}
    for name,indices in [('train',data.train_idx),('loss_train',metadata['loss_train_idx']),('test',data.test_idx)]:
        metrics[f'{name}_rmsd_MeV']=float(np.sqrt(np.mean((result.pred_binding_energy.to_numpy()[indices]-data.eexp.cpu().numpy()[indices])**2)))
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    print(output.resolve())


def predict_main():
    p = argparse.ArgumentParser(description='Predict using the same small/global graph used to train the checkpoint.')
    p.add_argument('--config')
    p.add_argument('--model', choices=['WS4','DZ','LDM'])
    p.add_argument('--variant', choices=['A','B'])
    p.add_argument('--graph', choices=['small','global'], default=None)
    p.add_argument('--global-features', type=Path)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    metadata_path=Path(a.checkpoint).resolve().parents[2]/'run_metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf8')) if metadata_path.exists() else None
    if metadata and 'experiment' in metadata:
        experiment=metadata['experiment']
        for field in ('model','variant','graph'):
            requested=getattr(a,field)
            if requested is not None and requested!=experiment[field]:
                p.error(f'--{field} disagrees with checkpoint training metadata ({experiment[field]})')
            setattr(a,field,experiment[field])
    else:
        for ancestor in Path(a.checkpoint).resolve().parents:
            if ancestor.name in [f'{m}-{v}' for m in ('WS4','DZ','LDM') for v in ('A','B')]:
                model,variant=ancestor.name.split('-')
                if a.graph=='global':p.error('Supplied historical weights were trained on the small graph. For global prediction use a checkpoint trained with --graph global.')
                if a.model and a.model!=model:p.error('Model does not match checkpoint directory')
                if a.variant and a.variant!=variant:p.error('Variant does not match checkpoint directory')
                a.model,a.variant,a.graph=model,variant,'small'
                break
    a.model=a.model or 'WS4';a.variant=a.variant or 'A';a.graph=a.graph or 'small'
    m, cfg, d, _ = load_experiment(a.config,model=a.model,variant=a.variant,graph=a.graph,global_features=a.global_features)
    if metadata:
        if [list(c) for c in d.coords]!=metadata['coordinates']:
            p.error('Input graph nodes/order differ from checkpoint training metadata')
        _,mean,std=m.standardize_by_idx(d.x,d.train_idx)
        if not np.allclose(mean.numpy(),metadata['scaler_mean'],rtol=1e-6,atol=1e-6) or not np.allclose(std.numpy(),metadata['scaler_std'],rtol=1e-6,atol=1e-6):
            p.error('Training scaler no longer matches input data')
        for field in ('hidden_dim','heads','dropout','concat','norm'):
            setattr(cfg,field,metadata['config'][field])
        cfg.neighbor_offsets=tuple(tuple(x) for x in metadata['config']['neighbor_offsets'])
        d.edge_index=m.build_edge_index(d.coords,cfg.neighbor_offsets)
    d = d.to(m.select_device(a.device))
    pred, _ = predict_checkpoint(m, cfg, d, a.checkpoint)
    result = pd.DataFrame(d.coords, columns=['Z', 'N'])
    result['pred_residual'] = pred
    result['pred_binding_energy'] = d.eth.cpu().numpy().astype(np.float64) - pred.astype(np.float64)
    with Path(a.output).open('x', encoding='utf8', newline='') as handle:
        result.to_csv(handle, index=False)
