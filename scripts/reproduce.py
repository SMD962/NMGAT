"""Recompute published ensemble predictions directly from frozen checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nmgat.workflow import load_experiment, predict_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['all','WS4','DZ','LDM'],default='all')
    parser.add_argument('--variant',choices=['all','A','B'],default='A')
    parser.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/reproduced')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0,help='For smoke tests only; 0 means all 200 seeds')
    args=parser.parse_args()
    if args.limit<0 or args.limit>200:parser.error('--limit must be in [0,200]')
    torch.set_num_threads(args.threads)
    models=['WS4','DZ','LDM'] if args.model=='all' else [args.model]
    variants=['A','B'] if args.variant=='all' else [args.variant]
    jobs=[]
    for model in models:
        for variant in variants:
            name=f'{model}-{variant}'
            count=args.limit or 200
            paths=[args.checkpoints/name/f'seed_{seed}'/'final.pt' for seed in range(2026013000,2026013000+count)]
            if any(not path.exists() for path in paths):
                parser.error(f'{name}: missing weights under {args.checkpoints}. Extract nmgat-checkpoints.zip next to the project and pass --checkpoints ../nmgat-checkpoints.')
            m,cfg,d,s=load_experiment(ROOT/'configs/pretrained.json',model=model,variant=variant,graph='small')
            jobs.append((name,paths,m,cfg,d,s))
    args.output.mkdir(parents=True,exist_ok=False)
    reports={}
    for name,paths,m,cfg,d,settings in jobs:
        d=d.to(m.select_device(args.device))
        predictions=[]; hashes={}
        for i,path in enumerate(paths):
            pred,_=predict_checkpoint(m,cfg,d,path)
            predictions.append(pred)
            hashes[path.parent.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if (i+1)%50==0: print(f'{name}: {i+1}/{len(paths)} checkpoints',flush=True)
        residual_stack=np.stack(predictions)
        stack=residual_stack.astype(np.float64)
        result=pd.DataFrame(d.coords,columns=['Z','N']);result['A']=d.a_vals
        result['pred_residual']=stack.mean(0)
        result['pred_binding_energy']=d.eth.cpu().numpy()-stack.mean(0)
        result['prediction_std_ddof0_MeV']=stack.std(0,ddof=0)
        result['prediction_std_ddof1_MeV']=stack.std(0,ddof=1) if len(paths)>1 else np.nan
        train=set(d.train_idx);test=set(d.test_idx);loss=set(getattr(d,'loss_train_idx',d.train_idx))
        result['split']=['train' if i in train else 'test' if i in test else 'unlabeled' for i in range(len(result))]
        result['participates_in_loss']=[i in loss for i in range(len(result))]
        metrics={'seeds':len(paths),'graph':'small','nodes':len(result),'full_200_seed_ensemble':len(paths)==200}
        for label,indices in [('train',d.train_idx),('loss_train',sorted(loss)),('test',d.test_idx)]:
            metrics[label+'_rmsd_MeV']=float(np.sqrt(np.mean((result.pred_binding_energy.to_numpy()[indices]-d.eexp.cpu().numpy()[indices])**2)))
        if len(paths)==200:
            reference=pd.read_csv(ROOT/f'results/{name}/summary_avg_all.csv')
            if not np.array_equal(reference[['Z','N']].to_numpy(),result[['Z','N']].to_numpy()):raise ValueError('Reference node mismatch')
            metrics['max_reference_residual_error_MeV']=float(np.max(np.abs(result.pred_residual-reference.pred_residual)))
            metrics['reference_match']=metrics['max_reference_residual_error_MeV']<1e-4
        else:
            metrics['reference_match']=None
        target=args.output/name;target.mkdir()
        result.to_csv(target/'predictions.csv',index=False)
        (target/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
        (target/'provenance.json').write_text(json.dumps({'settings':settings,'checkpoint_sha256':hashes},indent=2),encoding='utf8')
        reports[name]=metrics
        print(name,json.dumps(metrics),flush=True)
    (args.output/'summary.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
    if any(m['reference_match'] is False for m in reports.values()):raise SystemExit('Reference mismatch')


if __name__=='__main__':main()
