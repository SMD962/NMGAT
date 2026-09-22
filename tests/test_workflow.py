from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import unittest
import tempfile
import torch

from nmgat.workflow import load_experiment, validate_inputs, FEATURES
from nmgat.workflow import expand_graph
from nmgat.features import build_features

ROOT = Path(__file__).resolve().parents[1]


def test_inputs(model):
    d = validate_inputs(ROOT / 'data/processed' / model)
    assert [len(d[k]) for k in ('train', 'test', 'unlabeled')] == [2456, 23, 1085]


def test_heldout_labels_do_not_affect_training_inputs_or_gradients(variant):
    torch.set_num_threads(2)
    m, cfg, d, _ = load_experiment(model='WS4',variant=variant)
    assert tuple(d.feature_cols) == FEATURES
    assert set(FEATURES).isdisjoint({'Eexp', 'residual', 'uncertainty'})
    x, mean, _ = m.standardize_by_idx(d.x, d.train_idx)
    assert torch.allclose(mean, d.x[d.train_idx].mean(0))
    idx = getattr(d, 'loss_train_idx', d.train_idx)
    assert not set(idx) & set(d.test_idx)
    m.set_seed(17)
    model = m.MassNet(in_dim=17)
    model.eval()
    def gradients(target):
        model.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(model(x, d.edge_index)[idx], target[idx])
        loss.backward()
        return torch.cat([p.grad.ravel() for p in model.parameters() if p.grad is not None])
    g1 = gradients(d.residual)
    perturbed = d.residual.clone()
    perturbed[d.test_idx] = 1e6
    perturbed[d.unlabeled_idx] = -1e6
    assert torch.equal(g1, gradients(perturbed))
    if variant == 'B':
        assert len(d.loss_train_idx) == 2333
        assert len(d.masked_train_idx) == 123
        perturbed[d.masked_train_idx] = 1e6
        assert torch.equal(g1, gradients(perturbed))


def test_overlap_rejected(tmp_path):
    source = ROOT / 'data/processed/WS4'
    for name in ('train', 'test', 'unlabeled'):
        d = pd.read_csv(source / f'{name}.csv')
        if name == 'test':
            d = pd.concat([d, pd.read_csv(source / 'train.csv').iloc[:1]], ignore_index=True)
        d.to_csv(tmp_path / f'{name}.csv', index=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'overlap'):
        validate_inputs(tmp_path)


def test_feature_formulas():
    for model in ('WS4', 'DZ', 'LDM'):
        for d in validate_inputs(ROOT / 'data/processed' / model).values():
            assert np.allclose(d.I, (d.N-d.Z)/d.A)
            assert np.allclose(d.I2A, (d.N-d.Z)**2/d.A)
            assert np.allclose(d.A23, d.A**(2/3))
            assert np.allclose(d.Am13, d.A**(-1/3))
            assert np.allclose(d.Ec, d.Z**2/d.A**(1/3)*(1-0.76*d.Z**(-2/3)))
            assert np.array_equal(d.Npair, (d.Z % 2 == 0).astype(int)+(d.N % 2 == 0).astype(int))


class WorkflowTests(unittest.TestCase):
    def test_aligned_ab_only_loss_mask_differs(self):
        for model in ('WS4','DZ','LDM'):
            ma,ca,a,sa=load_experiment(model=model,variant='A')
            mb,cb,b,sb=load_experiment(model=model,variant='B')
            self.assertIs(ma,mb)
            da,db=vars(ca).copy(),vars(cb).copy()
            self.assertEqual(da.pop('uncertainty_trim_ratio'),0)
            self.assertEqual(db.pop('uncertainty_trim_ratio'),0.05)
            self.assertEqual(da,db)
            self.assertEqual(ca.epochs,12000)
            self.assertEqual(ca.noise_scope,'train')
            self.assertEqual(a.coords,b.coords)
            self.assertTrue(torch.equal(a.x,b.x))
            self.assertTrue(torch.equal(a.edge_index,b.edge_index))
            self.assertEqual(len(a.loss_train_idx),2456)
            self.assertEqual(len(b.loss_train_idx),2333)
            self.assertEqual(a.train_idx,b.train_idx)

    def test_ready_global_datasets(self):
        for model,nodes in [('WS4',7879),('DZ',7775),('LDM',7879)]:
            m,ca,a,_=load_experiment(model=model,variant='A',graph='global')
            _,cb,b,_=load_experiment(model=model,variant='B',graph='global')
            self.assertEqual(len(a.coords),nodes)
            self.assertEqual(a.coords,b.coords)
            self.assertTrue(torch.equal(a.x,b.x))
            self.assertTrue(torch.equal(a.edge_index,b.edge_index))
            self.assertEqual(len(a.loss_train_idx),2456)
            self.assertEqual(len(b.loss_train_idx),2333)
            self.assertTrue(torch.isnan(b.eexp[b.unlabeled_idx]).all())
            self.assertTrue(torch.isfinite(b.x).all())
            self.assertEqual(len(b.test_idx),23)

    def test_global_graph_preserves_scaler_and_hides_extra_targets(self):
        m,c,d,_=load_experiment(model='WS4',variant='B')
        small=pd.concat(validate_inputs(ROOT/'data/processed/WS4').values(),ignore_index=True)
        # Synthetic fixture tests graph plumbing only, not published global results.
        extra=small.iloc[:1].copy()
        extra['Z']=122;extra['N']=183;extra['A']=305
        extra['Eexp']=123456;extra['residual']=-123456
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'global.csv'
            pd.concat([small,extra],ignore_index=True).to_csv(path,index=False)
            large=expand_graph(m,c,d,path)
            self.assertEqual(len(large.coords),3565)
            self.assertEqual(large.train_idx,d.train_idx)
            self.assertEqual(large.loss_train_idx,d.loss_train_idx)
            self.assertTrue(torch.isnan(large.eexp[-1]))
            self.assertTrue(torch.isnan(large.residual[-1]))
            _,sm,ss=m.standardize_by_idx(d.x,d.train_idx)
            _,lm,ls=m.standardize_by_idx(large.x,large.train_idx)
            self.assertTrue(torch.equal(sm,lm))
            self.assertTrue(torch.equal(ss,ls))
            bad=pd.concat([small,extra],ignore_index=True)
            bad.loc[0,'Eth']+=1
            bad.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'disagree'):
                expand_graph(m,c,d,path)

    def test_all_inputs(self):
        for model in ('WS4', 'DZ', 'LDM'):
            with self.subTest(model=model):
                test_inputs(model)

    def test_label_isolation(self):
        for variant in ('A', 'B'):
            with self.subTest(variant=variant):
                test_heldout_labels_do_not_affect_training_inputs_or_gradients(variant)

    def test_overlap_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            test_overlap_rejected(Path(folder))

    def test_formulas(self):
        test_feature_formulas()

    def test_rebuild_all_features(self):
        for model in ('WS4', 'DZ', 'LDM'):
            for split, original in validate_inputs(ROOT / 'data/processed' / model).items():
                with self.subTest(model=model, split=split):
                    minimal = original[[c for c in ['Z','N','Esh','Edef','Eth','Eexp','uncertainty'] if c in original]]
                    rebuilt = build_features(minimal)
                    np.testing.assert_allclose(rebuilt[list(FEATURES)], original[list(FEATURES)], rtol=1e-10, atol=1e-10)
