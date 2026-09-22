"""Check dataset tables and reference predictions.

Does not change the original Excel workbooks or CSV exports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    tables = {f.name[0]: pd.read_csv(f).set_index(['Z', 'N']) for f in (root/'data/tables').glob('*.csv')}
    checks, failures = {}, []
    for name, table in tables.items():
        if not table.index.is_unique: failures.append(f'{name}: duplicate keys')
        if not np.allclose(table.sigma_exp_keV, table.sigma_exp_MeV*1000, equal_nan=True):
            failures.append(f'{name}: uncertainty units')
        if name in ('3', '4'):
            checks[name] = {'rows': len(table), 'splits': table.Split.value_counts().to_dict()}
            for model in ('WS4','DZ','LDM'):
                for kind in ('bare','NMGAT'):
                    error = table[f'{model}_{kind}_dev_MeV']-(table[f'{model}_{kind}_MeV']-table.BE_exp_MeV)
                    if error.abs().max() > 1e-9: failures.append(f'{name}/{model}/{kind}: deviation identity')
                variant='A' if name=='3' else 'B'
                raw = pd.read_csv(root/f'results/{model}-{variant}/summary_avg_all.csv').set_index(['Z','N'])
                common = table.index.intersection(raw.index)
                if not np.allclose(table.loc[common,f'{model}_NMGAT_MeV'],raw.loc[common,'pred_binding_energy'],rtol=0,atol=1e-9):
                    failures.append(f'{name}/{model}: original-graph prediction mismatch')
                source = pd.concat([pd.read_csv(root/f'data/processed/{model}/{s}.csv') for s in ('train','test','unlabeled')]).set_index(['Z','N'])
                aligned = source.loc[raw.index]
                for raw_col, input_col in [('theory_binding_energy','Eth'),('actual_binding_energy','Eexp'),('actual_residual','residual')]:
                    if not np.allclose(raw[raw_col], aligned[input_col].to_numpy(dtype=np.float32), rtol=0, atol=1e-10, equal_nan=True):
                        failures.append(f'{name}/{model}: stored {raw_col} does not match input float32 values')
                if not np.allclose(table.loc[common,f'{model}_bare_MeV'],source.loc[common,'Eth'],rtol=0,atol=1e-9):
                    failures.append(f'{name}/{model}: bare prediction mismatch')
            tr = tables['1' if name=='3' else '2']
            if set(table.index[table.Split.eq('Train')]) != set(tr.index):
                failures.append(f'{name}: Train membership')
            if not np.allclose(table.loc[tr.index,'BE_exp_MeV'],tr.BE_exp_MeV,rtol=0,atol=1e-9):
                failures.append(f'{name}: training energies')
    # Every original workbook is a byte-identical copy of the author's supplied source.
    expected_hashes = {'1_AME2020_datasetA_2456.xlsx': '787a43af22dd9e531d6c6de56676967bcb2bcab96a0d6a8c00666b7b24552ae1', '2_AME2020_datasetB_2333.xlsx': 'c9a73bc7df97669bb5d3f9cb7e9ef9f07b3557051eb1b3582410af9c6d9eb7db', '3_trainA_predictions_7775.xlsx': '83059fe2047cd5a0491365ef304d5b6112ebc91115b90e0db617e5e2f4fec052', '4_trainB_predictions_7652.xlsx': '65748592a26e8b78e8157af94895161b0b51abf9700afbdef1a1b02b72bff894'}
    for file in (root/'data/tables_original').glob('*.xlsx'):
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != expected_hashes[file.name]:
            failures.append(f'workbook altered: {file.name}')
    report={'tables':checks,'failures':failures}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__':main()
