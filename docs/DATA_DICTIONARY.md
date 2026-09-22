# Data dictionary

Coordinates are integers: Z is proton number, N neutron number, A=Z+N.
Binding energies, residuals, and experimental uncertainties are in MeV unless the
column explicitly says keV. Binding energies use a positive-bound convention.

The 17 input features are listed below. Their tensor order is defined in
`src/nmgat/constants.py`.

| Field | Definition |
|---|---|
| Z, N, A | Nuclear coordinates |
| I | (N-Z)/A |
| I2A | (N-Z)^2/A |
| A23, Am13 | A^(2/3), A^(-1/3) |
| Ec | Z^2 A^(-1/3) (1-0.76 Z^(-2/3)) |
| Npair | 0 odd-odd, 1 odd-A, 2 even-even |
| Zm, Nm | Ordinal shell indices |
| vp, vn | Distance to nearest boundary of the containing shell |
| Casten | vp*vn/(vp+vn), defined as zero for vp=vn=0 |
| Esh, Edef | WS4 shell/deformation quantities, also retained in DZ/LDM inputs |
| Eth | Binding energy from the corresponding base model |

`Eexp` is experimental binding energy; `residual=Eth-Eexp` is the regression
target; `uncertainty` is the experimental total-binding-energy uncertainty in MeV.
None is a model input. The model predicts residual r, then BE_pred=Eth-r.
Consequently `pred_minus_actual_residual` is the **negative** of BE_pred-Eexp.
Workbook `*_dev_MeV` columns use BE_pred-Eexp, so the two deviation signs differ
by definition.

The separate results export carries `split` and `participates_in_loss`. For B,
`split=train` can coexist with `participates_in_loss=False`: the label is masked,
not removed from the graph or the input scaler. Unlabelled target fields are NaN.

Source CSVs retain decimal precision. Historical training converted tensors to
float32; historical prediction CSV theory/experimental columns reflect that cast.
The Excel workbooks retain higher-precision source energies, so recomputing their
deviations against float32 values can yield small numerical differences.

Workbook uncertainty columns `*_NMGAT_std_MeV` are sample standard deviations
across 200 residual/binding predictions (`ddof=1`). They are not standard errors
of the mean, experimental uncertainty, or a calibrated total uncertainty.
The manuscript table's rounded parenthetical SDs match the population convention
(`ddof=0`). The difference changes the displayed rounding for WS4/Si-23 and
LDM/Ag-95.
