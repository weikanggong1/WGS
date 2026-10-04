"""Float64 PyTorch average-rank transform with R's AS241 quantile.

SPDX-License-Identifier: GPL-3.0-only
AS241 coefficients and evaluation order follow R src/nmath/qnorm.c,
R Core Team and Wichura (1988), https://svn.r-project.org/R/trunk/src/nmath/qnorm.c.
Only ordinary finite probabilities needed for ranked observations are used.
"""
import torch


def _horner(x, coefficients):
    result = torch.full_like(x, coefficients[0])
    for coefficient in coefficients[1:]:
        result = result * x + coefficient
    return result


def r_normal_quantile(probabilities):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if not bool(((p > 0) & (p < 1)).all()):
        raise ValueError("normal quantile requires finite probabilities strictly between zero and one")
    q = p - 0.5
    result = torch.empty_like(p)
    central = q.abs() <= 0.425
    r = 0.180625 - q[central].square()
    numerator = _horner(r, (2509.0809287301226727,33430.575583588128105,67265.770927008700853,
        45921.953931549871457,13731.693765509461125,1971.5909503065514427,133.14166789178437745,3.387132872796366608))
    denominator = _horner(r,(5226.495278852854561,28729.085735721942674,39307.89580009271061,
        21213.794301586595867,5394.1960214247511077,687.1870074920579083,42.313330701600911252,1.))
    result[central] = q[central] * numerator / denominator
    tail = ~central
    r = torch.sqrt(-torch.log(torch.where(q[tail] > 0, (0.5-p[tail])+0.5, p[tail])))
    middle = r <= 5
    z = r[middle] - 1.6
    value = torch.empty_like(r)
    value[middle] = _horner(z,(7.7454501427834140764e-4,.0227238449892691845833,.24178072517745061177,
        1.27045825245236838258,3.64784832476320460504,5.7694972214606914055,4.6303378461565452959,1.42343711074968357734)) / _horner(z,
        (1.05075007164441684324e-9,5.475938084995344946e-4,.0151986665636164571966,.14810397642748007459,
         .68976733498510000455,1.6763848301838038494,2.05319162663775882187,1.))
    z = r[~middle] - 5
    value[~middle] = _horner(z,(2.01033439929228813265e-7,2.71155556874348757815e-5,.0012426609473880784386,
        .026532189526576123093,.29656057182850489123,1.7848265399172913358,5.4637849111641143699,6.6579046435011037772)) / _horner(z,
        (2.04426310338993978564e-15,1.4215117583164458887e-7,1.8463183175100546818e-5,7.868691311456132591e-4,
         .0148753612908506148525,.13692988092273580531,.59983220655588793769,1.))
    result[tail] = torch.where(q[tail] < 0, -value, value)
    return result


def rank_inverse_normal_tensor(values, *, device=None):
    """Average ties and use REGENIE's Blom offset on each input column."""
    y = torch.as_tensor(values, dtype=torch.float64, device=device)
    single = y.ndim == 1
    if single:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] < 2 or not bool(torch.isfinite(y).all()):
        raise ValueError("RINT requires finite samples-by-traits values")
    n = y.shape[0]
    ranks = torch.empty_like(y)
    for column in range(y.shape[1]):
        sorted_values, indices = torch.sort(y[:, column], stable=True)
        _, inverse, counts = torch.unique_consecutive(sorted_values, return_inverse=True, return_counts=True)
        ends = counts.cumsum(0).to(y.dtype)
        starts = ends - counts + 1
        ranks[indices, column] = ((starts+ends)/2)[inverse]
    # Tensor denominators retain IEEE division. PyTorch scalar division can
    # multiply by a rounded reciprocal, changing upper-tail ranks by one ulp.
    probability = (ranks-0.375)/torch.full_like(ranks, n+0.25)
    result = r_normal_quantile(probability)
    return result[:, 0] if single else result
