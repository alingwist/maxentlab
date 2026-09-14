import numpy as np
from scipy.optimize import minimize as scipy_minimize
import pandas as pd


# Columns that describe the training data.
META_COLUMNS = ('Input', 'Output', 'Hidden', 'Frequency')

# Header spellings accepted for each canonical column.
_HEADER_ALIASES = {
    'input': 'Input',
    'inputs': 'Input',
    'ur': 'Input',
    'output': 'Output',
    'outputs': 'Output',
    'candidate': 'Output',
    'overt': 'Output',
    'sr': 'Output',
    'hidden': 'Hidden',
    'hidden structure': 'Hidden',
    'hidden_structure': 'Hidden',
    'parse': 'Hidden',
    'frequency': 'Frequency',
    'freq': 'Frequency',
    'probability': 'Frequency',
    'prob': 'Frequency',
    'count': 'Frequency',
    'counts': 'Frequency',
    'tokens': 'Frequency',
}

_PROBABILITY_SPELLINGS = ('probability', 'prob')
_COUNT_SPELLINGS = ('count', 'counts', 'tokens')


# I/O Functions

def read_input(filepath):
    """Read a tab-separated (or comma-separated .csv) input file.

    Expected format:
        Input <tab> Output <tab> Frequency <tab> C1 <tab> C2 <tab> ...
    or with hidden structure:
        Input <tab> Output <tab> Hidden <tab> Frequency <tab> C1 <tab> ...

    Frequencies may be probabilities, percentages or raw counts:
    they are always normalized per tableau.

    Returns:
        pd.DataFrame with columns Input, Output, [Hidden], Frequency, and the
        constraints in their original file order.
    """
    sep = ',' if str(filepath).lower().endswith('.csv') else '\t'
    data = pd.read_csv(filepath, sep=sep, encoding='utf-8')
    return prepare_dataframe(data)


def prepare_dataframe(data):
    """Canonicalize an already-loaded DataFrame (headers, dtypes, column order)."""
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]

    lowered = [c.lower() for c in data.columns]
    has_probability = any(c in _PROBABILITY_SPELLINGS for c in lowered)
    has_counts = any(c in _COUNT_SPELLINGS for c in lowered)

    rename = {}
    drop = []
    for col, low in zip(data.columns, lowered):
        canonical = _HEADER_ALIASES.get(low)
        if canonical is None:
            continue
        if canonical == 'Frequency' and has_probability and has_counts:
            # Prefer the counts column.
            if low in _PROBABILITY_SPELLINGS:
                drop.append(col)
                continue
        rename[col] = canonical

    data = data.drop(columns=drop).rename(columns=rename)

    for required in ('Input', 'Output', 'Frequency'):
        if required not in data.columns:
            raise ValueError(
                f"Input file is missing a '{required}' column. Expected columns: "
                "Input, Output, [Hidden], Frequency, then one column per constraint."
            )

    # Force the label columns to string so that numeric-looking item names
    # do not collide with real strings in unique / groupby / sort operations.
    for col in ('Input', 'Output', 'Hidden'):
        if col in data.columns:
            data[col] = data[col].astype(str)

    con_names = [c for c in data.columns if c not in META_COLUMNS]
    if not con_names:
        raise ValueError("Input file contains no constraint columns.")

    data['Frequency'] = pd.to_numeric(data['Frequency'], errors='coerce').fillna(0.0)
    for col in con_names:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)

    ordered = ['Input', 'Output']
    if 'Hidden' in data.columns:
        ordered.append('Hidden')
    ordered += ['Frequency'] + con_names
    return data[ordered]


def format_output(pred_df, weights, constraint_names):
    """Render predictions as the tab-separated text written by write_output."""
    w_vals = np.round(np.asarray(weights, dtype=float), 4)
    cols = list(pred_df.columns)
    lines = ['\t'.join(cols)]

    # Weights row: place weights under their constraint columns
    w_map = dict(zip(constraint_names, w_vals))
    lines.append('\t'.join(str(w_map[col]) if col in w_map else '' for col in cols))

    # Data rows
    for _, row in pred_df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, float):
                vals.append(f'{v:.6g}')
            else:
                vals.append(str(v))
        lines.append('\t'.join(vals))

    return '\n'.join(lines) + '\n'


def write_output(pred_df, weights, constraint_names, filepath):
    """Write predictions to file with weights in the second row."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(format_output(pred_df, weights, constraint_names))


# Internal Helpers

def constraint_columns(data):
    """Constraint column names, in file order."""
    return [c for c in data.columns if c not in META_COLUMNS]


def has_hidden_structure(data):
    """True if the data carries a Hidden column."""
    return 'Hidden' in data.columns


def _get_input_groups(inputs):
    """Precompute input groupings as (name, index_array) tuples."""
    groups = []
    seen = {}
    for i, inp in enumerate(inputs):
        if inp not in seen:
            seen[inp] = len(groups)
            groups.append((inp, []))
        groups[seen[inp]][1].append(i)
    return [(name, np.array(idx)) for name, idx in groups]


class _Groups:
    """Row groupings for one dataset."""

    def __init__(self, inputs, outputs=None, heads=None):
        self.inputs = inputs
        self.outputs = outputs
        self.heads = heads

    @property
    def hidden(self):
        return self.outputs is not None


def _build_groups(data):
    """Build the row groupings used by every computation below."""
    input_groups = _get_input_groups(data['Input'].values)
    if not has_hidden_structure(data):
        return _Groups(input_groups)

    overt = data['Output'].values
    output_groups = []
    heads = []
    for _, idx in input_groups:
        seen = {}
        buckets = []
        for i in idx:
            key = overt[i]
            if key not in seen:
                seen[key] = len(buckets)
                buckets.append([])
            buckets[seen[key]].append(i)
        buckets = [np.array(b) for b in buckets]
        output_groups.append(buckets)
        heads.extend(int(b[0]) for b in buckets)
    return _Groups(input_groups, output_groups, np.array(heads, dtype=int))


def _compute_probs(violations, weights, input_groups):
    """Compute predicted candidate probabilities.

    P(y|x) = exp(-v_y . w) / Z(x)
    where Z(x) = sum_y' exp(-v_y' . w)

    With hidden structure these are per-parse probabilities; sum them over the
    parses of an output with _overt_probs to get the overt distribution.
    """
    harmonies = violations @ weights
    probs = np.zeros(len(violations))

    for _, idx in input_groups:
        neg_h = -harmonies[idx]
        max_neg_h = np.max(neg_h)
        exp_vals = np.exp(neg_h - max_neg_h)
        Z = np.sum(exp_vals)
        probs[idx] = exp_vals / Z

    return probs


def _overt_probs(probs, groups):
    """Sum per-parse probabilities over the parses of each overt output."""
    if not groups.hidden:
        return probs
    out = np.empty_like(probs)
    for buckets in groups.outputs:
        for b in buckets:
            out[b] = np.sum(probs[b])
    return out


def _overt_frequency(values):
    """Reduce the frequencies on the parse rows of one overt output to one number.

    Frequency is a property of the overt output, not of a parse, so both of the
    conventions in use are accepted:
      - the output's frequency repeated on each of its parse rows (all equal)
        -> that value;
      - the frequency divided among its parse rows -> their sum.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values[0])
    return float(np.sum(values))


def _compute_observed(freqs, groups):
    """Normalize frequencies to observed probabilities per tableau."""
    observed = np.zeros(len(freqs))

    if not groups.hidden:
        for _, idx in groups.inputs:
            total = np.sum(freqs[idx])
            if total > 0:
                observed[idx] = freqs[idx] / total
            else:
                observed[idx] = 1.0 / len(idx)
        return observed

    for (_, _idx), buckets in zip(groups.inputs, groups.outputs):
        per_output = np.array([_overt_frequency(freqs[b]) for b in buckets])
        total = np.sum(per_output)
        if total > 0:
            q = per_output / total
        else:
            q = np.full(len(buckets), 1.0 / len(buckets))
        for q_y, b in zip(q, buckets):
            observed[b] = q_y
    return observed


def _build_prior_vector(n_constraints, reg, sigma, alpha):
    """Build per-constraint regularization coefficients.

    For L2: coefficient = 1 / (2 * sigma^2)    -> penalty = coeff * (w - mu)^2
    For L1: coefficient = alpha                 -> penalty = coeff * |w - mu|
    """
    prior = np.zeros(n_constraints)

    if reg == 'L2':
        sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=float))
        if sigma_arr.size == n_constraints:
            prior[:] = 1.0 / (2.0 * sigma_arr ** 2)
        else:
            prior[:] = 1.0 / (2.0 * float(sigma_arr[0]) ** 2)
    elif reg == 'L1':
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        if alpha_arr.size == n_constraints:
            prior[:] = alpha_arr
        else:
            prior[:] = float(alpha_arr[0])

    return prior


def _objective_and_gradient(weights, violations, observed, groups,
                            reg, mu, prior_coeff):
    """Compute negative log-likelihood + regularization and its gradient.

    Objective:
        NLL = -sum_x sum_y q(y|x) * log p(y|x) + prior
    Gradient:
        dNLL/dw = sum_x (E_q[v|x] - E_p[v|x]) + d(prior)/dw

    With hidden structure, p(y|x) is the overt probability (summed over the
    parses of y) and E_q[v|x] weights each parse by its posterior share
    q(y|x) * p(parse | y, x) — the standard expected-count form.
    """
    probs = _compute_probs(violations, weights, groups.inputs)

    if not groups.hidden:
        mask = observed > 0
        safe_probs = np.maximum(probs, 1e-300)
        nll = -np.sum(observed[mask] * np.log(safe_probs[mask]))
        target = observed
    else:
        overt = _overt_probs(probs, groups)
        safe_overt = np.maximum(overt, 1e-300)
        # One term per overt output: take each output's representative row.
        head_obs = observed[groups.heads]
        mask = head_obs > 0
        nll = -np.sum(head_obs[mask] * np.log(safe_overt[groups.heads][mask]))
        # Split each output's observed probability among its parses in
        # proportion to the model's current posterior over those parses.
        target = observed * (probs / safe_overt)

    # Gradient of NLL (observed - expected violations)
    gradient = np.zeros(len(weights))
    for _, idx in groups.inputs:
        diff = target[idx] - probs[idx]
        gradient += diff @ violations[idx]

    # Regularization
    if reg == 'L2':
        nll += np.sum(prior_coeff * (weights - mu) ** 2)
        gradient += 2.0 * prior_coeff * (weights - mu)
    elif reg == 'L1':
        nll += np.sum(prior_coeff * np.abs(weights - mu))
        gradient += prior_coeff * np.sign(weights - mu)

    return nll, gradient


def _scalar_or_array(val, n, default=None):
    """Convert scalar/None to array of length n."""
    if val is None:
        return [default] * n
    elif np.isscalar(val):
        return np.full(n, float(val))
    else:
        return np.asarray(val, dtype=float)


# Prediction

def predict_probabilities(data, weights, constraint_names=None):
    """Compute predicted probabilities for each candidate.

    Args:
        data: DataFrame with Input, Output, [Hidden], Frequency, + constraints.
        weights: array of constraint weights.
        constraint_names: constraint column names.

    Returns:
        DataFrame with the original columns plus:
            Harmony   -(violations . weights)
            Predicted predicted probability of the candidate
            Observed  observed probability of the candidate's overt output
            Error     Observed - Predicted

        With hidden structure, two prediction columns are returned instead of
        one: Predicted_parse (the probability of that individual parse) and
        Predicted (the probability of the overt output, summed over its
        parses). Error compares observed overt probability
        against predicted overt probability.
    """
    if constraint_names is None:
        constraint_names = constraint_columns(data)

    violations = data[constraint_names].values.astype(float)
    weights = np.asarray(weights, dtype=float)
    groups = _build_groups(data)

    probs = _compute_probs(violations, weights, groups.inputs)
    overt = _overt_probs(probs, groups)
    freqs = data['Frequency'].values.astype(float)
    observed = _compute_observed(freqs, groups)

    # Build result DataFrame in one shot to avoid fragmentation
    result_dict = {'Input': data['Input'].values, 'Output': data['Output'].values}
    if groups.hidden:
        result_dict['Hidden'] = data['Hidden'].values
    result_dict['Frequency'] = freqs
    for cn in constraint_names:
        result_dict[cn] = data[cn].values
    result_dict['Harmony'] = np.round(-(violations @ weights), 6)
    if groups.hidden:
        result_dict['Predicted_parse'] = np.round(probs, 6)
    result_dict['Predicted'] = np.round(overt, 6)
    result_dict['Observed'] = np.round(observed, 6)
    result_dict['Error'] = np.round(observed - overt, 6)

    return pd.DataFrame(result_dict)


def compute_accuracy(pred_df, threshold=0.05):
    """Threshold accuracy: an input counts as correct if all of its candidates
    have |error| <= threshold.

    Returns:
        (n_correct, n_total, accuracy_pct)
    """
    inputs = pred_df['Input'].unique()
    correct = 0
    for inp in inputs:
        errors = pred_df.loc[pred_df['Input'] == inp, 'Error'].values
        if np.all(np.abs(errors) <= threshold):
            correct += 1
    return correct, len(inputs), 100.0 * correct / len(inputs) if len(inputs) > 0 else 0.0


def compute_errors(pred_df):
    """Summarize fit: mean absolute error overall and per tableau.

    With hidden structure the error is computed once per overt output rather
    than once per parse row, so a tableau is not penalized for spreading an
    output's probability over several parses.

    Returns:
        (mae_overall, {input_name: mae_for_that_tableau})
    """
    has_hidden = 'Hidden' in pred_df.columns
    if has_hidden:
        rows = pred_df.drop_duplicates(subset=['Input', 'Output'])
    else:
        rows = pred_df

    per_tableau = {}
    for inp, sub in rows.groupby('Input', sort=False):
        per_tableau[inp] = float(np.mean(np.abs(sub['Error'].values)))

    overall = float(np.mean(np.abs(rows['Error'].values))) if len(rows) else 0.0
    return overall, per_tableau


# Optimization

def optimize_weights(
    data,
    reg='L2',
    sigma=1.0,
    alpha=1.0,
    mu=0.0,
    init_weights=0.0,
    lower_bound=0.0,
    upper_bound=None,
    constraint_names=None,
    verbose=True,
):
    """Optimize MaxEnt constraint weights with L-BFGS-B.

    Args:
        data: DataFrame (Input, Output, [Hidden], Frequency, C1, C2, ...).

        Regularization:
        reg:    'L2' (Gaussian prior), 'L1', or None.
        sigma:  L2 prior std dev (higher = weaker prior). Default 1.0.
        alpha:  L1 prior strength (higher = stronger). Default 1.0.
        mu:     Prior mean. Default 0.0.

        Weight initialization and bounds:
        init_weights: initial weight value(s). Float or array. Default 0.0.
        lower_bound:  lower bound on weights. Float, array, or None.
                      Default 0.0 (non-negative weights).
        upper_bound:  upper bound on weights. Float, array, or None.

        constraint_names: constraint columns to fit (default: all non-metadata).

    Returns:
        dict with keys:
            weights:          optimized weight array
            loglik:           log-likelihood at optimum (without prior)
            constraint_names: list of constraint names (same order as weights)
            hidden:           whether hidden structure was marginalized over
            converged:        bool
            message:          optimizer message string
    """
    # Setup
    if constraint_names is None:
        constraint_names = constraint_columns(data)
    n_con = len(constraint_names)

    violations = data[constraint_names].values.astype(float)
    freqs = data['Frequency'].values.astype(float)
    groups = _build_groups(data)
    observed = _compute_observed(freqs, groups)

    # Prior coefficients per constraint
    prior_coeff = _build_prior_vector(n_con, reg, sigma, alpha)

    # Ensure mu is a numpy-compatible value
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
    if mu_arr.size == 1:
        mu_val = float(mu_arr[0])
    elif mu_arr.size == n_con:
        mu_val = mu_arr
    else:
        raise ValueError(
            f"mu length ({mu_arr.size}) != number of constraints ({n_con})"
        )

    # Initial weights
    if np.isscalar(init_weights):
        w0 = np.full(n_con, float(init_weights))
    else:
        w0 = np.asarray(init_weights, dtype=float)
        if len(w0) != n_con:
            raise ValueError(
                f"init_weights length ({len(w0)}) != number of constraints ({n_con})"
            )

    # L-BFGS-B
    lb = _scalar_or_array(lower_bound, n_con, default=None)
    ub = _scalar_or_array(upper_bound, n_con, default=None)
    bounds = list(zip(lb, ub))

    def obj_grad(w):
        return _objective_and_gradient(
            w, violations, observed, groups, reg, mu_val, prior_coeff
        )

    result = scipy_minimize(
        obj_grad, w0, method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': 15000, 'ftol': 1e-15, 'gtol': 1e-10}
    )

    final_w = result.x
    converged = result.success
    message = result.message

    # Compute final log-likelihood (without regularization)
    probs = _compute_probs(violations, final_w, groups.inputs)
    overt = _overt_probs(probs, groups)
    if groups.hidden:
        head_obs = observed[groups.heads]
        mask = head_obs > 0
        loglik = np.sum(head_obs[mask]
                        * np.log(np.maximum(overt[groups.heads][mask], 1e-300)))
    else:
        mask = observed > 0
        loglik = np.sum(observed[mask] * np.log(np.maximum(probs[mask], 1e-300)))

    if verbose:
        status = 'converged' if converged else 'did not converge'
        print(f"Optimization {status}: {message}")
        print(f"Log-likelihood: {loglik:.4f}")

    return {
        'weights': final_w,
        'loglik': loglik,
        'constraint_names': constraint_names,
        'hidden': groups.hidden,
        'converged': converged,
        'message': str(message),
    }
