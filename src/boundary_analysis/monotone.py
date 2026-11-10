"""Shape-constrained boundary fits anchored at H=0.5."""

import numpy as np


def fit_monotone_boundaries(boundaries, *, include_unconfirmed=True):
    """Least-squares projection within localization envelopes, independently per branch.

    With include_unconfirmed, enclosed point crossings without bounded intervals
    participate too, marked fitted_unconfirmed. They have sign/order constraints
    but no finite localization envelope. This does not create confidence bounds.
    An infeasible branch has no fitted values. Missing observations remain gaps.
    """
    import cvxpy as cp

    out = boundaries.sort_values("H").reset_index(drop=True).copy()
    if out.H.duplicated().any():
        raise ValueError("Duplicate H values")
    for side in (1, 2):
        out[f"mu{side}_fit"] = np.nan
        out[f"mu{side}_shift"] = np.nan
        out[f"mu{side}_fit_status"] = "unavailable"
        anchor = out.H.eq(0.5) & out[f"mu{side}_status"].eq("analytic")
        out.loc[anchor, [f"mu{side}_fit", f"mu{side}_shift"]] = 0.0
        out.loc[anchor, f"mu{side}_fit_status"] = "analytic"
    for left in (True, False):
        branch = out.H.lt(0.5) if left else out.H.gt(0.5)
        variables, constraints, objectives, groups = [], [], [], []
        for side in (1, 2):
            eligible = out[f"mu{side}_status"].eq("localized_grid")
            if include_unconfirmed:
                eligible |= (
                    out[f"mu{side}_status"].isin(["expand_window", "insufficient_precision"])
                    & out[f"mu{side}_raw"].notna()
                )
            ids = out.index[branch & eligible]
            if not len(ids):
                continue
            raw = out.loc[ids, f"mu{side}_raw"].to_numpy()
            localized = out.loc[ids, f"mu{side}_status"].eq("localized_grid").to_numpy()
            lo = np.where(localized, out.loc[ids, f"mu{side}_low"], -np.inf)
            hi = np.where(localized, out.loc[ids, f"mu{side}_high"], np.inf)
            if (
                not np.isfinite(raw).all()
                or not np.isfinite(lo[localized]).all()
                or not np.isfinite(hi[localized]).all()
                or np.any(lo > hi)
            ):
                raise ValueError("Invalid localized interval")
            x = cp.Variable(len(ids))
            constraints += [x <= 0 if side == 1 else x >= 0]
            if localized.any():
                constraints += [x[localized] >= lo[localized], x[localized] <= hi[localized]]
            if len(ids) > 1:
                increasing = left == (side == 1)
                constraints += [x[1:] >= x[:-1] if increasing else x[1:] <= x[:-1]]
            objectives.append(cp.sum_squares(x - raw))
            variables.append(x)
            groups.append((side, ids, raw, lo, hi))
        if not variables:
            continue
        problem = cp.Problem(cp.Minimize(sum(objectives)), constraints)
        try:
            problem.solve(solver="CLARABEL")
            status = problem.status
        except cp.error.SolverError:
            status = "solver_error"
        valid = status == cp.OPTIMAL
        if valid:
            for x, (side, ids, raw, lo, hi) in zip(variables, groups):
                v = x.value
                direction = 1 if left == (side == 1) else -1
                valid &= (
                    v is not None
                    and np.isfinite(v).all()
                    and np.all(v >= lo - 1e-7)
                    and np.all(v <= hi + 1e-7)
                    and np.all(direction * np.diff(v) >= -1e-7)
                    and np.all(v <= 1e-7 if side == 1 else v >= -1e-7)
                )
        for x, (side, ids, raw, lo, hi) in zip(variables, groups):
            if valid:
                out.loc[ids, f"mu{side}_fit"] = x.value
                out.loc[ids, f"mu{side}_shift"] = x.value - raw
                out.loc[ids, f"mu{side}_fit_status"] = np.where(
                    out.loc[ids, f"mu{side}_status"].eq("localized_grid"),
                    "fitted",
                    "fitted_unconfirmed",
                )
            else:
                out.loc[ids, f"mu{side}_fit_status"] = (
                    "incompatible_shape"
                    if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
                    else "solver_error"
                )
    return out
