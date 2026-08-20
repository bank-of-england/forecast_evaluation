"""Temporary backward-compatibility shims for the ``forecast_horizon`` -> ``horizon`` rename.

Everything in this module exists only to keep pre-rename calls working. Nothing here
changes a computation. To drop the compatibility layer in a future version, revert the
commit that added this file: it also removes the decorator lines that reference it.
"""

import functools
import warnings


def accept_forecast_horizon_kwarg(func):
    """Accept the old ``forecast_horizon=`` keyword as an alias for ``horizon=``.

    The alias is a rename, not a translation. ``horizon`` is the calendar distance from
    the forecast vintage to the target date, whereas the old ``forecast_horizon``
    argument selected the information horizon. The two coincide only when forecasts are
    published as soon as the previous period's outturn is released; where there is a
    publication lag, an unchanged call now selects a different set of observations.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "forecast_horizon" in kwargs:
            if "horizon" in kwargs:
                raise TypeError(
                    f"{func.__name__}() received both 'horizon' and the deprecated "
                    "'forecast_horizon'. Pass 'horizon' only."
                )

            warnings.warn(
                f"The 'forecast_horizon' argument of {func.__name__}() is deprecated; use 'horizon'. "
                "Note that 'horizon' is the calendar distance from the forecast vintage to the target "
                "date (target_minus_vintage), not the information horizon the old argument selected. "
                "The two differ whenever forecasts are published with a lag.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs["horizon"] = kwargs.pop("forecast_horizon")

        return func(*args, **kwargs)

    return wrapper


def resolve_forecast_horizon_column(key, columns):
    """Rewrite a ``forecast_horizon`` lookup on a result frame to ``horizon``.

    Results no longer carry a ``forecast_horizon`` column. The rewrite only happens on
    the access paths ``TestResult`` owns, where a warning can be raised at the point of
    use; the underlying DataFrame never gains an alias column, so exports keep the
    single, unambiguous name.

    Note that ``horizon`` is not the old column's values. The old column reported the
    information horizon, which the results now report as ``hac_maxlags``.

    Parameters
    ----------
    key : str, list of str, or any
        Column key, or list of column keys, as passed by the caller.
    columns : pandas.Index or list of str
        Columns of the frame being accessed.

    Returns
    -------
    str, list of str, or any
        The key with ``forecast_horizon`` replaced by ``horizon``, or the key unchanged
        when the rename does not apply.
    """
    if "forecast_horizon" in columns or "horizon" not in columns:
        return key

    if isinstance(key, str) and key == "forecast_horizon":
        _warn_forecast_horizon_column()
        return "horizon"

    if isinstance(key, list) and "forecast_horizon" in key:
        _warn_forecast_horizon_column()
        return ["horizon" if item == "forecast_horizon" else item for item in key]

    return key


def _warn_forecast_horizon_column():
    warnings.warn(
        "'forecast_horizon' is no longer a column of test results; use 'horizon'. "
        "Note that 'horizon' is the calendar distance from the forecast vintage to the target "
        "date (target_minus_vintage), not the information horizon the old column reported. "
        "The information horizon is now reported as 'hac_maxlags'.",
        FutureWarning,
        stacklevel=4,
    )
