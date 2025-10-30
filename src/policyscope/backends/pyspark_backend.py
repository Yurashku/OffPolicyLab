"""PySpark backend for large-scale Policyscope pipelines.

Only a subset of the Pandas-like API that Policyscope relies on is ported.
The adapter is intentionally stateless to simplify serialization across
executors. The API mirrors the small set of aggregate helpers that the
higher-level estimators use. Additional methods can be implemented
incrementally without affecting the Pandas code path.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from pyspark.sql import Column, DataFrame
else:  # pragma: no cover - help static analyzers without importing pyspark
    Column = object  # type: ignore[assignment]
    DataFrame = object  # type: ignore[assignment]


__all__ = ["SparkFrameAdapter"]


class SparkFrameAdapter:
    """A lightweight wrapper that exposes Policyscope-friendly helpers.

    Parameters
    ----------
    df:
        A :class:`pyspark.sql.DataFrame` to operate on.
    """

    def __init__(self, df: "DataFrame") -> None:  # pragma: no cover - simple init
        self._df = df

    # Public API -----------------------------------------------------------------
    def mean(
        self,
        columns: Union[str, Sequence[str], None],
        where: Optional[Union[str, "Column"]] = None,
        *,
        weight_col: Optional[str] = None,
        skipna: bool = True,
    ) -> Union[float, Dict[str, float]]:
        """Compute column-wise means efficiently on a Spark DataFrame.

        Parameters
        ----------
        columns:
            Column name or a sequence of column names. If ``None`` the adapter
            computes the mean for every numeric column in the frame.
        where:
            Optional filter expression (:class:`str` or :class:`~pyspark.sql.Column`).
        weight_col:
            Name of the column that contains non-negative observation weights.
        skipna:
            When ``True`` (default) null values are ignored. If ``False`` the
            result for a column is ``NaN`` whenever that column (or the weight
            column) contains nulls in the filtered dataset.

        Returns
        -------
        float | dict
            A single float when a single column is requested; otherwise a
            mapping from column name to the computed mean.
        """

        try:  # Import lazily to avoid importing PySpark in environments without it
            from pyspark.sql import Column as SparkColumn  # type: ignore
            from pyspark.sql.types import NumericType
        except ImportError as exc:  # pragma: no cover - exercised in runtime only
            raise ImportError(
                "SparkFrameAdapter requires pyspark. Install pyspark>=3.0 to "
                "use the Spark backend."
            ) from exc

        df = self._df
        if where is not None:
            if isinstance(where, str):
                df = df.filter(where)
            elif isinstance(where, SparkColumn):
                df = df.filter(where)
            else:  # pragma: no cover - defensive programming
                raise TypeError("where must be a SQL expression string or Column")

        selected: List[str]
        if columns is None:
            selected = [
                field.name
                for field in df.schema.fields
                if isinstance(getattr(field, "dataType", None), NumericType)
            ]
            if not selected:
                raise ValueError(
                    "No numeric columns available to compute the mean for."
                )
        elif isinstance(columns, str):
            selected = [columns]
        else:
            selected = list(columns)
            if not selected:
                raise ValueError("columns sequence must not be empty")

        # Deduplicate while preserving order for deterministic output.
        seen = set()
        ordered_columns: List[str] = []
        for col_name in selected:
            if col_name in seen:
                continue
            seen.add(col_name)
            ordered_columns.append(col_name)

        if not ordered_columns:
            raise ValueError("No columns remain after removing duplicates")

        single_output = len(ordered_columns) == 1

        if weight_col is not None and not isinstance(weight_col, str):
            raise TypeError("weight_col must be the name of a column")

        if weight_col is not None and weight_col not in df.columns:
            raise ValueError(f"Weight column '{weight_col}' is not present in the DataFrame")

        if weight_col is None:
            result = self._mean_unweighted(df, ordered_columns, skipna)
        else:
            result = self._mean_weighted(df, ordered_columns, weight_col, skipna)

        if single_output:
            # ``result`` is guaranteed to contain exactly one key.
            return next(iter(result.values()))
        return result

    # Internal helpers -----------------------------------------------------------
    @staticmethod
    def _mean_unweighted(df: "DataFrame", columns: Sequence[str], skipna: bool) -> Dict[str, float]:
        from pyspark.sql import functions as F

        stats_row = SparkFrameAdapter._aggregate_unweighted(df, columns, skipna)
        output: Dict[str, float] = {}
        for col, mean_value in stats_row:
            output[col] = mean_value
        return output

    @staticmethod
    def _aggregate_unweighted(
        df: "DataFrame", columns: Sequence[str], skipna: bool
    ) -> List[Tuple[str, float]]:
        from pyspark.sql import functions as F

        agg_exprs = []
        aliases: List[Tuple[str, str, str]] = []
        if skipna:
            for column in columns:
                agg_exprs.append(F.avg(F.col(column)).alias(column))
            row = df.agg(*agg_exprs).first()
            assert row is not None  # Spark always returns a row for aggregations
            return [
                (
                    column,
                    float(row[column]) if row[column] is not None else math.nan,
                )
                for column in columns
            ]

        # skipna=False branch: detect nulls and fall back to NaN.
        for column in columns:
            sum_alias = f"__sum_{column}"
            count_alias = f"__count_{column}"
            null_alias = f"__nulls_{column}"
            aliases.append((column, sum_alias, count_alias))
            agg_exprs.extend(
                [
                    F.sum(F.col(column)).alias(sum_alias),
                    F.count(F.col(column)).alias(count_alias),
                    F.sum(F.when(F.col(column).isNull(), F.lit(1))).alias(null_alias),
                ]
            )
        row = df.agg(*agg_exprs).first()
        assert row is not None
        output: List[Tuple[str, float]] = []
        for column, sum_alias, count_alias in aliases:
            null_alias = f"__nulls_{column}"
            nulls = row[null_alias] or 0
            if nulls:
                output.append((column, math.nan))
                continue
            count = row[count_alias]
            if count in (None, 0):
                output.append((column, math.nan))
                continue
            total = row[sum_alias]
            if total is None:
                output.append((column, math.nan))
            else:
                output.append((column, float(total) / float(count)))
        return output

    @staticmethod
    def _mean_weighted(
        df: "DataFrame",
        columns: Sequence[str],
        weight_col: str,
        skipna: bool,
    ) -> Dict[str, float]:
        from pyspark.sql import functions as F

        agg_exprs = []
        aliases: List[Tuple[str, str, str, str]] = []
        weight_column = F.col(weight_col)
        for column in columns:
            valid_mask = F.col(column).isNotNull() & weight_column.isNotNull()
            value_alias = f"__weighted_sum_{column}"
            weight_alias = f"__weight_sum_{column}"
            invalid_alias = f"__invalid_{column}"
            aliases.append((column, value_alias, weight_alias, invalid_alias))
            agg_exprs.extend(
                [
                    F.sum(
                        F.when(valid_mask, F.col(column) * weight_column).otherwise(F.lit(0.0))
                    ).alias(value_alias),
                    F.sum(F.when(valid_mask, weight_column).otherwise(F.lit(0.0))).alias(weight_alias),
                    F.sum(
                        F.when(valid_mask, F.lit(0)).otherwise(F.lit(1))
                    ).alias(invalid_alias),
                ]
            )
        row = df.agg(*agg_exprs).first()
        assert row is not None
        output: Dict[str, float] = {}
        for column, value_alias, weight_alias, invalid_alias in aliases:
            if not skipna and row[invalid_alias]:
                output[column] = math.nan
                continue
            weight_sum = row[weight_alias]
            if weight_sum in (None, 0.0):
                output[column] = math.nan
                continue
            weighted_sum = row[value_alias]
            if weighted_sum is None:
                output[column] = math.nan
            else:
                output[column] = float(weighted_sum) / float(weight_sum)
        return output
