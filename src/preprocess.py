"""
preprocess.py
-------------
Unified builders for tabular preprocessing used across all models.

- ColumnTransformer with:
    * OneHotEncoder for categorical columns
    * (optional) StandardScaler for numeric columns
- (optional) PCA on the combined feature space
- Helpers to assemble a full sklearn Pipeline with an estimator

Design goals:
- Keep behavior consistent with existing scripts (sparse_threshold=1.0 so
  the output is dense when any dense transformer is present).
- Make PCA optional and easy to toggle.
- Centralize OneHotEncoder settings to avoid drift between models.

Usage:
    from src.preprocess import build_preprocessor, build_model_pipeline
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    *,
    scale_numeric: bool = False,
    sparse_ohe: bool = True,
    # Keep consistent with prior scripts: dense output if any dense part exists
    sparse_threshold: float = 1.0,
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that encodes categoricals and (optionally) scales numerics.

    Parameters
    ----------
    num_cols : list of str
        Numeric feature column names.
    cat_cols : list of str
        Categorical feature column names.
    scale_numeric : bool, default False
        If True, apply StandardScaler to numeric columns.
    sparse_ohe : bool, default True
        If True, OneHotEncoder returns a sparse matrix (sklearn >=1.2 uses 'sparse_output').
    sparse_threshold : float, default 1.0
        ColumnTransformer's sparse threshold. With 1.0 the combined output will be dense
        if any transformer outputs dense (matches your existing scripts).

    Returns
    -------
    ColumnTransformer
    """
    print(
        f"[PRE] Building preprocessor | "
        f"#num={len(num_cols)} scale_numeric={scale_numeric} | "
        f"#cat={len(cat_cols)} sparse_ohe={sparse_ohe} | "
        f"sparse_threshold={sparse_threshold}"
    )

    num_tf = "passthrough"
    if num_cols:
        if scale_numeric:
            num_tf = Pipeline([("scaler", StandardScaler())])
        else:
            num_tf = "passthrough"

    # Note: 'sparse_output' is the modern arg name (sklearn >=1.2).
    # Your existing code already uses it, so we keep it for consistency.
    cat_tf = "drop"
    if cat_cols:
        cat_tf = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=sparse_ohe,
        )

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
        n_jobs=None,  # easy to expose later if needed
    )
    return pre


def attach_pca(
    base: Pipeline | ColumnTransformer,
    *,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Optionally append a PCA stage after the base preprocessor.

    Notes
    -----
    - PCA expects dense input. With sparse_threshold=1.0 and a dense numeric
      branch, the ColumnTransformer output will be dense already. If your
      entire design is sparse (rare here), PCA will densify internally.
    """
    if n_components is None:
        print("[PCA] PCA not enabled (n_components=None). Returning base pipeline.")
        # Wrap in a Pipeline anyway for consistent return type
        if isinstance(base, Pipeline):
            return base
        return Pipeline([("pre", base)])

    print(f"[PCA] Attaching PCA with n_components={n_components}, random_state={random_state}")
    return Pipeline([
        ("pre", base),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
    ])


def build_model_pipeline(
    estimator,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    scale_numeric: bool = False,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    random_state: int = 42,
    sparse_ohe: bool = True,
    sparse_threshold: float = 1.0,
) -> Pipeline:
    """
    Assemble a full sklearn Pipeline: [preprocess (+ optional PCA)] -> estimator.

    Parameters
    ----------
    estimator : sklearn estimator
        Any regressor/estimator compatible with sklearn Pipeline.
    num_cols, cat_cols : list of str
        Feature column names.
    scale_numeric : bool, default False
        Standardize numeric features before modeling.
    use_pca : bool, default False
        If True, attach a PCA stage after preprocessing.
    pca_components : int or None
        Number of PCA components when use_pca=True.
    random_state : int, default 42
        Seed for PCA reproducibility (does not set estimator seed).
    sparse_ohe : bool, default True
        Whether OneHotEncoder should emit sparse output.
    sparse_threshold : float, default 1.0
        ColumnTransformer's sparse threshold.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    print(
        f"[PIPE] Building model pipeline | "
        f"scale_numeric={scale_numeric} use_pca={use_pca} pca_components={pca_components}"
    )

    pre = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        scale_numeric=scale_numeric,
        sparse_ohe=sparse_ohe,
        sparse_threshold=sparse_threshold,
    )

    pipe = attach_pca(
        pre,
        n_components=(pca_components if use_pca else None),
        random_state=random_state,
    )

    # If 'pipe' is just a ColumnTransformer, wrap it first
    if not isinstance(pipe, Pipeline):
        pipe = Pipeline([("pre", pipe)])

    # Append estimator
    steps = list(pipe.steps) + [("est", estimator)]
    print("[PIPE] Final pipeline steps:", [name for name, _ in steps])
    return Pipeline(steps)


# -------- Convenience: drop-in builder from a dataframe (optional) -------- #

def build_pipeline_from_df(
    df: pd.DataFrame,
    estimator,
    *,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scale_numeric: bool = False,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    random_state: int = 42,
    sparse_ohe: bool = True,
    sparse_threshold: float = 1.0,
) -> Pipeline:
    """
    Build a model pipeline directly from a DataFrame by inferring feature lists.

    If numeric_cols / categorical_cols are not provided, this function will
    try to infer them based on pandas dtypes (numeric = number types; categorical = others).

    NOTE:
    - Prefer explicitly passing column lists from src.features for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print(
        f"[PIPE-DF] Building pipeline from DataFrame | "
        f"inferred #num={len(numeric_cols)} #cat={len(categorical_cols)}"
    )

    return build_model_pipeline(
        estimator=estimator,
        num_cols=numeric_cols,
        cat_cols=categorical_cols,
        scale_numeric=scale_numeric,
        use_pca=use_pca,
        pca_components=pca_components,
        random_state=random_state,
        sparse_ohe=sparse_ohe,
        sparse_threshold=sparse_threshold,
    )
