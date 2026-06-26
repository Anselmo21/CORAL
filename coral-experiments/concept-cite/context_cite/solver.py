import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline


class BaseSolver(ABC):
    """
    A base solver class.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    @abstractmethod
    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]: ...


class LassoRegression(BaseSolver):
    """
    A LASSO solver using the scikit-learn library.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.
        fit_intercept (bool):
            If True (default), fit an intercept. If False, the regression line
            is constrained through the origin in standardized feature space; use
            with targets already centered on a reference (e.g. empty or full
            criteria baseline).

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, lasso_alpha: float = 0.01, fit_intercept: bool = True) -> None:
        self.lasso_alpha = lasso_alpha
        self.fit_intercept = fit_intercept
        self.scaler_: StandardScaler | None = None
        self.model_: Lasso | None = None
        self.pipeline_ = None

    def fit(
        self, masks: NDArray, outputs: NDArray, num_output_tokens: int
    ) -> Tuple[NDArray, NDArray]:
        X = masks.astype(np.float32)
        Y = outputs / num_output_tokens
        scaler = StandardScaler()
        lasso = Lasso(
            alpha=self.lasso_alpha, random_state=0, fit_intercept=self.fit_intercept
        )
        # Pipeline is ((X - scaler.mean_) / scaler.scale_) @ lasso.coef_.T + lasso.intercept_
        pipeline = make_pipeline(scaler, lasso)
        pipeline.fit(X, Y)
        self.scaler_ = scaler
        self.model_ = lasso
        self.pipeline_ = pipeline
        # Rescale back to original scale
        weight = lasso.coef_ / scaler.scale_
        if self.fit_intercept:
            bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
        else:
            bias = 0.0
        return weight * num_output_tokens, bias * num_output_tokens

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    def predict(self, masks: NDArray) -> NDArray:
        if self.pipeline_ is None:
            raise RuntimeError("LassoRegression is not fitted yet. Run fit() first.")
        X = masks.astype(np.float32)
        return self.pipeline_.predict(X)
