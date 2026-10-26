"""The Lasso regression that LSM fits on the concepts each ablated prompt kept

The prediction for a node says how likely the response is when only that node's concepts are kept
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class LassoSurrogate:
    """Predicts the response's logit probability log(p / (1 - p)) from the kept concepts

    Targets are relative to the prompt with no concepts and there is no intercept
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Store the regularization strength

        Args:
            alpha: The Lasso penalty
        """
        self.alpha = alpha
        self.pipeline: Pipeline | None = None

    def fit(self, masks: NDArray[np.bool_], outputs: NDArray, token_count: int) -> None:
        """Fit the regression on the ablations

        Outputs are divided by the token count so the weights do not depend on how long the response
        is

        Args:
            masks: One boolean row per ablation, True for each concept it kept
            outputs: The response's logit probability under each mask
            token_count: The number of tokens in the response
        """
        self.pipeline = make_pipeline(
            StandardScaler(), Lasso(alpha=self.alpha, random_state=0, fit_intercept=False)
        )
        self.pipeline.fit(masks.astype(np.float32), outputs / token_count)

    def predict(self, masks: NDArray[np.bool_]) -> NDArray:
        """Predict the per-token logit probability for each mask

        Calling this before fit raises a RuntimeError

        Args:
            masks: One boolean row per node, True for each concept it keeps

        Returns:
            One prediction per mask
        """
        if self.pipeline is None:
            raise RuntimeError('the surrogate must be fitted before it predicts')
        return self.pipeline.predict(masks.astype(np.float32))
