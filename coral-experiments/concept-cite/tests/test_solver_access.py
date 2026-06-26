import numpy as np
import importlib.util
from pathlib import Path


def _load_lasso_regression():
    pytest.importorskip("sklearn")
    solver_path = Path(__file__).resolve().parents[1] / "context_cite" / "solver.py"
    spec = importlib.util.spec_from_file_location("context_cite_solver_for_tests", solver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.LassoRegression


def test_lasso_regression_exposes_fitted_artifacts():
    LassoRegression = _load_lasso_regression()
    solver = LassoRegression(lasso_alpha=0.001)
    assert solver.is_fitted is False

    masks = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    outputs = np.array([0.8, 0.7, 0.5, 0.2, 1.0], dtype=np.float32)

    weight, bias = solver.fit(masks, outputs, num_output_tokens=1)

    assert weight.shape == (masks.shape[1],)
    assert np.isscalar(bias)
    assert solver.is_fitted is True
    assert solver.model_ is not None
    assert solver.scaler_ is not None
    assert solver.pipeline_ is not None


def test_lasso_regression_predict_matches_affine_form():
    LassoRegression = _load_lasso_regression()
    solver = LassoRegression(lasso_alpha=0.001)
    masks = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    outputs = np.array([0.2, 0.4, 0.6, 1.2], dtype=np.float32)
    weight, bias = solver.fit(masks, outputs, num_output_tokens=1)

    test_masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
    y_pipeline = solver.predict(test_masks)
    y_affine = test_masks.astype(np.float32) @ weight + bias

    np.testing.assert_allclose(y_pipeline, y_affine, rtol=1e-5, atol=1e-6)

    print(y_pipeline)
    print(y_affine)
    
if __name__ == "__main__":
    # test_lasso_regression_exposes_fitted_artifacts()
    test_lasso_regression_predict_matches_affine_form()