"""Run a tiny, data-independent XGBoost training smoke check."""
import numpy as np
from xgboost import XGBClassifier

X = np.array([[i, i % 2, i % 5] for i in range(32)], dtype=np.float32)
y = np.array([0, 1] * 16, dtype=np.int32)
model = XGBClassifier(
    n_estimators=3,
    max_depth=2,
    learning_rate=0.1,
    tree_method='hist',
    eval_metric='logloss',
    n_jobs=1,
    random_state=0,
    verbosity=0,
)
model.fit(X, y)
probabilities = model.predict_proba(X)
assert probabilities.shape == (len(y), 2)
assert np.isfinite(probabilities).all()
print('XGBoost synthetic training smoke check passed.')
