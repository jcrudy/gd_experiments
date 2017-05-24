import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from numpy.ma.testutils import assert_array_almost_equal

# Create some data
m = 10000
X = np.random.normal(size=(m,10))
thresh = np.random.normal(size=10)
X_transformed = X * (X > thresh)
beta = np.random.normal(size=10)
y = (np.dot(X_transformed, beta) + np.random.normal(size=m)) > 0

# Train a gradient boosting classifier
model = GradientBoostingClassifier()
model.fit(X, y)
print model.score(X, y)

# Inspect
pred = model.predict_proba(X)

approx = model.loss_._score_to_proba(model.learning_rate * sum(map(lambda est: est.predict(X), model.estimators_[:,0])) + np.ravel(model.init_.predict(X)))

assert_array_almost_equal(pred, approx)
