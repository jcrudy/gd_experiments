import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from numpy.ma.testutils import assert_array_almost_equal
from sklearn.tree.tree import DecisionTreeRegressor
from sympy.core.numbers import RealNumber
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.symbol import Symbol
import pandas
from nose.tools import assert_almost_equal

# Create some data
m = 10000
X = np.random.normal(size=(m,10))
thresh = np.random.normal(size=10)
X_transformed = X * (X > thresh)
beta = np.random.normal(size=10)
y = np.dot(X_transformed, beta) + np.random.normal(size=m)

# Train a decision tree regressor
model = DecisionTreeRegressor()
model.fit(X, y)
print model.score(X, y)

# Inspect
def _sym_predict_decision_tree(model, names, current_node=0, output_idx=0, class_idx=0):
    left = model.tree_.children_left[current_node]
    right = model.tree_.children_right[current_node]
    if left == -1:
        assert right == -1
        left_expr = RealNumber(model.tree_.value[current_node, output_idx, class_idx])
        right_expr = left_expr
    else:
        left_expr = _sym_predict_decision_tree(model, names, current_node=left, output_idx=output_idx, class_idx=class_idx)
        right_expr = _sym_predict_decision_tree(model, names, current_node=right, output_idx=output_idx, class_idx=class_idx)
    return Piecewise((left_expr, Symbol(names[model.tree_.feature[current_node]]) <= model.tree_.threshold[current_node]),
                     (right_expr, Symbol(names[model.tree_.feature[current_node]]) > model.tree_.threshold[current_node]),
                     )


y_pred = model.predict(X)
X_ = pandas.DataFrame(X, columns=map(lambda i: 'x' + str(i), range(10)))
names = list(X_.columns)
expr = _sym_predict_decision_tree(model, names)
for i in range(10):
    row = dict(X_.loc[i,:])
    assert_almost_equal(y_pred[i], expr.evalf(16, row))
print expr

