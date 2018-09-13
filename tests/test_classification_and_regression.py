import unittest
from pyBreakDown.explainer import Explainer
from sklearn import model_selection
import numpy as np
from sklearn import tree
from sklearn import linear_model


class Testbreakdown(unittest.TestCase):
    def setUp(self):
        np.random.seed(11231231)  # Must not be changed as test 2 often returns a 0 prediction which breaks percent_diff
        data = np.random.rand(10, 4)
        target = np.reshape([1, 0, 1, 0, 1, 0, 1, 1, 1, 0], (10, 1))
        self.train, self.test, self.labels_train, self.labels_test = model_selection.train_test_split(
            data, target, train_size=0.80)

    def test_explainer_returns_sensible_output_for_classifier_up(self):
        #  arrange
        dtree = tree.DecisionTreeClassifier()
        dtree.fit(self.train, self.labels_train)
        expected_proba = 1.625

        #  act
        bd_exp = Explainer(clf=dtree, data=self.train, colnames=['foo', 'bar', 'baz', 'qux'])
        explanation = bd_exp.explain(self.test[0], direction="up", useIntercept=False, baseline=0)
        percentage_diff = abs(explanation._attributes[-1].cumulative / expected_proba - 1)

        #  assert
        self.assertLessEqual(percentage_diff, 0.2)
        self.assertEqual(len(explanation._attributes)-1, np.shape(self.train)[1])

    def test_explainer_returns_sensible_output_for_classifier_down(self):
        #  arrange
        dtree = tree.DecisionTreeClassifier()
        dtree.fit(self.train, self.labels_train)
        expected_proba = 1

        #  act
        bd_exp = Explainer(clf=dtree, data=self.train, colnames=['foo', 'bar', 'baz', 'qux'])
        explanation = bd_exp.explain(self.test[0], direction="down", useIntercept=False, baseline=0)
        percentage_diff = abs(explanation._attributes[-1].cumulative / expected_proba - 1)

        #  assert
        self.assertLessEqual(percentage_diff, 0.2)
        self.assertEqual(len(explanation._attributes)-1, np.shape(self.train)[1])

    def test_explainer_returns_sensible_output_for_regressor_up(self):
        #  arrange
        lregressor = linear_model.LinearRegression()
        lregressor.fit(self.train, self.labels_train)
        expected_val = - 1.6407540804833103

        #  act
        bd_exp = Explainer(clf=lregressor, data=self.train, colnames=['foo', 'bar', 'baz', 'qux'])
        explanation = bd_exp.explain(self.test[0], direction="up", useIntercept=False, baseline=0)
        percentage_diff = abs(explanation._attributes[-1].cumulative / expected_val -1)

        #  assert
        self.assertLessEqual(percentage_diff, 0.2)
        self.assertEqual(len(explanation._attributes)-1, np.shape(self.train)[1])

    def test_explainer_returns_sensible_output_for_regressor_down(self):
        #  arrange
        lregressor = linear_model.LinearRegression()
        lregressor.fit(self.train, self.labels_train)
        expected_val = - 0.576140993143524

        #  act
        bd_exp = Explainer(clf=lregressor, data=self.train, colnames=['foo', 'bar', 'baz', 'qux'])
        explanation = bd_exp.explain(self.test[0], direction="down", useIntercept=False, baseline=0)
        percentage_diff = abs(explanation._attributes[-1].cumulative / expected_val -1)

        #  assert
        self.assertLessEqual(percentage_diff, 0.2)
        self.assertEqual(len(explanation._attributes)-1, np.shape(self.train)[1])
