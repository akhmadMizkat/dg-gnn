import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc
)

try:
    import torch
except ImportError:
    torch = None


### Evaluator for node property prediction
class Evaluator:
    def __init__(self, eval_metric):
        if eval_metric not in ['acc', 'auc', 'pr_auc']:
            raise ValueError(
                'eval_metric should be acc, auc, or pr_auc'
            )

        self.eval_metric = eval_metric

    def _check_input(self, y_true, y_pred):
        '''
            y_true: numpy ndarray or torch tensor of shape (num_node)
            y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
        '''

        # converting to numpy on CPU
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # check type
        if not (
            isinstance(y_true, np.ndarray)
            and isinstance(y_pred, np.ndarray)
        ):
            raise RuntimeError(
                'Arguments to Evaluator need to be either '
                'numpy ndarray or torch tensor'
            )

        if y_pred.ndim != 2:
            raise RuntimeError(
                'y_pred must be 2-dim array, '
                '{}-dim array given'.format(y_pred.ndim)
            )

        return y_true, y_pred

    def eval(self, y_true, y_pred):
        y_true, y_pred = self._check_input(y_true, y_pred)

        if self.eval_metric == 'auc':
            return self._eval_rocauc(y_true, y_pred)

        if self.eval_metric == 'pr_auc':
            return self._eval_pr_auc(y_true, y_pred)

        if self.eval_metric == 'acc':
            return self._eval_acc(y_true, y_pred)

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC
        '''

        if y_pred.shape[1] == 1:
            auc_score = roc_auc_score(y_true, y_pred[:, 0])

        elif y_pred.shape[1] == 2:
            auc_score = roc_auc_score(y_true, y_pred[:, 1])

        else:
            onehot_code = np.eye(y_pred.shape[1])
            y_true_onehot = onehot_code[y_true]

            auc_score = roc_auc_score(
                y_true_onehot,
                y_pred
            )

        return {'auc': auc_score}

    def _eval_pr_auc(self, y_true, y_pred):
        '''
            compute PR-AUC and Average Precision (AP)
        '''

        if y_pred.shape[1] == 1:
            y_score = y_pred[:, 0]

        elif y_pred.shape[1] == 2:
            # Model output is log-probability from log_softmax
            # Convert fraud-class log-probability to probability
            y_score = np.exp(y_pred[:, 1])

        else:
            raise ValueError(
                'PR-AUC evaluation is currently supported '
                'for binary classification only.'
            )

        # Average Precision
        ap = average_precision_score(
            y_true,
            y_score
        )

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score
        )

        # Trapezoidal PR-AUC
        pr_auc = auc(
            recall,
            precision
        )

        return {
            'pr_auc': pr_auc,
            'ap': ap
        }

    def _eval_acc(self, y_true, y_pred):
        y_pred = y_pred.argmax(axis=-1)

        correct = y_true == y_pred

        acc = float(np.sum(correct)) / len(correct)

        return {'acc': acc}