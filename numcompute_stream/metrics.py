"""
Classification metrics (accuracy, precision, recall, F1, confusion matrix) and MSE.
"""
import numpy as np

def accuracy(y_pred, y_true):
  """Fraction of correct predictions. NaN entries are dropped."""
  y_pred_mask = ~np.isnan(y_pred)
  y_pred = y_pred[y_pred_mask]
  y_true = y_true[y_pred_mask]
  y_true_mask = ~np.isnan(y_true)
  y_pred = y_pred[y_true_mask]
  y_true = y_true[y_true_mask]
  
  
  return np.divide(np.sum(np.where(y_pred ==y_true,1,0)),y_true.size)

def precision(y_pred, y_true, pred_class=1):
  """Precision of pred_class: correct predictions of it / all predictions of it."""
  y_pred_mask = ~np.isnan(y_pred)
  y_pred = y_pred[y_pred_mask]
  y_true = y_true[y_pred_mask]
  y_true_mask = ~np.isnan(y_true)
  y_pred = y_pred[y_true_mask]
  y_true = y_true[y_true_mask]
  pos_index = y_pred == pred_class
  return np.divide(np.sum(y_pred[pos_index] == y_true[pos_index]), y_pred[pos_index].size)

def recall(y_pred, y_true, pred_class=1):
  """Recall of pred_class: correct predictions of it / all true rows of it."""
  y_pred_mask = ~np.isnan(y_pred)
  y_pred = y_pred[y_pred_mask]
  y_true = y_true[y_pred_mask]
  y_true_mask = ~np.isnan(y_true)
  y_pred = y_pred[y_true_mask]
  y_true = y_true[y_true_mask]
  pos_index = y_true == pred_class
  return np.divide(np.sum(y_pred[pos_index] == y_true[pos_index]), y_true[pos_index].size)

def f1score(y_pred, y_true, pred_class=1):
  """F1 of pred_class: 2PR / (P + R)."""
  y_pred_mask = ~np.isnan(y_pred)
  y_pred = y_pred[y_pred_mask]
  y_true = y_true[y_pred_mask]
  y_true_mask = ~np.isnan(y_true)
  y_pred = y_pred[y_true_mask]
  y_true = y_true[y_true_mask]
  prec = precision(y_pred,y_true,pred_class)
  reca = recall(y_pred,y_true,pred_class)
  return 2* ((prec*reca)/(prec+reca))

def confusion_matrix(y_pred, y_true, pred_class=1):
  """(true_pos, false_pos, true_neg, false_neg) for pred_class."""
  y_pred_mask = ~np.isnan(y_pred)
  y_pred = y_pred[y_pred_mask]
  y_true = y_true[y_pred_mask]
  y_true_mask = ~np.isnan(y_true)
  y_pred = y_pred[y_true_mask]
  y_true = y_true[y_true_mask]
  pred_pos = y_pred == pred_class
  pred_neg = y_pred != pred_class
  true_pos = np.count_nonzero(y_pred[pred_pos] == y_true[pred_pos])
  true_neg = np.count_nonzero(y_pred[pred_neg] == y_true[pred_neg])
  false_pos = np.count_nonzero(y_pred[pred_pos] != y_true[pred_pos])
  flase_neg = np.count_nonzero(y_pred[pred_neg] != y_true[pred_neg])
  return (true_pos,false_pos,true_neg,flase_neg)

def MSE(y_pred, y_true):
  """Mean squared error."""
  y_pred_mask = ~np.isnan(y_pred)
  y_pred = y_pred[y_pred_mask]
  y_true = y_true[y_pred_mask]
  y_true_mask = ~np.isnan(y_true)
  y_pred = y_pred[y_true_mask]
  y_true = y_true[y_true_mask]
  return np.divide(np.sum(np.power(y_pred-y_true,2)),y_true.size)
