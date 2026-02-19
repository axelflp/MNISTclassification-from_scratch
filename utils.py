import numpy as np
from sklearn.metrics import classification_report
def metrics_mlp(model, x, y):
    return [classification_report(y, model.eval(x)), classification_report(y, model.eval(x), output_dict=True)]