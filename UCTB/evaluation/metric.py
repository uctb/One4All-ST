import numpy as np

def rmse(prediction, target, threshold=None):
    """
    Root Mean Square Error (RMSE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target will be removed in computing the rmse
    """
    if threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]


def mape(prediction, target, threshold=0):
    """
    Mean Absolute Percentage Error (MAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the mape.
    """
    assert threshold > 0
    return (np.dot((np.abs(prediction - target) / (target + (1 - (target > threshold)))).reshape([1, -1]),
                   target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]


def mae(prediction, target, threshold=None):
    """
    Mean Absolute Error (MAE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target will be removed in computing the mae
    """
    if threshold is None:
        return np.mean(np.abs(prediction - target))
    else:
        return (np.dot(np.abs(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0, 0]

def smape(prediction, target, threshold=0):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller than threshold in target will be removed in computing the smape.
    """
    prediction[prediction<=threshold] = threshold
    target[target<=threshold] = threshold
    
    return np.mean(np.abs(prediction - target) / ((np.abs(prediction) + np.abs(target))*0.5))

def traffic_targeted_rmse(prediction, target, lower_threshold=None,higher_threshold=None,selected_station=None):
    """
    Root Mean Square Error (RMSE)
    
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target will be removed in computing the rmse
    """
    if not selected_station is None:
        return np.sqrt(np.mean(np.square(prediction[selected_station]-target[selected_station])))
    if lower_threshold is None and higher_threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
    elif higher_threshold is None:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > lower_threshold) / np.sum(target > lower_threshold))[0][0]
    elif lower_threshold is None:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) <= higher_threshold) / np.sum(target <= higher_threshold))[0][0]
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              (target.reshape([-1, 1]) <= higher_threshold)&(target.reshape([-1, 1]) > lower_threshold)) / np.sum((target <= higher_threshold)&(target > lower_threshold)))[0][0]
