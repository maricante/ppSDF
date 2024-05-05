import torch


def mse(yhat,y):
    """
    Calculate the mean squared error.
    args:
        yhat: predicted values
        y: ground truth values
    """
    return torch.nn.MSELoss(reduction='mean')(yhat,y)

def rmse(yhat,y):
    """
    Calculate the root mean squared error.
    args:
        yhat: predicted values
        y: ground truth values
    """
    return torch.sqrt(mse(yhat,y))

def collision_cost(yhat,y):
    """
    Calculate the collision cost.
    args:
        yhat: predicted values
        y: ground truth values
    """
    epsilon = 0.03
    chat = torch.zeros_like(yhat)
    chat[yhat <= epsilon] = 1/(2 * epsilon) * (yhat[yhat <= epsilon] - epsilon)**2
    chat[yhat < 0] = -yhat[yhat < 0] + epsilon/2
    c = torch.zeros_like(y)
    c[y <= epsilon] = 1/(2 * epsilon) * (y[y <= epsilon] - epsilon)**2
    c[y < 0] = -y[y < 0] + epsilon/2
    return torch.abs(chat - c).mean()

def gradient_cosine_distance(dyhat,dy):
    """
    Calculate the gradient cosine distance.
    args:
        dyhat: predicted gradients
        dy: ground truth gradients
    """
    dyhat = dyhat / torch.norm(dyhat,dim=-1,keepdim=True)
    dy = dy / torch.norm(dy,dim=-1,keepdim=True)
    return 1 - (dyhat*dy).sum(dim=-1).mean()

def print_eval(yhat,y, dyhat=None, dy=None, string='default'):
    """
    Print evaluation metrics.
    args:
        yhat: predicted values
        y: ground truth values
        dyhat: predicted gradients
        dy: ground truth gradients
        string: string to print
    """
    yhat,y = yhat.reshape(-1).abs(),y.reshape(-1).abs()
    dy = dy.reshape(-1,3)
    dyhat = dyhat.reshape(-1,3)

    MAE = (yhat-y).abs().mean()
    MSE = mse(yhat,y)
    RMSE = rmse(yhat,y)
    collision_cost_ = collision_cost(yhat,y)
    gcd = gradient_cosine_distance(dyhat,dy)

    print(f'{string}\n'
          f'abs:{MAE:.6f}\t'
          f'mse:{MSE:.6f}\t'
          f'rmse:{RMSE:.6f}\t'
          f'collision_cost:{collision_cost_:.6f}\t'
          f'gcd:{gcd:.6f}')
