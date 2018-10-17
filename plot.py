import io
import numpy as np
import matplotlib.pyplot as plt

def plot_embedding(x, y, d, title=None, filename=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    From: https://github.com/pumpikano/tf-dann/blob/master/utils.py
    """
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    # We'd get an error if nan or inf
    if np.isnan(x).any() or np.isinf(x).any():
        return None

    # XKCD colors: https://matplotlib.org/users/colors.html
    colors = {
        0: 'xkcd:orange', # source
        1: 'xkcd:darkgreen', # target
    }

    domain = {
        0: 'S',
        1: 'T',
    }

    # Plot colors numbers
    fig = plt.figure(figsize=(10,10))
    plt.subplot(111)
    for i in range(x.shape[0]):
        # plot colored number
        plt.text(x[i, 0], x[i, 1], domain[d[i]]+str(y[i]),
                 color=colors[d[i]],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    # Save to buffer for sending to TensorBoard
    # See: https://stackoverflow.com/a/38676842/2698494
    buf = io.BytesIO()
    plt.savefig(buf, format='png',
        bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)

    # Make sure we don't keep these plots in memory forever. Otherwise we get:
    #
    # "More than 20 figures have been opened. Figures created through the pyplot
    # interface (`matplotlib.pyplot.figure`) are retained until explicitly closed
    # and may consume too much memory."
    plt.close(fig)

    return buf.getvalue()

def plot_random_time_series(mu, sigma, num_samples=5, title=None, filename=None):
    """
    Using the mu and sigma given at each time step, generate sample time-series
    using these Gaussian parameters learned by the VRNN

    Input:
        mu, sigma -- each time step, learned in VRNN,
            each shape: [batch_size, time_steps, num_features]
        num_samples -- how many lines/curves you want to plot
        title, filename -- optional
    Output:
        plot of sample time-series

    Note: at the moment we're assuming num_features=1 (plot will be 2D)
    """
    mu = np.squeeze(mu)
    sigma = np.squeeze(sigma)
    length = mu.shape[1]

    # Take only desired number of time-series
    num_samples = min(num_samples, mu.shape[0])
    mu = mu[:num_samples,:]
    sigma = sigma[:num_samples,:]

    # x axis is just 0, 1, 2, 3, ...
    x = np.arange(length)

    # y is values sampled from mu and simga
    y = sigma*np.random.normal(0, 1, (num_samples, length)) + mu

    fig = plt.figure()
    for i in range(y.shape[0]):
        plt.plot(x, y[i,:])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    # Save to buffer for sending to TensorBoard
    # See: https://stackoverflow.com/a/38676842/2698494
    buf = io.BytesIO()
    plt.savefig(buf, format='png',
        bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)

    plt.close(fig)

    return buf.getvalue()

def plot_real_time_series(y, num_samples=5, title=None, filename=None):
    """
    Plot the real time-series data for comparison with the reconstruction

    Input:
        input x values (real data)
            shape: [batch_size, time_steps, num_features]
        num_samples -- how many lines/curves you want to plot
        title, filename -- optional
    Output:
        plot of input time-series

    Note: at the moment we're assuming num_features=1 (plot will be 2D)
    """
    y = np.squeeze(y)
    length = y.shape[1]

    # Take only desired number of time-series
    num_samples = min(num_samples, y.shape[0])
    y = y[:num_samples,:]

    # x axis is just 0, 1, 2, 3, ...
    x = np.arange(length)

    fig = plt.figure()
    for i in range(y.shape[0]):
        plt.plot(x, y[i,:])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    # Save to buffer for sending to TensorBoard
    # See: https://stackoverflow.com/a/38676842/2698494
    buf = io.BytesIO()
    plt.savefig(buf, format='png',
        bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)

    plt.close(fig)

    return buf.getvalue()
