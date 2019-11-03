
class Fitter():
  """Helper class for training a simple pytorch network.

  Args:
    network: PyTorch network
    loss_fn: Loss function
    optimizer: Optimizer object
    acc_fn: Accuracy function. By default, simple "ratio of right" accuracy.

  Methods (See method docstrings for more info):
    fit: Trains a network.
    predict: Runs the network
    evaluate: Runs the network and uses the `acc_fn` to compute the performance.

  Example:

    loss_fn = BCELoss()
    optimizer = RMSprop()
    acc_fn = lambda y_hat, y: (y_hat.argmax(0) == y).sum().item() / len(y_hat)

    fitter = Fitter(network, loss_fn=loss_fn, optimizer=optimizer, acc_fn=acc_fn)
    fitter.fit(x_train, y_train, epochs=25)
    test_accuracy = fitter.evaluate(x_test, y_test)
  """
  def __init__(self, network, loss_fn, optimizer, acc_fn=None):
    self.net = network
    self._input_shape = None
    self.loss_fn = loss_fn
    self.opt = optimizer
    self.acc_fn = acc_fn
    if self.acc_fn is None:
      self.acc_fn = lambda y_hat, y: \
                      (y_hat.argmax(1) == y).sum().item() / len(y_hat)

  def fit(self, x, y, epochs=1, batch_size=None, x_val=None, y_val=None,
          quiet=False):
    """Trains the PyTorch network.

    Args:
      x, y: Training predictors and targets.
      x_val, y_val: Validation predictors and targets.
      epochs: Number of epochs to train for.
      quiet: Don't print information while training
    Returns:
      history: Training history as a dict with the following keys:
          epochs, loss, accuracy, loss_val, accuracy_val
        The keys store appropriate variable history.
    """
    if batch_size is None or batch_size == len(x):
      batches = 1
    else:
      batches = len(x) // batch_size
      if batches * batch_size < len(x):
        batches += 1

    if x_val is None:
      batches_val = 0
    elif batch_size is None or batch_size == len(x_val):
      batches_val = 1
    else:
      batches_val = len(x_val) // batch_size
      if batches_val * batch_size < len(x):
        batches_val += 1

    history = {
      "loss": [],
      "accuracy": [],
      "loss_val": [],
      "accuracy_val": [],
      "epochs": []
    }

    # Save the input shape, just in case. Assume, first element in sample num.
    self._input_shape = (-1,) + x.shape[1:]

    ## This might be broken outside the notebook
    for epoch in range(1, epochs+1):
      epoch_loss = 0.0
      epoch_acc = 0.0
      for batch in range(batches):
        self.opt.zero_grad()

        batch_begin = batch * batch_size
        batch_end = batch_begin + batch_size
        batch_x = x[batch_begin:batch_end]
        batch_y = y[batch_begin:batch_end]

        y_hat = self.predict(batch_x, autoreshape=False)

        loss = self.loss_fn(y_hat, batch_y)
        loss.backward()
        self.opt.step()

        epoch_loss += loss.item()
        epoch_acc += self.acc_fn(y_hat, batch_y)

      epoch_loss /= batches
      epoch_acc /= batches
      history["loss"].append(epoch_loss)
      history["accuracy"].append(epoch_acc)

      if not quiet:
        print("{}/{}".format(epoch, epochs), end="")
        print(" | Train loss: {:.4f}, acc: {:.4f}"
          .format(epoch_loss, epoch_acc), end="")

      if batches_val > 0:
        epoch_loss_val = 0.0
        epoch_acc_val = 0.0
        for batch in range(batches_val):
          batch_begin = batch * batch_size
          batch_end = batch_begin + batch_size
          batch_x = x_val[batch_begin:batch_end]
          batch_y = y_val[batch_begin:batch_end]

          y_hat = self.predict(batch_x, autoreshape=False)

          loss = self.loss_fn(y_hat, batch_y)

          epoch_loss_val += loss.item()
          epoch_acc_val += self.acc_fn(y_hat, batch_y)

        epoch_loss_val /= batches_val
        epoch_acc_val /= batches_val
        history["loss_val"].append(epoch_loss_val)
        history["accuracy_val"].append(epoch_acc_val)
        if not quiet:
          print(" | Val loss: {:.4f}, acc: {:.4f}"
                  .format(epoch_loss_val, epoch_acc_val), end="")
      if quiet:
        char = "."
        if epoch % 5 == 0:
          char = "o"
        if epoch % 10 == 0:
          char = "|"
        print(char, end="")
      else:
        print("")
      history["epochs"].append(epoch)
    return history

  def predict(self, x, autoreshape=True):
    """Predicts the x.

    Args:
      x: Input tensor of shape (N, ...)
      autoreshape: Automatically reshape the input into the size used during
                   training. This will reshape into (-1, ...).

    Returns:
      y: Predicted tensor
    """
    if autoreshape and self._input_shape is not None:
      x = x.reshape(self._input_shape)
    return self.net(x)

  def evaluate(self, x, y, more_fn=None):
    """Evaluates the statistical performance of the model.

    The evaluation is done using the same loss function that was used during the
    training. The accuracy is simple classification accuracy.

    Args:
      y_hat: Predicted values
      y: True values
      more_fn: List of other functions that need to be evaluated on y_hat and y.
        The functions should take only 2 parameters: func(y_hat, y), and all
        the returns will be put in a list
    Returns:
      - accuracy of the model
      - loss of the model
      - list of lists with the results of the `more_fn`
    """
    y_hat = self.predict(x)
    accuracy = self.acc_fn(y_hat, y)
    loss = self.loss_fn(y_hat, y)
    fns = []
    if more_fn is not None:
      for fn in more_fn:
        fns.append(fn(y_hat, y))
    return accuracy, loss, fns
