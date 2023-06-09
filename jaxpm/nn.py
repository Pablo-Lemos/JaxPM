import jax
import jax.numpy as jnp

import haiku as hk
from jaxpm.painting import cic_read 

def _deBoorVectorized(x, t, c, p):
    """
    Evaluates S(x).

    Args
    ----
    x: position
    t: array of knot positions, needs to be padded as described above
    c: array of control points
    p: degree of B-spline
    """
    k = jnp.digitize(x, t) -1
    
    d = [c[j + k - p] for j in range(0, p+1)]
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (x - t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter(hk.Module):
  """A rotationally invariant filter parameterized by 
  a b-spline with parameters specified by a small NN."""

  def __init__(self, n_knots=8, latent_size=16, name=None):
    """
    n_knots: number of control points for the spline  
    """
    super().__init__(name=name)
    self.n_knots = n_knots
    self.latent_size = latent_size

  def __call__(self, x, a):
    """ 
    x: array, scale, normalized to fftfreq default
    a: scalar, scale factor
    """

    net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
    net = jnp.sin(hk.Linear(self.latent_size)(net))

    w = hk.Linear(self.n_knots+1)(net) 
    k = hk.Linear(self.n_knots-1)(net)
    
    # make sure the knots sum to 1 and are in the interval 0,1
    k = jnp.concatenate([jnp.zeros((1,)),
                        jnp.cumsum(jax.nn.softmax(k))])

    w = jnp.concatenate([jnp.zeros((1,)),
                         w])

    # Augment with repeating points
    ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

    return _deBoorVectorized(jnp.clip(x/jnp.sqrt(3), 0, 1-1e-4), ak, w, 3)

import haiku as hk
class CNN(hk.Module):
  def __init__(self, n_features: int,):
      super().__init__(name="CNN")
      self.conv1 = hk.Conv3D(
          output_channels=n_features, 
          kernel_shape=(3,3,3),  
          padding="SAME",
        )
      self.conv2 = hk.Conv3D(output_channels=n_features, kernel_shape=(3,3,3),  padding="SAME")
      self.conv3 = hk.Conv3D(output_channels=n_features, kernel_shape=(3,3,3),  padding="SAME")
      # Dense layers
      self.flatten = hk.Flatten()
      self.linear1 = hk.Linear(n_features)
      self.linear2 = hk.Linear(n_features)
      self.linear3 = hk.Linear(1)

  def __call__(self, x, positions, global_features):
    x = self.conv1(x)
    x = jax.nn.tanh(x)
    x = self.conv2(x)
    x = jax.nn.tanh(x)
    x = self.conv3(x)
    features = self.linear1(x)
    vmap_features_cic = jax.vmap(
      cic_read,
      in_axes = (-1,None),
    )
    vmap_batch_cic = jax.vmap(
      vmap_features_cic,
      in_axes=(0,0)
    )
    features_at_pos = vmap_batch_cic(features, positions).swapaxes(-2,-1)
    broadcast_globals = jnp.broadcast_to(
      global_features[:,:,None], (features_at_pos.shape[0], features_at_pos.shape[1], 1),
    )
    features_at_pos = jnp.concatenate([features_at_pos, broadcast_globals], axis=-1)
    features_at_pos = self.linear2(features_at_pos)
    features_at_pos = jax.nn.tanh(features_at_pos)
    features_at_pos = self.linear3(features_at_pos)
    return features_at_pos

def ConvNet(x, positions, global_features):
    cnn = CNN(n_features=8)
    return cnn(x, positions, global_features)