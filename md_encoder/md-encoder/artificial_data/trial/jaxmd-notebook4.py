#@title Imports and Definitions
#!pip install jax-md
!pip install -q git+https://www.github.com/google/jax-md

import numpy as onp

import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)

from jax import random
from jax import jit, lax, grad, vmap, hessian
import jax.scipy as jsp

from jax_md import space, energy, smap, simulate, minimize, util, elasticity, quantity, partition
from jax_md.colab_tools import renderer

f32 = jnp.float32
f64 = jnp.float64

from functools import partial

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def format_plot(x, y):  
  plt.grid(True)
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)
  
def finalize_plot(shape=(1, 0.7)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])

def prepare_trajectory(trajectory, expand_axis = None, color_scheme = 'Set1'):
  """(t,n,N,d) -> (t,n*N,d) plus colors
  """
  if expand_axis is not None:
    trajectory = jnp.expand_dims(trajectory, expand_axis)
  
  s = trajectory.shape
  assert len(s) == 4
  new_trajectory = trajectory.reshape((s[0], s[1]*s[2], s[3]))

  cm = plt.get_cmap(color_scheme)
  cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=s[1])
  scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
  colors = scalarMap.to_rgba(jnp.arange(s[1]*s[2]) // s[2])

  return new_trajectory, colors[:,:3]

def show_trajectory(trajectory, 
                    expand_axis = None, 
                    color_scheme = 'Set1',
                    diameters = 1):
  traj, colors = prepare_trajectory(trajectory, expand_axis, color_scheme)

  renderer.render(box_size,
                  {'particles': renderer.Disk(traj, diameters, color=colors)},
                  resolution=(512, 512))