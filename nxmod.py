# Modified Networkx function to draw graphs with traits.

from matplotlib.path import Path
from networkx.drawing.layout import circular_layout
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _GetColors(n, cmap='gist_rainbow'):
  cm = plt.get_cmap(cmap)
  return [cm(float(i) / float(n)) for i in range(n)]


def _Draw(Gtraits, G, pos=None, ax=None, hold=None, **kwds):
  if ax is None:
    cf = plt.gcf()
  else:
    cf = ax.get_figure()
  cf.set_facecolor('w')
  if ax is None:
    if cf._axstack() is None:
      ax = cf.add_axes((0, 0, 1, 1))
    else:
      ax = cf.gca()

  #b = plt.ishold()
  h = kwds.pop('hold', None)
  #if h is not None:
    #plt.hold(h)
  try:
    #plt.hold(h)
    _DrawGraph(Gtraits, G, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
  except:
    #plt.hold(b)
    raise
  #plt.hold(b)
  return cf


def _DrawGraph(Gtraits, G, pos=None, trait_cmap='gist_rainbow', **kwds):
  if pos is None:
    pos = nx.drawing.spring_layout(G)
  num_traits = Gtraits.shape[1]
  trait_color = _GetColors(num_traits, trait_cmap)
  max_trait = np.max(np.sum(Gtraits, axis=0))
  for ti in range(num_traits):
    _DrawNodes(G, pos, node_size=Gtraits[:, ti] / float(max_trait), node_index=ti, node_color=trait_color[ti], **kwds)
  nx.draw_networkx_edges(G, pos, **kwds)
  plt.draw_if_interactive()


def _DrawNodes(G, pos, nodelist=None, node_size=300, node_index=0, node_color='r',
               node_shape='o', alpha=1.0, cmap=None, vmin=None, vmax=None, ax=None,
               width=None, label=None, **kwds):
  if ax is None:
    ax = plt.gca()
  if nodelist is None:
    nodelist = G.nodes()
  if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
    return None
  try:
    xy = np.asarray([pos[v] for v in nodelist])
  except KeyError as e:
    raise nx.NetworkXError('Node %s has no position.' % e)
  except ValueError:
    raise nx.NetworkXError('Bad value in node positions.')

  #ax.hold(True)
  for i in range(xy.shape[0]):
    x_pos = xy[i, 0]
    y_pos = xy[i, 1]
    size = node_size[i]
    offset = node_index + 2

    verts = []
    codes = []
    bar = size
    verts.append((x_pos, y_pos + float(offset) * 0.02))
    codes.append(Path.MOVETO)
    verts.append((x_pos+bar, y_pos + float(offset) * 0.02))
    codes.append(Path.LINETO)
    verts.append((x_pos+bar, y_pos + float(offset) * 0.02 - 0.02))
    codes.append(Path.LINETO)
    verts.append((x_pos, y_pos + float(offset) * 0.02 - 0.02))
    codes.append(Path.LINETO)

    verts.append((0, 0))
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    node_collection = patches.PathPatch(path, facecolor=node_color, lw=1)
    ax.add_patch(node_collection)

    if node_index == 0:
      degrees = (np.linspace(0.0, 360.0, 20) / 180.0 * np.pi).tolist()
      verts = []
      codes = []
      for j, d in enumerate(degrees):
        verts.append((x_pos + math.cos(d) * float(offset - 1) * 0.02,
                      y_pos + math.sin(d) * float(offset - 1) * 0.02))
        codes.append(Path.MOVETO if j == 0 else Path.LINETO)
      verts.append((0, 0))
      codes.append(Path.CLOSEPOLY)
      path = Path(verts, codes)
      node_collection = patches.PathPatch(path, facecolor='white', lw=1)
      ax.add_patch(node_collection)
  return node_collection


def DrawCircular(Gtraits, G, **kwargs):
    fig = _Draw(Gtraits, G, circular_layout(G), **kwargs)
    return fig
