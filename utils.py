import os


RED = 31
GREEN = 32
BLUE = 35


# Little helper to make nice looking terminal prints.
def Highlight(string, color=None, bold=False):
  attr = []
  if color is not None:
    attr.append('%d' % color)
  if bold:
    attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def GetTerminalWidth():
  _, columns = os.popen('stty size', 'r').read().split()
  return int(columns)
