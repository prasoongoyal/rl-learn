import numpy as np
import sys

def get_final_success_rate(filename, max_steps=500000.):
  for line in open(filename).readlines():
    line = line.strip()
    parts = line.split()
    if len(parts) != 11:
      continue
    n_timesteps = eval(parts[1][:-1])
    if n_timesteps > max_steps:
      break
  return eval(parts[5])

def compute_area(filename, max_steps=500000.):
  area = 0.
  x = [0]
  y = [0]
  for line in open(filename).readlines():
    line = line.strip()
    parts = line.split()
    if len(parts) != 11:
      continue
    x.append(eval(parts[1][:-1]))
    y.append(eval(parts[5]))
    area += (y[-1] + y[-2])/2.0 * (x[-1] - x[-2])
    if x[-1] >= max_steps:
      break
  return (area / max_steps)

if __name__ == "__main__":
  print (compute_area(sys.argv[1]))
