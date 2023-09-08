import os

file_name = "fastwaves.mlir"

with open(file_name, "r") as f:
    lines = f.read().split("\n")

for line in lines:
    # line = l.strip()
    if "stencil.apply" in line and "attributes" not in line:
        apply_index = line.split('=')[0].strip()[1:]
        line = line[:-1] + f"attributes {{name=\"apply_{apply_index}\"}}" + "{"
        print(line)
    else:
        print(line)
