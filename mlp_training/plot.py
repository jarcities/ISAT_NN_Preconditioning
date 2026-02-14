import matplotlib.pyplot as plt
import sys

# files
filename = 'out2'

# read files
try:
    with open(filename, 'r') as f:
        losses = [float(line.strip()) for line in f if line.strip()]
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except ValueError as e:
    print(f"Error parsing file: {e}")
    sys.exit(1)

# y axis
iterations = list(range(len(losses)))

# plot results
plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, marker='o', linestyle='-', color='b', markersize=2)
plt.xlabel('iters')
plt.ylabel('loss')
plt.title(f'loss vs iters {filename}')
plt.grid(True)
plt.tight_layout()
plt.show()