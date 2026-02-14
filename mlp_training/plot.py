import matplotlib.pyplot as plt
import sys

# files
filename_cpu = 'out2_cpu'
filename_gpu = 'out2_gpu'
filename_none = 'out2_none'
files = [filename_cpu, filename_gpu, filename_none]

colors = ['b', 'r', 'g']

plt.figure(figsize=(10, 6))

# read files
for f in range(3):
    filename = files[f]
    try:
        with open(filename, 'r') as file:
            losses = [float(line.strip()) for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    # y axis
    iterations = list(range(len(losses)))

    # plot results
    plt.plot(iterations, losses, marker='o', linestyle='-', color=colors[f], markersize=2, label=filename)

plt.xlabel('iters')
plt.ylabel('loss')
plt.title('loss vs iters')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()