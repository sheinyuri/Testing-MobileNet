from collections import defaultdict
import matplotlib.pyplot as plt

def results_plotter(path, name):
    data_to_plot = defaultdict(list)
    with open(path, "r") as f:
        f.readline()

        for line in f:
            data = line.strip().split(",")
            data_to_plot[data[0]].append((int(data[1]), float(data[2])))

    print(data_to_plot)

    plt.figure(figsize=(10, 6))

    for key, coordinates in data_to_plot.items():
        x_vals, y_vals = zip(*coordinates)
        plt.plot(x_vals, y_vals, marker='o', label=key)

    plt.title('Combined Trend Plot', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('f{name}', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend()
    plt.savefig(f'plots/{name}.png', dpi=300, bbox_inches='tight')
    plt.show()    