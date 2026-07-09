import matplotlib.pyplot as plt

file_path = "log/history.txt"
output_image_path = "log/training.png"

epochs = []
mse_values = []

try:
    with open(file_path, 'r') as file:
        for line in file:
            if "Epoch" in line and "MSE" in line:
                parts = line.strip().split(",")
                epoch = int(parts[0].split()[1])
                mse = float(parts[1].split()[1])
                epochs.append(epoch)
                mse_values.append(mse)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

plt.figure(figsize=(8, 4))
plt.plot(epochs, mse_values, marker='o', label="Loss (Mean Squared Error)")
plt.title("Training Loss", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(output_image_path, dpi=300)
print(f"Plot saved as {output_image_path}")

plt.show()