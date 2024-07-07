import csv
import math

def write_sine_function_to_csv(filename, start, end, step):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])

        x = start
        while x <= end:
            y = math.sin(x)
            writer.writerow([x, y])
            x += step

# Example usage
filename = 'sine_function.csv'
start = 0
end = 2 * math.pi
step = 0.1

write_sine_function_to_csv(filename, start, end, step)
print(f"Sine function written to {filename} successfully.")