import openpyxl
import math
import numpy as np
import matplotlib.pyplot as graph

path = "D:/docs/personal/blood-pressure.xlsx"
sheet_names = ["2021", "2022", "2023"]

data_set = []

workbook = openpyxl.load_workbook(path)
for sheet_name in sheet_names:
    sheet = workbook[sheet_name]
    for row in sheet:
        for pulse_cell in [row[4], row[9]]:
            if pulse_cell.data_type == 'n' and pulse_cell.value:
                data_set.append(pulse_cell.value)

data_set.sort()


def distribution_range(values):
    return values[-1] - values[0]


def mean(values):
    return sum(values) / len(values)


def median(values):
    values_count = len(values)
    if values_count % 2 == 1:
        return values[values_count // 2]
    median_index = values_count // 2
    return (values[median_index] + values[median_index - 1]) / 2


def standard_deviation(values, mean_value=None):
    if mean_value is None:
        mean_value = sum(values) / len(values)

    variance = 0
    for value in values:
        variance += ((value - mean_value) ** 2)
    variance /= len(values)

    return math.sqrt(variance)


def percentile_range(values, percentage):
    values_sorted = sorted(values, reverse=False)
    num_values = len(values)
    num_to_keep = int(num_values * percentage)
    return values_sorted[:num_to_keep]


from scipy.stats import t


def mean_ci(values, confidence=0.8):
    n = len(values)
    mean = sum(values) / n
    s = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    df = n - 1
    t_crit = t.ppf((1 + confidence) / 2, df)
    margin_error = t_crit * s / math.sqrt(n)
    return (mean - margin_error, mean + margin_error)


def iqr(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    q2_index = n // 2
    q2 = median(sorted_values)  # assuming median() function has already been defined
    q1 = median(sorted_values[:q2_index])
    q3 = median(sorted_values[q2_index + (n % 2):])
    return q3 - q1


from scipy.stats import norm


def eighty_percent_ci(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z = norm.ppf(0.9)  # 90th percentile of standard normal distribution
    error = z * std / np.sqrt(n)
    lower_bound = mean - error
    upper_bound = mean + error
    return (lower_bound, upper_bound)


distribution_range = distribution_range(data_set)
mean = mean(data_set)
standard_deviation = standard_deviation(data_set, mean)

print("Heart rate measurements (bpm):")
print(f"- Total: {len(data_set)}")
print(f"- Distribution range: {distribution_range}: [{data_set[0]}:{data_set[-1]}]")
print(f"- Mean value (average): {mean}")
print(f"- Median: {median(data_set)}")
print(f"- Standard deviation: {standard_deviation}")
print(f"- Power range: [{int(mean - standard_deviation)}:{int(mean + standard_deviation)}]")
print(f"- 68% confidence interval: [{int(mean - standard_deviation)}:{int(mean + standard_deviation)}]")
print(f"- 95% confidence interval: [{int(mean - 2 * standard_deviation)}:{int(mean + 2 * standard_deviation)}]")
print(f"- 99.7% confidence interval: [{int(mean - 3 * standard_deviation)}:{int(mean + 3 * standard_deviation)}]")
print(f"- 80th percentile range: {percentile_range(data_set, 0.8)}")
print(f"- 80th confidence interval: {mean_ci(data_set, 0.99)}")
print(iqr(data_set))
print(eighty_percent_ci(data_set))

graph.hist(data_set, bins=40)
graph.show()
