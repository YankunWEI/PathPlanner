import scipy.stats as stats
import numpy as np
coverage_1 = stats.chi2.cdf(1, 1)
coverage_2 = stats.chi2.cdf(4, 1)
print(coverage_1)
print(coverage_2)
def calculate_radius(dimensions, variance, coverage=coverage_2):
    # 计算卡方分布的临界值
    chi_square_value = stats.chi2.ppf(coverage, dimensions)
    
    # 计算标准高斯分布的半径
    radius = (chi_square_value * variance) ** 0.5
    return radius

def calculate_variance(dimensions, radius, coverage = stats.chi2.cdf(4, 1)):
    # 计算给定覆盖率的卡方值
    chi_square_value = stats.chi2.ppf(coverage, dimensions)
    
    # 计算方差
    variance = (radius ** 2) / chi_square_value
    return variance

print(np.log(0.5))

print(-(0.5 * np.log(0.5) + 0.5 * np.log(0.5)))