# Day 8 – Hypothesis Testing

## What is Hypothesis Testing and Why is it Important for Deep Learning?

Hypothesis testing is a statistical method used to make decisions or inferences about population parameters based on sample data. It provides a formal framework for determining whether the observed effects in data are statistically significant or could have occurred by chance.

In deep learning, hypothesis testing is important because:

- It helps validate whether model improvements are statistically significant
- It allows us to compare different models or algorithms objectively
- It provides a framework for A/B testing in production systems
- It helps in feature selection and importance evaluation
- It supports the interpretation of experimental results

## Essential Subtopics

### 1. Basic Concepts of Hypothesis Testing

**What are the fundamental concepts of hypothesis testing?**
- **Null Hypothesis (H₀)**: The default assumption that there is no effect or no difference
- **Alternative Hypothesis (H₁ or Hₐ)**: The claim that contradicts the null hypothesis
- **Test Statistic**: A value calculated from sample data used to decide whether to reject H₀
- **Significance Level (α)**: The probability of rejecting H₀ when it is true (Type I error)
- **Critical Region**: The range of values that leads to rejection of H₀
- **p-value**: The probability of obtaining test results at least as extreme as the observed results, assuming H₀ is true
- **Type I Error**: Rejecting H₀ when it is true (false positive)
- **Type II Error**: Failing to reject H₀ when H₁ is true (false negative)
- **Power**: The probability of correctly rejecting H₀ when H₁ is true (1 - Type II error rate)

**PyTorch Implementation:**
```python
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats

# Basic hypothesis testing example
# Suppose we want to test if a coin is fair (p = 0.5)
# H₀: p = 0.5 (coin is fair)
# H₁: p ≠ 0.5 (coin is biased)

# Simulate coin flips
def simulate_coin_flips(num_flips, true_p, num_simulations=10000):
    """Simulate coin flips and calculate test statistics"""
    # Generate random coin flips
    flips = torch.rand(num_simulations, num_flips) < true_p
    
    # Count number of heads
    num_heads = torch.sum(flips, dim=1)
    
    # Calculate test statistic (proportion of heads)
    test_stats = num_heads / num_flips
    
    return test_stats

# Parameters
num_flips = 100
true_p = 0.5  # Assume coin is fair
num_simulations = 10000

# Simulate under H₀ (coin is fair)
h0_test_stats = simulate_coin_flips(num_flips, true_p, num_simulations)

# Simulate under H₁ (coin is biased, say p = 0.6)
h1_test_stats = simulate_coin_flips(num_flips, 0.6, num_simulations)

# Plot distributions
plt.figure(figsize=(12, 6))
plt.hist(h0_test_stats.numpy(), bins=30, alpha=0.5, label='H₀: p = 0.5')
plt.hist(h1_test_stats.numpy(), bins=30, alpha=0.5, label='H₁: p = 0.6')
plt.title('Distribution of Test Statistics under H₀ and H₁')
plt.xlabel('Proportion of Heads')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate critical values for a two-tailed test with α = 0.05
alpha = 0.05
lower_critical = torch.quantile(h0_test_stats, alpha/2)
upper_critical = torch.quantile(h0_test_stats, 1 - alpha/2)

print(f"Critical region: proportion < {lower_critical.item():.4f} or proportion > {upper_critical.item():.4f}")

# Calculate power of the test (probability of correctly rejecting H₀ when H₁ is true)
power = torch.sum((h1_test_stats < lower_critical) | (h1_test_stats > upper_critical)).float() / num_simulations
print(f"Power of the test: {power.item():.4f}")
```

**When and where we use hypothesis testing:**
- In model comparison to determine if one model significantly outperforms another
- In A/B testing to evaluate the effectiveness of changes in a system
- In feature selection to determine if a feature significantly contributes to model performance
- In validating experimental results in research

**Real-world examples:**
- Testing if a new drug is more effective than a placebo
- Determining if a website redesign increases user engagement
- Evaluating if a new machine learning algorithm improves prediction accuracy
- Assessing if a feature in a recommendation system increases click-through rates

### 2. Z-test

**What is a Z-test?**
A Z-test is a hypothesis test that uses the standard normal distribution as the sampling distribution. It's typically used when:
- The sample size is large (n ≥ 30)
- The population standard deviation is known
- The data follows a normal distribution (or approximately normal due to Central Limit Theorem)

**PyTorch Implementation:**
```python
import torch
import math

def z_test(sample_mean, population_mean, population_std, sample_size, alternative='two-sided'):
    """
    Perform a Z-test
    
    Parameters:
    - sample_mean: Mean of the sample
    - population_mean: Mean of the population under H₀
    - population_std: Standard deviation of the population
    - sample_size: Size of the sample
    - alternative: Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    - z_stat: Z-test statistic
    - p_value: p-value of the test
    """
    # Calculate standard error
    standard_error = population_std / math.sqrt(sample_size)
    
    # Calculate Z-test statistic
    z_stat = (sample_mean - population_mean) / standard_error
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        p_value = 2 * (1 - torch.distributions.Normal(0, 1).cdf(torch.abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - torch.distributions.Normal(0, 1).cdf(z_stat)
    elif alternative == 'less':
        p_value = torch.distributions.Normal(0, 1).cdf(z_stat)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    return z_stat, p_value

# Example: Test if a new teaching method improves test scores
# H₀: μ = 75 (no improvement)
# H₁: μ > 75 (improvement)

# Known population parameters
population_mean = 75
population_std = 10

# Sample data after implementing new method
sample_scores = torch.tensor([82, 79, 85, 78, 90, 83, 76, 88, 84, 81])
sample_mean = torch.mean(sample_scores.float())
sample_size = len(sample_scores)

# Perform one-sided Z-test
z_stat, p_value = z_test(sample_mean, population_mean, population_std, sample_size, alternative='greater')

print(f"Sample mean: {sample_mean.item():.2f}")
print(f"Z-statistic: {z_stat.item():.4f}")
print(f"P-value: {p_value.item():.4f}")

# Decision at α = 0.05 significance level
alpha = 0.05
if p_value < alpha:
    print("Reject H₀: There is evidence that the new teaching method improves test scores.")
else:
    print("Fail to reject H₀: There is not enough evidence to conclude that the new teaching method improves test scores.")
```

**When and where we use the Z-test:**
- When comparing a sample mean to a known population mean
- In quality control to test if a process meets specifications
- In large-scale A/B testing where sample sizes are large
- When the population standard deviation is known

**Real-world examples:**
- Testing if a manufacturing process produces items within specifications
- Evaluating if a new fertilizer increases crop yield compared to the historical average
- Assessing if a training program improves employee productivity
- Determining if a website change increases conversion rates

### 3. T-test

**What is a T-test?**
A T-test is a hypothesis test that uses the Student's t-distribution as the sampling distribution. It's typically used when:
- The sample size is small (n < 30)
- The population standard deviation is unknown (estimated from the sample)
- The data follows a normal distribution (or approximately normal)

There are three main types of T-tests:
1. **One-sample T-test**: Compares a sample mean to a known or hypothesized population mean
2. **Independent two-sample T-test**: Compares the means of two independent groups
3. **Paired T-test**: Compares the means of two related groups (e.g., before and after measurements)

**PyTorch Implementation:**
```python
import torch
import math

def one_sample_t_test(sample, population_mean, alternative='two-sided'):
    """
    Perform a one-sample T-test
    
    Parameters:
    - sample: Sample data as a tensor
    - population_mean: Hypothesized population mean under H₀
    - alternative: Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    - t_stat: T-test statistic
    - p_value: p-value of the test
    - df: Degrees of freedom
    """
    # Calculate sample statistics
    sample_size = len(sample)
    sample_mean = torch.mean(sample.float())
    sample_std = torch.std(sample.float(), unbiased=True)
    
    # Calculate standard error
    standard_error = sample_std / math.sqrt(sample_size)
    
    # Calculate T-test statistic
    t_stat = (sample_mean - population_mean) / standard_error
    
    # Degrees of freedom
    df = sample_size - 1
    
    # Calculate p-value based on alternative hypothesis
    t_dist = torch.distributions.StudentT(df)
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - t_dist.cdf(torch.abs(t_stat)))
    elif alternative == 'greater':
        p_value = 1 - t_dist.cdf(t_stat)
    elif alternative == 'less':
        p_value = t_dist.cdf(t_stat)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    return t_stat, p_value, df

def independent_t_test(sample1, sample2, alternative='two-sided', equal_var=True):
    """
    Perform an independent two-sample T-test
    
    Parameters:
    - sample1: First sample data as a tensor
    - sample2: Second sample data as a tensor
    - alternative: Type of test ('two-sided', 'greater', 'less')
    - equal_var: If True, assume equal population variances (Student's t-test)
                 If False, do not assume equal variances (Welch's t-test)
    
    Returns:
    - t_stat: T-test statistic
    - p_value: p-value of the test
    - df: Degrees of freedom
    """
    # Calculate sample statistics
    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = torch.mean(sample1.float())
    mean2 = torch.mean(sample2.float())
    var1 = torch.var(sample1.float(), unbiased=True)
    var2 = torch.var(sample2.float(), unbiased=True)
    
    if equal_var:
        # Pooled variance (Student's t-test)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Standard error
        standard_error = math.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # Degrees of freedom
        df = n1 + n2 - 2
    else:
        # Welch's t-test (unequal variances)
        standard_error = math.sqrt(var1/n1 + var2/n2)
        
        # Degrees of freedom (approximation)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # Calculate T-test statistic
    t_stat = (mean1 - mean2) / standard_error
    
    # Calculate p-value based on alternative hypothesis
    t_dist = torch.distributions.StudentT(df)
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - t_dist.cdf(torch.abs(t_stat)))
    elif alternative == 'greater':
        p_value = 1 - t_dist.cdf(t_stat)
    elif alternative == 'less':
        p_value = t_dist.cdf(t_stat)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    return t_stat, p_value, df

def paired_t_test(sample1, sample2, alternative='two-sided'):
    """
    Perform a paired T-test
    
    Parameters:
    - sample1: First sample data as a tensor
    - sample2: Second sample data as a tensor
    - alternative: Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    - t_stat: T-test statistic
    - p_value: p-value of the test
    - df: Degrees of freedom
    """
    # Calculate differences
    differences = sample1.float() - sample2.float()
    
    # Perform one-sample t-test on the differences
    return one_sample_t_test(differences, 0, alternative)

# Example 1: One-sample T-test
# Test if the mean IQ of a sample is different from the population mean of 100
iq_scores = torch.tensor([105, 110, 95, 120, 100, 98, 112, 108, 103, 97])
population_mean = 100

t_stat, p_value, df = one_sample_t_test(iq_scores, population_mean, alternative='two-sided')

print("One-sample T-test:")
print(f"T-statistic: {t_stat.item():.4f}")
print(f"P-value: {p_value.item():.4f}")
print(f"Degrees of freedom: {df}")

# Decision at α = 0.05 significance level
alpha = 0.05
if p_value < alpha:
    print("Reject H₀: There is evidence that the mean IQ is different from 100.")
else:
    print("Fail to reject H₀: There is not enough evidence to conclude that the mean IQ is different from 100.")

# Example 2: Independent two-sample T-test
# Test if there is a difference in test scores between two teaching methods
method1_scores = torch.tensor([82, 79, 85, 78, 90, 83, 76, 88, 84, 81])
method2_scores = torch.tensor([75, 80, 72, 85, 78, 76, 82, 79, 77, 80])

t_stat, p_value, df = independent_t_test(method1_scores, method2_scores, alternative='two-sided')

print("\nIndependent two-sample T-test:")
print(f"T-statistic: {t_stat.item():.4f}")
print(f"P-value: {p_value.item():.4f}")
print(f"Degrees of freedom: {df}")

# Decision at α = 0.05 significance level
if p_value < alpha:
    print("Reject H₀: There is evidence of a difference in test scores between the two teaching methods.")
else:
    print("Fail to reject H₀: There is not enough evidence to conclude that there is a difference in test scores between the two teaching methods.")

# Example 3: Paired T-test
# Test if a new diet program leads to weight loss
before_weights = torch.tensor([180, 195, 210, 175, 190, 205, 185, 200, 170, 195])
after_weights = torch.tensor([175, 190, 200, 170, 185, 195, 180, 190, 165, 185])

t_stat, p_value, df = paired_t_test(before_weights, after_weights, alternative='greater')

print("\nPaired T-test:")
print(f"T-statistic: {t_stat.item():.4f}")
print(f"P-value: {p_value.item():.4f}")
print(f"Degrees of freedom: {df}")

# Decision at α = 0.05 significance level
if p_value < alpha:
    print("Reject H₀: There is evidence that the diet program leads to weight loss.")
else:
    print("Fail to reject H₀: There is not enough evidence to conclude that the diet program leads to weight loss.")
```

**When and where we use T-tests:**
- When comparing means with small sample sizes
- When the population standard deviation is unknown
- In experimental designs with control and treatment groups
- In before-and-after studies to measure the effect of an intervention

**Real-world examples:**
- Testing if a new drug lowers blood pressure compared to a placebo
- Evaluating if two different teaching methods result in different test scores
- Determining if a training program improves employee performance
- Assessing if a new manufacturing process produces different product dimensions

### 4. ANOVA (Analysis of Variance)

**What is ANOVA?**
ANOVA (Analysis of Variance) is a statistical method used to compare the means of three or more groups. It tests whether there are statistically significant differences between the means of independent groups.

The basic idea behind ANOVA is to partition the total variation in the data into:
- **Between-group variation**: Variability due to differences between group means
- **Within-group variation**: Variability due to differences within each group

The F-statistic is calculated as the ratio of between-group variability to within-group variability. A large F-statistic suggests that the between-group variability is larger than would be expected by chance.

**Types of ANOVA:**
1. **One-way ANOVA**: Tests the effect of one independent variable (factor) on a dependent variable
2. **Two-way ANOVA**: Tests the effects of two independent variables and their interaction on a dependent variable
3. **Repeated Measures ANOVA**: Tests the effect of one or more factors when the same subjects are measured multiple times

**PyTorch Implementation:**
```python
import torch
import math

def one_way_anova(*groups):
    """
    Perform one-way ANOVA
    
    Parameters:
    - groups: Variable number of tensors, each representing a group
    
    Returns:
    - f_stat: F-statistic
    - p_value: p-value of the test
    - df_between: Degrees of freedom between groups
    - df_within: Degrees of freedom within groups
    """
    # Number of groups
    k = len(groups)
    
    # Total number of observations
    n_total = sum(len(group) for group in groups)
    
    # Calculate overall mean
    all_data = torch.cat([group.float() for group in groups])
    overall_mean = torch.mean(all_data)
    
    # Calculate sum of squares between groups (SSB)
    ss_between = torch.tensor(0.0)
    for group in groups:
        group_mean = torch.mean(group.float())
        group_size = len(group)
        ss_between += group_size * (group_mean - overall_mean)**2
    
    # Calculate sum of squares within groups (SSW)
    ss_within = torch.tensor(0.0)
    for group in groups:
        group_mean = torch.mean(group.float())
        for value in group:
            ss_within += (value.float() - group_mean)**2
    
    # Calculate degrees of freedom
    df_between = k - 1
    df_within = n_total - k
    
    # Calculate mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # Calculate F-statistic
    f_stat = ms_between / ms_within
    
    # Calculate p-value
    f_dist = torch.distributions.F(df_between, df_within)
    p_value = 1 - f_dist.cdf(f_stat)
    
    return f_stat, p_value, df_between, df_within

# Example: Test if there are differences in test scores among three teaching methods
method1 = torch.tensor([82, 79, 85, 78, 90, 83, 76, 88, 84, 81])
method2 = torch.tensor([75, 80, 72, 85, 78, 76, 82, 79, 77, 80])
method3 = torch.tensor([90, 85, 92, 88, 95, 87, 91, 89, 93, 86])

f_stat, p_value, df_between, df_within = one_way_anova(method1, method2, method3)

print("One-way ANOVA:")
print(f"F-statistic: {f_stat.item():.4f}")
print(f"P-value: {p_value.item():.4f}")
print(f"Degrees of freedom (between): {df_between}")
print(f"Degrees of freedom (within): {df_within}")

# Decision at α = 0.05 significance level
alpha = 0.05
if p_value < alpha:
    print("Reject H₀: There is evidence of a difference in test scores among the teaching methods.")
    print("Post-hoc tests (e.g., Tukey's HSD) would be needed to determine which specific methods differ.")
else:
    print("Fail to reject H₀: There is not enough evidence to conclude that there is a difference in test scores among the teaching methods.")

# Calculate group means for interpretation
print("\nGroup means:")
print(f"Method 1: {torch.mean(method1.float()).item():.2f}")
print(f"Method 2: {torch.mean(method2.float()).item():.2f}")
print(f"Method 3: {torch.mean(method3.float()).item():.2f}")
```

**When and where we use ANOVA:**
- When comparing the means of three or more groups
- In experimental designs with multiple treatment groups
- When analyzing the effects of categorical independent variables on a continuous dependent variable
- In quality control to compare multiple production processes

**Real-world examples:**
- Comparing the effectiveness of multiple drugs in a clinical trial
- Evaluating the performance of different marketing campaigns
- Testing the yield of different crop varieties
- Analyzing customer satisfaction across multiple store locations

### 5. A/B Testing

**What is A/B Testing?**
A/B testing (also known as bucket testing or split testing) is a randomized experiment with two variants, A and B. It's a way to compare two versions of a single variable to determine which one performs better in a controlled environment.

In A/B testing:
- **Version A**: The control (current version)
- **Version B**: The variation (new version)
- **Metric**: A measure of performance (e.g., click-through rate, conversion rate)

**Key Concepts in A/B Testing:**
- **Randomization**: Randomly assigning users to different versions to ensure groups are comparable
- **Sample Size**: Determining how many users are needed to detect a meaningful difference
- **Statistical Significance**: Determining whether observed differences are likely due to the intervention or chance
- **Practical Significance**: Considering whether the observed difference is meaningful in a business context

**PyTorch Implementation:**
```python
import torch
import math

def calculate_sample_size(baseline_rate, minimum_detectable_effect, significance_level=0.05, power=0.8):
    """
    Calculate the required sample size for an A/B test
    
    Parameters:
    - baseline_rate: Baseline conversion rate (e.g., 0.1 for 10%)
    - minimum_detectable_effect: Minimum effect size to detect (e.g., 0.02 for 2%)
    - significance_level: Type I error rate (α)
    - power: Statistical power (1 - Type II error rate)
    
    Returns:
    - sample_size_per_group: Required sample size per group
    """
    # Convert rates to proportions
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    
    # Calculate pooled proportion
    pooled_p = (p1 + p2) / 2
    
    # Calculate Z values for the given significance level and power
    z_alpha = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - significance_level/2))
    z_beta = torch.distributions.Normal(0, 1).icdf(torch.tensor(power))
    
    # Calculate sample size per group
    numerator = (z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    denominator = (p2 - p1)**2
    
    sample_size_per_group = math.ceil(numerator / denominator)
    
    return sample_size_per_group

def ab_test(group_a, group_b, alternative='two-sided'):
    """
    Perform an A/B test (two-proportion z-test)
    
    Parameters:
    - group_a: Binary outcomes for group A (1 = success, 0 = failure)
    - group_b: Binary outcomes for group B (1 = success, 0 = failure)
    - alternative: Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    - z_stat: Z-test statistic
    - p_value: p-value of the test
    - ci_lower: Lower bound of confidence interval for the difference
    - ci_upper: Upper bound of confidence interval for the difference
    """
    # Calculate counts and proportions
    n_a = len(group_a)
    n_b = len(group_b)
    successes_a = torch.sum(group_a)
    successes_b = torch.sum(group_b)
    
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    
    # Calculate pooled proportion
    pooled_p = (successes_a + successes_b) / (n_a + n_b)
    
    # Calculate standard error
    standard_error = math.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
    
    # Calculate Z-test statistic
    z_stat = (p_a - p_b) / standard_error
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        p_value = 2 * (1 - torch.distributions.Normal(0, 1).cdf(torch.abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - torch.distributions.Normal(0, 1).cdf(z_stat)
    elif alternative == 'less':
        p_value = torch.distributions.Normal(0, 1).cdf(z_stat)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    # Calculate confidence interval for the difference
    margin_of_error = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.975)) * standard_error
    difference = p_a - p_b
    ci_lower = difference - margin_of_error
    ci_upper = difference + margin_of_error
    
    return z_stat, p_value, ci_lower, ci_upper

# Example: A/B test for a website button color change
# Group A sees the blue button (control)
# Group B sees the green button (variation)
# We want to know if the green button leads to a higher click-through rate

# Simulate data
torch.manual_seed(42)  # For reproducibility

# Group A (blue button): 10% click-through rate
n_a = 1000
group_a = torch.rand(n_a) < 0.1

# Group B (green button): 12% click-through rate
n_b = 1000
group_b = torch.rand(n_b) < 0.12

# Perform A/B test
z_stat, p_value, ci_lower, ci_upper = ab_test(group_a, group_b, alternative='greater')

print("A/B Test Results:")
print(f"Group A (blue) click-through rate: {torch.mean(group_a.float()).item():.4f}")
print(f"Group B (green) click-through rate: {torch.mean(group_b.float()).item():.4f}")
print(f"Difference: {(torch.mean(group_b.float()) - torch.mean(group_a.float())).item():.4f}")
print(f"Z-statistic: {z_stat.item():.4f}")
print(f"P-value: {p_value.item():.4f}")
print(f"95% Confidence Interval for Difference: [{ci_lower.item():.4f}, {ci_upper.item():.4f}]")

# Decision at α = 0.05 significance level
alpha = 0.05
if p_value < alpha:
    print("\nReject H₀: There is evidence that the green button has a higher click-through rate.")
else:
    print("\nFail to reject H₀: There is not enough evidence to conclude that the green button has a higher click-through rate.")

# Calculate required sample size for future tests
baseline_rate = 0.1  # 10% baseline click-through rate
minimum_detectable_effect = 0.02  # Want to detect a 2% increase
required_sample_size = calculate_sample_size(baseline_rate, minimum_detectable_effect)

print(f"\nRequired sample size per group to detect a {minimum_detectable_effect*100:.1f}% increase from a baseline of {baseline_rate*100:.1f}%:")
print(f"{required_sample_size} users per group")
```

**When and where we use A/B testing:**
- When comparing two versions of a webpage, app, or marketing campaign
- In product development to test new features
- In marketing to evaluate different ad creatives or messaging
- In user experience design to optimize interfaces

**Real-world examples:**
- Testing different button colors on a website to increase click-through rates
- Comparing two email subject lines to improve open rates
- Evaluating different pricing strategies to maximize revenue
- Testing different recommendation algorithms to increase user engagement

## Applications in Deep Learning

### 1. Model Comparison

Hypothesis testing can be used to determine if one deep learning model significantly outperforms another:

```python
import torch
import numpy as np
from scipy import stats

def compare_models(model1_scores, model2_scores, alpha=0.05):
    """
    Compare two models using a paired t-test
    
    Parameters:
    - model1_scores: Accuracy scores for model 1 on multiple test sets
    - model2_scores: Accuracy scores for model 2 on multiple test sets
    - alpha: Significance level
    
    Returns:
    - t_stat: T-test statistic
    - p_value: p-value of the test
    - significant: Whether the difference is statistically significant
    """
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Determine if the difference is significant
    significant = p_value < alpha
    
    return t_stat, p_value, significant

# Example: Compare two models using cross-validation
# Simulate accuracy scores for two models across 10 folds
torch.manual_seed(42)
model1_scores = torch.tensor([0.85, 0.87, 0.83, 0.86, 0.84, 0.85, 0.87, 0.86, 0.84, 0.85])
model2_scores = torch.tensor([0.87, 0.88, 0.85, 0.89, 0.86, 0.87, 0.89, 0.88, 0.86, 0.87])

# Compare models
t_stat, p_value, significant = compare_models(model1_scores, model2_scores)

print("Model Comparison Results:")
print(f"Model 1 mean accuracy: {torch.mean(model1_scores).item():.4f}")
print(f"Model 2 mean accuracy: {torch.mean(model2_scores).item():.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if significant:
    print("The difference in performance is statistically significant.")
    if torch.mean(model2_scores) > torch.mean(model1_scores):
        print("Model 2 performs significantly better than Model 1.")
    else:
        print("Model 1 performs significantly better than Model 2.")
else:
    print("The difference in performance is not statistically significant.")
```

### 2. Feature Importance Testing

Hypothesis testing can be used to determine if a feature significantly contributes to model performance:

```python
import torch
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_feature_importance(X, y, feature_idx, n_folds=5, alpha=0.05):
    """
    Test if a feature significantly contributes to model performance
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - feature_idx: Index of the feature to test
    - n_folds: Number of cross-validation folds
    - alpha: Significance level
    
    Returns:
    - p_value: p-value of the test
    - significant: Whether the feature is statistically significant
    """
    # Model with all features
    model_all = RandomForestClassifier(random_state=42)
    scores_all = cross_val_score(model_all, X, y, cv=n_folds)
    
    # Model without the feature being tested
    X_without = np.delete(X, feature_idx, axis=1)
    model_without = RandomForestClassifier(random_state=42)
    scores_without = cross_val_score(model_without, X_without, y, cv=n_folds)
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(scores_all, scores_without)
    
    # Determine if the feature is significant
    significant = p_value < alpha
    
    return p_value, significant, np.mean(scores_all), np.mean(scores_without)

# Example usage with synthetic data
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Test importance of the first feature
feature_idx = 0
p_value, significant, mean_score_all, mean_score_without = test_feature_importance(X, y, feature_idx)

print(f"Feature Importance Test for Feature {feature_idx}:")
print(f"Mean accuracy with all features: {mean_score_all:.4f}")
print(f"Mean accuracy without feature {feature_idx}: {mean_score_without:.4f}")
print(f"P-value: {p_value:.4f}")

if significant:
    print(f"Feature {feature_idx} is statistically significant (p < 0.05).")
else:
    print(f"Feature {feature_idx} is not statistically significant (p >= 0.05).")
```

### 3. Hyperparameter Tuning

Hypothesis testing can be used to determine if changing a hyperparameter significantly affects model performance:

```python
import torch
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import stats

def test_hyperparameter(X, y, param_name, param_values, n_folds=5, alpha=0.05):
    """
    Test if changing a hyperparameter significantly affects model performance
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - param_name: Name of the hyperparameter to test
    - param_values: Values of the hyperparameter to test
    - n_folds: Number of cross-validation folds
    - alpha: Significance level
    
    Returns:
    - results: Dictionary containing test results for each parameter value
    """
    results = {}
    
    # Test each parameter value
    for value in param_values:
        # Set hyperparameter
        params = {param_name: value}
        model = RandomForestClassifier(random_state=42, **params)
        
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=n_folds)
        
        # Store results
        results[value] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
    
    # Compare each pair of parameter values
    comparisons = []
    param_value_list = list(param_values)
    
    for i in range(len(param_value_list)):
        for j in range(i+1, len(param_value_list)):
            value1 = param_value_list[i]
            value2 = param_value_list[j]
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(results[value1]['scores'], results[value2]['scores'])
            
            # Determine if the difference is significant
            significant = p_value < alpha
            
            comparisons.append({
                'value1': value1,
                'value2': value2,
                'mean1': results[value1]['mean'],
                'mean2': results[value2]['mean'],
                'p_value': p_value,
                'significant': significant
            })
    
    return results, comparisons

# Example usage with synthetic data
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Test different values for n_estimators
param_name = 'n_estimators'
param_values = [10, 50, 100]

results, comparisons = test_hyperparameter(X, y, param_name, param_values)

# Print results
print(f"Hyperparameter Test for {param_name}:")
for value in param_values:
    print(f"\n{param_name} = {value}:")
    print(f"Mean accuracy: {results[value]['mean']:.4f}")
    print(f"Standard deviation: {results[value]['std']:.4f}")

print("\nComparisons:")
for comp in comparisons:
    print(f"\n{param_name} {comp['value1']} vs {comp['value2']}:")
    print(f"Mean accuracy: {comp['mean1']:.4f} vs {comp['mean2']:.4f}")
    print(f"P-value: {comp['p_value']:.4f}")
    if comp['significant']:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")
```

## Common Problems and Advancements

### 1. Problem: Multiple Comparisons

**Issue:**
When conducting multiple hypothesis tests, the probability of making at least one Type I error increases. This is known as the multiple comparisons problem or family-wise error rate.

**How advancements solved this:**
- **Bonferroni Correction**: Adjusting the significance level by dividing it by the number of tests
- **False Discovery Rate (FDR)**: Controlling the expected proportion of false positives among rejected hypotheses
- **Holm-Bonferroni Method**: A step-down procedure that is more powerful than the simple Bonferroni correction
- **Permutation Tests**: Non-parametric methods that don't rely on assumptions about the data distribution

```python
import torch
import math
from scipy import stats

def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction to multiple p-values
    
    Parameters:
    - p_values: List or array of p-values
    - alpha: Original significance level
    
    Returns:
    - adjusted_alpha: Adjusted significance level
    - significant: Boolean array indicating which tests are significant after correction
    """
    p_values = torch.tensor(p_values)
    num_tests = len(p_values)
    
    # Adjust significance level
    adjusted_alpha = alpha / num_tests
    
    # Determine which tests are significant after correction
    significant = p_values < adjusted_alpha
    
    return adjusted_alpha, significant

def false_discovery_rate(p_values, q=0.05):
    """
    Control the False Discovery Rate using the Benjamini-Hochberg procedure
    
    Parameters:
    - p_values: List or array of p-values
    - q: Desired FDR level
    
    Returns:
    - significant: Boolean array indicating which tests are significant after FDR control
    """
    p_values = torch.tensor(p_values)
    num_tests = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_p_values, indices = torch.sort(p_values)
    
    # Calculate critical values
    critical_values = torch.arange(1, num_tests + 1) * q / num_tests
    
    # Find the largest p-value that is less than its critical value
    significant_indices = torch.where(sorted_p_values <= critical_values)[0]
    
    if len(significant_indices) > 0:
        max_index = torch.max(significant_indices)
        # All tests with p-values less than or equal to the p-value at max_index are significant
        threshold = sorted_p_values[max_index]
        significant = p_values <= threshold
    else:
        significant = torch.zeros(num_tests, dtype=torch.bool)
    
    return significant

# Example: Multiple model comparisons
# Simulate p-values from 10 different tests
torch.manual_seed(42)
p_values = torch.rand(10) * 0.1  # P-values between 0 and 0.1

print("Original p-values:")
for i, p in enumerate(p_values):
    print(f"Test {i+1}: {p.item():.4f}")

# Apply Bonferroni correction
adjusted_alpha, significant_bonferroni = bonferroni_correction(p_values)
print(f"\nBonferroni correction:")
print(f"Adjusted significance level: {adjusted_alpha:.4f}")
print("Significant tests:", torch.where(significant_bonferroni)[0].numpy() + 1)

# Apply FDR control
significant_fdr = false_discovery_rate(p_values)
print(f"\nFalse Discovery Rate control (q=0.05):")
print("Significant tests:", torch.where(significant_fdr)[0].numpy() + 1)
```

### 2. Problem: Assumption Violations

**Issue:**
Many hypothesis tests rely on assumptions about the data, such as normality, equal variances, or independence. When these assumptions are violated, the test results may not be valid.

**How advancements solved this:**
- **Non-parametric tests**: Methods that don't rely on distributional assumptions (e.g., Mann-Whitney U test, Kruskal-Wallis test)
- **Robust statistical methods**: Techniques that are less sensitive to assumption violations
- **Transformations**: Applying transformations to data to meet assumptions
- **Bootstrap methods**: Resampling techniques that don't rely on distributional assumptions

```python
import torch
import numpy as np
from scipy import stats

def mann_whitney_u_test(sample1, sample2, alternative='two-sided'):
    """
    Perform the Mann-Whitney U test (non-parametric alternative to independent t-test)
    
    Parameters:
    - sample1: First sample data as a tensor
    - sample2: Second sample data as a tensor
    - alternative: Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    - u_stat: U-test statistic
    - p_value: p-value of the test
    """
    # Convert to numpy arrays for scipy implementation
    sample1_np = sample1.numpy()
    sample2_np = sample2.numpy()
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(sample1_np, sample2_np, alternative=alternative)
    
    return u_stat, p_value

def bootstrap_test(sample1, sample2, n_bootstrap=10000, alternative='two-sided'):
    """
    Perform a bootstrap test for the difference in means
    
    Parameters:
    - sample1: First sample data as a tensor
    - sample2: Second sample data as a tensor
    - n_bootstrap: Number of bootstrap samples
    - alternative: Type of test ('two-sided', 'greater', 'less')
    
    Returns:
    - observed_diff: Observed difference in means
    - p_value: p-value of the test
    - ci_lower: Lower bound of confidence interval
    - ci_upper: Upper bound of confidence interval
    """
    # Calculate observed difference in means
    observed_diff = torch.mean(sample1.float()) - torch.mean(sample2.float())
    
    # Combine samples
    combined = torch.cat([sample1.float(), sample2.float()])
    n1 = len(sample1)
    n_total = len(combined)
    
    # Bootstrap distribution
    bootstrap_diffs = torch.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resample = combined[torch.randint(0, n_total, (n_total,))]
        
        # Split into two groups
        resample1 = resample[:n1]
        resample2 = resample[n1:]
        
        # Calculate difference in means
        bootstrap_diffs[i] = torch.mean(resample1) - torch.mean(resample2)
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        # Two-sided test: count how many bootstrap differences are as extreme as observed
        extreme_count = torch.sum(torch.abs(bootstrap_diffs) >= torch.abs(observed_diff))
        p_value = extreme_count / n_bootstrap
    elif alternative == 'greater':
        # One-sided test (greater): count how many bootstrap differences are >= observed
        extreme_count = torch.sum(bootstrap_diffs >= observed_diff)
        p_value = extreme_count / n_bootstrap
    elif alternative == 'less':
        # One-sided test (less): count how many bootstrap differences are <= observed
        extreme_count = torch.sum(bootstrap_diffs <= observed_diff)
        p_value = extreme_count / n_bootstrap
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    # Calculate confidence interval
    ci_lower = torch.quantile(bootstrap_diffs, 0.025)
    ci_upper = torch.quantile(bootstrap_diffs, 0.975)
    
    return observed_diff, p_value, ci_lower, ci_upper

# Example: Compare two groups with non-normal data
# Generate non-normal data (exponential distribution)
torch.manual_seed(42)
sample1 = torch.distributions.Exponential(torch.tensor([1.0])).sample((50,))
sample2 = torch.distributions.Exponential(torch.tensor([1.5])).sample((50,))

# Perform parametric t-test (may not be valid due to non-normality)
t_stat, p_value_t = stats.ttest_ind(sample1.numpy(), sample2.numpy())
print(f"Independent t-test (parametric):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value_t:.4f}")

# Perform non-parametric Mann-Whitney U test
u_stat, p_value_u = mann_whitney_u_test(sample1, sample2)
print(f"\nMann-Whitney U test (non-parametric):")
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value_u:.4f}")

# Perform bootstrap test
observed_diff, p_value_bootstrap, ci_lower, ci_upper = bootstrap_test(sample1, sample2)
print(f"\nBootstrap test:")
print(f"Observed difference in means: {observed_diff.item():.4f}")
print(f"P-value: {p_value_bootstrap.item():.4f}")
print(f"95% Confidence Interval: [{ci_lower.item():.4f}, {ci_upper.item():.4f}]")
```

### 3. Problem: Small Sample Sizes

**Issue:**
With small sample sizes, hypothesis tests may have low power (high Type II error rate), making it difficult to detect true effects.

**How advancements solved this:**
- **Bayesian methods**: Incorporating prior knowledge to improve inference with limited data
- **Effect size estimation**: Focusing on the magnitude of effects rather than just statistical significance
- **Sequential testing**: Continuously monitoring data and stopping when sufficient evidence is accumulated
- **Meta-analysis**: Combining results from multiple studies to increase effective sample size

```python
import torch
import math
from scipy import stats

def bayesian_t_test(sample1, sample2, prior_mean=0, prior_std=1, n_samples=10000):
    """
    Perform a Bayesian t-test using a normal-gamma conjugate prior
    
    Parameters:
    - sample1: First sample data as a tensor
    - sample2: Second sample data as a tensor
    - prior_mean: Prior mean for the difference
    - prior_std: Prior standard deviation for the difference
    - n_samples: Number of MCMC samples
    
    Returns:
    - posterior_mean: Posterior mean of the difference
    - posterior_std: Posterior standard deviation of the difference
    - prob_positive: Probability that the difference is positive
    """
    # Calculate sample statistics
    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = torch.mean(sample1.float())
    mean2 = torch.mean(sample2.float())
    var1 = torch.var(sample1.float(), unbiased=True)
    var2 = torch.var(sample2.float(), unbiased=True)
    
    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # Standard error
    se = math.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # Observed difference
    observed_diff = mean1 - mean2
    
    # Posterior parameters (using normal approximation)
    # Posterior precision = 1/prior_var + 1/se^2
    # Posterior mean = (prior_mean/prior_var + observed_diff/se^2) / posterior_precision
    
    prior_var = prior_std**2
    posterior_precision = 1/prior_var + 1/se**2
    posterior_var = 1/posterior_precision
    posterior_mean = (prior_mean/prior_var + observed_diff/se**2) / posterior_precision
    posterior_std = math.sqrt(posterior_var)
    
    # Sample from posterior
    posterior_samples = torch.normal(posterior_mean, posterior_std, (n_samples,))
    
    # Calculate probability that difference is positive
    prob_positive = torch.mean((posterior_samples > 0).float())
    
    return posterior_mean, posterior_std, prob_positive.item()

def calculate_effect_size(sample1, sample2):
    """
    Calculate Cohen's d effect size for two samples
    
    Parameters:
    - sample1: First sample data as a tensor
    - sample2: Second sample data as a tensor
    
    Returns:
    - effect_size: Cohen's d effect size
    - interpretation: Interpretation of the effect size
    """
    # Calculate sample statistics
    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = torch.mean(sample1.float())
    mean2 = torch.mean(sample2.float())
    var1 = torch.var(sample1.float(), unbiased=True)
    var2 = torch.var(sample2.float(), unbiased=True)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    effect_size = (mean1 - mean2) / pooled_std
    
    # Interpret effect size
    abs_effect = abs(effect_size.item())
    if abs_effect < 0.2:
        interpretation = "Small"
    elif abs_effect < 0.5:
        interpretation = "Medium"
    elif abs_effect < 0.8:
        interpretation = "Large"
    else:
        interpretation = "Very large"
    
    return effect_size.item(), interpretation

# Example: Compare two groups with small sample sizes
# Generate data with small sample sizes
torch.manual_seed(42)
sample1 = torch.normal(0, 1, (10,))  # n=10
sample2 = torch.normal(0.5, 1, (10,))  # n=10

# Perform traditional t-test
t_stat, p_value = stats.ttest_ind(sample1.numpy(), sample2.numpy())
print(f"Traditional t-test:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Perform Bayesian t-test
posterior_mean, posterior_std, prob_positive = bayesian_t_test(sample1, sample2)
print(f"\nBayesian t-test:")
print(f"Posterior mean of difference: {posterior_mean:.4f}")
print(f"Posterior standard deviation: {posterior_std:.4f}")
print(f"Probability that difference is positive: {prob_positive:.4f}")

# Calculate effect size
effect_size, interpretation = calculate_effect_size(sample1, sample2)
print(f"\nEffect size (Cohen's d): {effect_size:.4f} ({interpretation})")
```

### 4. Problem: Sequential Testing and Peeking

**Issue:**
In traditional hypothesis testing, repeatedly checking the results as data accumulates (peeking) can inflate the Type I error rate. This is a common problem in A/B testing where stakeholders want to monitor results in real-time.

**How advancements solved this:**
- **Sequential hypothesis testing**: Methods designed for continuous monitoring of data
- **Alpha-spending functions**: Allocating the Type I error rate across multiple looks at the data
- **Bayesian approaches**: Using Bayesian methods that naturally accommodate sequential updating
- **Group sequential designs**: Pre-specifying interim analyses and adjusting critical values

```python
import torch
import math
from scipy import stats

def sequential_test(stream1, stream2, alpha=0.05, max_looks=10):
    """
    Perform sequential hypothesis testing with alpha-spending
    
    Parameters:
    - stream1: Data stream for group 1 (tensor or list)
    - stream2: Data stream for group 2 (tensor or list)
    - alpha: Overall significance level
    - max_looks: Maximum number of times to look at the data
    
    Returns:
    - results: List of results at each look
    - stopped_at: Look at which the test stopped (None if not stopped)
    """
    # Convert to tensors if needed
    if not isinstance(stream1, torch.Tensor):
        stream1 = torch.tensor(stream1, dtype=torch.float)
    if not isinstance(stream2, torch.Tensor):
        stream2 = torch.tensor(stream2, dtype=torch.float)
    
    # Ensure streams are the same length
    min_length = min(len(stream1), len(stream2))
    stream1 = stream1[:min_length]
    stream2 = stream2[:min_length]
    
    # Calculate look times (equally spaced)
    look_times = torch.linspace(1, min_length, max_looks, dtype=torch.long)
    
    # Alpha-spending function (O'Brien-Fleming-like)
    def alpha_spending(t, T, alpha):
        # t is current time, T is total time
        return alpha * (1 - math.exp(-math.log(2) * t / T))
    
    results = []
    stopped_at = None
    
    for i, look in enumerate(look_times):
        # Get data up to this look
        data1 = stream1[:look]
        data2 = stream2[:look]
        
        # Calculate test statistic
        n1 = len(data1)
        n2 = len(data2)
        mean1 = torch.mean(data1)
        mean2 = torch.mean(data2)
        var1 = torch.var(data1, unbiased=True)
        var2 = torch.var(data2, unbiased=True)
        
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Standard error
        se = math.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # T-statistic
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Two-sided p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat.item()), df))
        
        # Calculate critical value based on alpha-spending
        alpha_spent = alpha_spending(i+1, max_looks, alpha)
        critical_value = stats.t.ppf(1 - alpha_spent/2, df)
        
        # Determine if test is significant
        significant = abs(t_stat.item()) > critical_value
        
        # Store results
        results.append({
            'look': i+1,
            'sample_size': look,
            't_statistic': t_stat.item(),
            'p_value': p_value,
            'critical_value': critical_value,
            'alpha_spent': alpha_spent,
            'significant': significant
        })
        
        # Check if test should stop
        if significant and stopped_at is None:
            stopped_at = i+1
    
    return results, stopped_at

# Example: Sequential A/B test
# Simulate data streams
torch.manual_seed(42)
n_total = 1000
stream1 = torch.normal(0, 1, (n_total,))  # Control group
stream2 = torch.normal(0.2, 1, (n_total,))  # Treatment group (small effect)

# Perform sequential test
results, stopped_at = sequential_test(stream1, stream2, alpha=0.05, max_looks=10)

# Print results
print("Sequential Test Results:")
print("Look | Sample Size | T-statistic | P-value | Critical Value | Alpha Spent | Significant")
print("-" * 85)

for result in results:
    print(f"{result['look']:4} | {result['sample_size']:11} | {result['t_statistic']:12.4f} | {result['p_value']:7.4f} | {result['critical_value']:14.4f} | {result['alpha_spent']:12.4f} | {result['significant']}")

if stopped_at is not None:
    print(f"\nTest stopped at look {stopped_at} with significant result.")
else:
    print("\nTest did not stop early. No significant result detected.")

# Compare with fixed sample size test
t_stat, p_value = stats.ttest_ind(stream1.numpy(), stream2.numpy())
print(f"\nFixed sample size test (n={n_total}):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at alpha=0.05: {p_value < 0.05}")
```

## Project: Sales Performance Tester

Now, let's implement a sales performance tester using PyTorch that demonstrates the hypothesis testing concepts we've covered:

```python
import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class SalesPerformanceTester:
    def __init__(self):
        pass
    
    def one_sample_t_test(self, sales_data, target_mean, alpha=0.05):
        """
        Test if the mean sales is different from a target value
        
        Parameters:
        - sales_data: Sales data as a tensor
        - target_mean: Target mean to test against
        - alpha: Significance level
        
        Returns:
        - t_stat: T-test statistic
        - p_value: p-value of the test
        - significant: Whether the result is statistically significant
        """
        # Convert to float tensor
        sales_data = sales_data.float()
        
        # Calculate sample statistics
        sample_size = len(sales_data)
        sample_mean = torch.mean(sales_data)
        sample_std = torch.std(sales_data, unbiased=True)
        
        # Calculate standard error
        standard_error = sample_std / torch.sqrt(torch.tensor(sample_size, dtype=torch.float))
        
        # Calculate T-test statistic
        t_stat = (sample_mean - target_mean) / standard_error
        
        # Degrees of freedom
        df = sample_size - 1
        
        # Calculate p-value (two-sided test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat.item()), df))
        
        # Determine if significant
        significant = p_value < alpha
        
        return t_stat.item(), p_value, significant
    
    def two_sample_t_test(self, sales_data1, sales_data2, alpha=0.05, equal_var=True):
        """
        Test if there is a difference in mean sales between two groups
        
        Parameters:
        - sales_data1: Sales data for group 1 as a tensor
        - sales_data2: Sales data for group 2 as a tensor
        - alpha: Significance level
        - equal_var: Whether to assume equal variances
        
        Returns:
        - t_stat: T-test statistic
        - p_value: p-value of the test
        - significant: Whether the result is statistically significant
        """
        # Convert to float tensors
        sales_data1 = sales_data1.float()
        sales_data2 = sales_data2.float()
        
        # Calculate sample statistics
        n1 = len(sales_data1)
        n2 = len(sales_data2)
        mean1 = torch.mean(sales_data1)
        mean2 = torch.mean(sales_data2)
        var1 = torch.var(sales_data1, unbiased=True)
        var2 = torch.var(sales_data2, unbiased=True)
        
        if equal_var:
            # Pooled variance (Student's t-test)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            
            # Standard error
            standard_error = torch.sqrt(pooled_var * (1/n1 + 1/n2))
            
            # Degrees of freedom
            df = n1 + n2 - 2
        else:
            # Welch's t-test (unequal variances)
            standard_error = torch.sqrt(var1/n1 + var2/n2)
            
            # Degrees of freedom (approximation)
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Calculate T-test statistic
        t_stat = (mean1 - mean2) / standard_error
        
        # Calculate p-value (two-sided test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat.item()), df))
        
        # Determine if significant
        significant = p_value < alpha
        
        return t_stat.item(), p_value, significant
    
    def anova_test(self, *groups, alpha=0.05):
        """
        Test if there are differences in mean sales among multiple groups
        
        Parameters:
        - groups: Variable number of tensors, each representing a group
        - alpha: Significance level
        
        Returns:
        - f_stat: F-statistic
        - p_value: p-value of the test
        - significant: Whether the result is statistically significant
        """
        # Number of groups
        k = len(groups)
        
        # Total number of observations
        n_total = sum(len(group) for group in groups)
        
        # Calculate overall mean
        all_data = torch.cat([group.float() for group in groups])
        overall_mean = torch.mean(all_data)
        
        # Calculate sum of squares between groups (SSB)
        ss_between = torch.tensor(0.0)
        for group in groups:
            group_mean = torch.mean(group.float())
            group_size = len(group)
            ss_between += group_size * (group_mean - overall_mean)**2
        
        # Calculate sum of squares within groups (SSW)
        ss_within = torch.tensor(0.0)
        for group in groups:
            group_mean = torch.mean(group.float())
            for value in group:
                ss_within += (value.float() - group_mean)**2
        
        # Calculate degrees of freedom
        df_between = k - 1
        df_within = n_total - k
        
        # Calculate mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        # Calculate F-statistic
        f_stat = ms_between / ms_within
        
        # Calculate p-value
        p_value = 1 - stats.f.cdf(f_stat.item(), df_between, df_within)
        
        # Determine if significant
        significant = p_value < alpha
        
        return f_stat.item(), p_value, significant
    
    def ab_test(self, group_a, group_b, alpha=0.05):
        """
        Perform an A/B test on sales data
        
        Parameters:
        - group_a: Sales data for group A as a tensor
        - group_b: Sales data for group B as a tensor
        - alpha: Significance level
        
        Returns:
        - results: Dictionary containing test results
        """
        # Calculate basic statistics
        n_a = len(group_a)
        n_b = len(group_b)
        mean_a = torch.mean(group_a.float()).item()
        mean_b = torch.mean(group_b.float()).item()
        std_a = torch.std(group_a.float()).item()
        std_b = torch.std(group_b.float()).item()
        
        # Perform t-test
        t_stat, p_value, significant = self.two_sample_t_test(group_a, group_b, alpha)
        
        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = (mean_a - mean_b) / pooled_std
        
        # Calculate confidence interval for the difference
        se_diff = math.sqrt(std_a**2/n_a + std_b**2/n_b)
        margin_of_error = stats.t.ppf(1 - alpha/2, n_a + n_b - 2) * se_diff
        ci_lower = (mean_a - mean_b) - margin_of_error
        ci_upper = (mean_a - mean_b) + margin_of_error
        
        # Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_interpretation = "Small"
        elif abs_effect < 0.5:
            effect_interpretation = "Medium"
        elif abs_effect < 0.8:
            effect_interpretation = "Large"
        else:
            effect_interpretation = "Very large"
        
        return {
            'group_a_size': n_a,
            'group_b_size': n_b,
            'group_a_mean': mean_a,
            'group_b_mean': mean_b,
            'group_a_std': std_a,
            'group_b_std': std_b,
            'difference': mean_a - mean_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'effect_size': effect_size,
            'effect_interpretation': effect_interpretation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def plot_sales_comparison(self, *groups, labels=None):
        """
        Plot a comparison of sales data across groups
        
        Parameters:
        - groups: Variable number of tensors, each representing a group
        - labels: Labels for each group
        """
        if labels is None:
            labels = [f"Group {i+1}" for i in range(len(groups))]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        plt.subplot(1, 2, 1)
        plt.boxplot([group.numpy() for group in groups], labels=labels)
        plt.title('Sales Distribution by Group')
        plt.ylabel('Sales')
        
        # Create bar chart of means
        plt.subplot(1, 2, 2)
        means = [torch.mean(group.float()).item() for group in groups]
        stds = [torch.std(group.float()).item() for group in groups]
        plt.bar(labels, means, yerr=stds, alpha=0.7)
        plt.title('Mean Sales by Group')
        plt.ylabel('Mean Sales')
        
        plt.tight_layout()
        plt.show()
    
    def simulate_sales_data(self, base_mean=100, base_std=20, group_sizes=None, effects=None, n_groups=2):
        """
        Simulate sales data for testing
        
        Parameters:
        - base_mean: Base mean for the control group
        - base_std: Standard deviation for all groups
        - group_sizes: List of group sizes
        - effects: List of effects to add to each group (first group should be 0 for control)
        - n_groups: Number of groups (if group_sizes or effects not provided)
        
        Returns:
        - groups: List of tensors with simulated sales data
        """
        if group_sizes is None:
            group_sizes = [100] * n_groups
        
        if effects is None:
            effects = [0] * n_groups
        
        groups = []
        for i in range(len(group_sizes)):
            group_mean = base_mean + effects[i]
            group_data = torch.normal(group_mean, base_std, (group_sizes[i],))
            groups.append(group_data)
        
        return groups

# Example usage
tester = SalesPerformanceTester()

# Simulate sales data for three different sales strategies
torch.manual_seed(42)
control = torch.normal(100, 20, (100,))  # Control group
strategy1 = torch.normal(110, 20, (100,))  # Strategy 1 (10% improvement)
strategy2 = torch.normal(105, 20, (100,))  # Strategy 2 (5% improvement)

# One-sample t-test: Test if strategy1 is different from a target of 100
t_stat, p_value, significant = tester.one_sample_t_test(strategy1, 100)
print(f"One-sample t-test for Strategy 1 vs target of 100:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant: {significant}")
print()

# Two-sample t-test: Compare strategy1 vs control
t_stat, p_value, significant = tester.two_sample_t_test(strategy1, control)
print(f"Two-sample t-test for Strategy 1 vs Control:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant: {significant}")
print()

# ANOVA: Compare all three groups
f_stat, p_value, significant = tester.anova_test(control, strategy1, strategy2)
print(f"ANOVA for all three groups:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant: {significant}")
print()

# A/B test: Compare strategy1 vs control
results = tester.ab_test(control, strategy1)
print("A/B Test Results for Strategy 1 vs Control:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")
print()

# Plot comparison
tester.plot_sales_comparison(control, strategy1, strategy2, labels=["Control", "Strategy 1", "Strategy 2"])

# Simulate data for a sequential test
stream1 = torch.normal(100, 20, (1000,))  # Control group
stream2 = torch.normal(105, 20, (1000,))  # Treatment group (5% improvement)

# Perform sequential test
def sequential_test(stream1, stream2, alpha=0.05, max_looks=10):
    # Convert to tensors if needed
    if not isinstance(stream1, torch.Tensor):
        stream1 = torch.tensor(stream1, dtype=torch.float)
    if not isinstance(stream2, torch.Tensor):
        stream2 = torch.tensor(stream2, dtype=torch.float)
    
    # Ensure streams are the same length
    min_length = min(len(stream1), len(stream2))
    stream1 = stream1[:min_length]
    stream2 = stream2[:min_length]
    
    # Calculate look times (equally spaced)
    look_times = torch.linspace(1, min_length, max_looks, dtype=torch.long)
    
    results = []
    stopped_at = None
    
    for i, look in enumerate(look_times):
        # Get data up to this look
        data1 = stream1[:look]
        data2 = stream2[:look]
        
        # Calculate test statistic
        n1 = len(data1)
        n2 = len(data2)
        mean1 = torch.mean(data1)
        mean2 = torch.mean(data2)
        var1 = torch.var(data1, unbiased=True)
        var2 = torch.var(data2, unbiased=True)
        
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Standard error
        se = math.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # T-statistic
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Two-sided p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat.item()), df))
        
        # Determine if test is significant
        significant = p_value < alpha
        
        # Store results
        results.append({
            'look': i+1,
            'sample_size': look,
            't_statistic': t_stat.item(),
            'p_value': p_value,
            'significant': significant
        })
        
        # Check if test should stop
        if significant and stopped_at is None:
            stopped_at = i+1
    
    return results, stopped_at

results, stopped_at = sequential_test(stream1, stream2, alpha=0.05, max_looks=10)

# Print sequential test results
print("Sequential Test Results:")
print("Look | Sample Size | T-statistic | P-value | Significant")
print("-" * 60)

for result in results:
    print(f"{result['look']:4} | {result['sample_size']:11} | {result['t_statistic']:12.4f} | {result['p_value']:7.4f} | {result['significant']}")

if stopped_at is not None:
    print(f"\nTest stopped at look {stopped_at} with significant result.")
else:
    print("\nTest did not stop early. No significant result detected.")
```

This sales performance tester demonstrates several hypothesis testing concepts:

1. **One-sample T-test**: Testing if sales performance differs from a target value
2. **Two-sample T-test**: Comparing sales performance between two groups
3. **ANOVA**: Comparing sales performance among multiple groups
4. **A/B testing**: A comprehensive approach to comparing two groups with effect size and confidence intervals
5. **Sequential testing**: Monitoring results as data accumulates

The project also includes data visualization capabilities and functions to simulate sales data for testing purposes.

## Summary

Hypothesis testing is a fundamental statistical tool in deep learning and data science. In this lesson, we've covered:

1. **Basic Concepts of Hypothesis Testing**: Including null and alternative hypotheses, Type I and Type II errors, and statistical significance
2. **Z-test**: A hypothesis test using the standard normal distribution, typically for large samples with known population standard deviation
3. **T-test**: A hypothesis test using the t-distribution, typically for small samples with unknown population standard deviation
4. **ANOVA**: A method for comparing means among three or more groups
5. **A/B Testing**: A practical application of hypothesis testing for comparing two versions of a product or service

We've also explored how these concepts are applied in deep learning, common problems encountered, and how advancements in the field have addressed these challenges.

The sales performance tester project provides a practical implementation of these concepts using PyTorch, demonstrating how hypothesis testing can be used to make data-driven decisions in business contexts.

Understanding hypothesis testing is crucial for anyone looking to master deep learning, as it provides the statistical foundation for evaluating model performance, comparing algorithms, and making informed decisions based on data.