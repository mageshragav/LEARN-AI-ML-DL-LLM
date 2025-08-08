# Day 7 – Probability & Distributions

## What is Probability Theory and Why is it Important for Deep Learning?

Probability theory is the branch of mathematics concerned with analyzing random phenomena and quantifying uncertainty. In deep learning, probability is fundamental because:

- Real-world data is often noisy and uncertain
- Many deep learning models are probabilistic in nature
- We use probability to evaluate model performance and make predictions
- It forms the basis for many loss functions and optimization techniques

Probability distributions are mathematical functions that describe the likelihood of different outcomes. They help us model and understand the underlying patterns in data.

## Essential Subtopics

### 1. Basic Probability Concepts

**What are the fundamental concepts of probability?**
- **Random Experiment**: A process that leads to uncertain outcomes
- **Sample Space (S)**: The set of all possible outcomes
- **Event**: A subset of the sample space
- **Probability**: A value between 0 and 1 that quantifies the likelihood of an event
- **Conditional Probability**: The probability of event A given that event B has occurred, denoted P(A|B)
- **Independence**: Two events are independent if the occurrence of one doesn't affect the probability of the other

**PyTorch Implementation:**
```python
import torch

# Basic probability calculations
# Suppose we have a fair six-sided die
outcomes = torch.tensor([1, 2, 3, 4, 5, 6])
probabilities = torch.ones(6) / 6  # Equal probability for each outcome

# Probability of rolling an even number
even_numbers = outcomes % 2 == 0
prob_even = torch.sum(probabilities[even_numbers])
print(f"Probability of even number: {prob_even.item()}")

# Conditional probability example
# P(A|B) = P(A ∩ B) / P(B)
# Suppose we have two dice
die1 = torch.tensor([1, 2, 3, 4, 5, 6])
die2 = torch.tensor([1, 2, 3, 4, 5, 6])

# All possible outcomes (36 possibilities)
all_outcomes = torch.cartesian_prod(die1, die2)

# Event A: sum of dice is 7
sum_is_7 = torch.sum(all_outcomes, dim=1) == 7
prob_A = torch.sum(sum_is_7) / len(all_outcomes)

# Event B: first die shows 4
first_die_4 = all_outcomes[:, 0] == 4
prob_B = torch.sum(first_die_4) / len(all_outcomes)

# A ∩ B: sum is 7 AND first die is 4
A_and_B = sum_is_7 & first_die_4
prob_A_and_B = torch.sum(A_and_B) / len(all_outcomes)

# Conditional probability P(A|B)
prob_A_given_B = prob_A_and_B / prob_B
print(f"P(sum is 7 | first die is 4): {prob_A_given_B.item()}")
```

**When and where we use basic probability:**
- In Bayesian deep learning for updating beliefs based on new evidence
- In reinforcement learning for modeling action selection
- In generative models for sampling new data points
- In uncertainty quantification of model predictions

**Real-world examples:**
- In medical diagnosis, conditional probability helps determine the likelihood of a disease given certain symptoms
- In spam filters, probability is used to classify emails based on word occurrences
- In recommendation systems, probability models predict user preferences

### 2. Common Probability Distributions

#### Normal (Gaussian) Distribution

**What is the normal distribution?**
The normal distribution is a continuous probability distribution characterized by its bell-shaped curve. It's defined by two parameters: mean (μ) and standard deviation (σ). The normal distribution is ubiquitous in nature and statistics due to the Central Limit Theorem.

**PyTorch Implementation:**
```python
import torch
import matplotlib.pyplot as plt

# Normal distribution
mean = torch.tensor([0.0])
std = torch.tensor([1.0])

# Create a normal distribution
normal_dist = torch.distributions.Normal(mean, std)

# Sample from the distribution
samples = normal_dist.sample((1000,))

# Compute probability density function (PDF) values
x = torch.linspace(-4, 4, 100)
pdf_values = torch.exp(normal_dist.log_prob(x))

# Plot
plt.figure(figsize=(10, 6))
plt.hist(samples.numpy(), bins=30, density=True, alpha=0.6, label='Samples')
plt.plot(x.numpy(), pdf_values.numpy(), 'r-', lw=2, label='PDF')
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Multivariate normal distribution
mean_mv = torch.tensor([0.0, 0.0])
cov_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
mv_normal = torch.distributions.MultivariateNormal(mean_mv, cov_matrix)
mv_samples = mv_normal.sample((500,))
```

**When and where we use the normal distribution:**
- In modeling continuous data that clusters around a mean
- As a prior distribution in Bayesian methods
- In noise modeling for many signal processing applications
- In weight initialization for neural networks

**Real-world examples:**
- Heights and weights of people in a population
- Measurement errors in scientific experiments
- Daily returns of stock prices (approximately)

#### Bernoulli Distribution

**What is the Bernoulli distribution?**
The Bernoulli distribution is a discrete probability distribution for a random variable that takes only two possible outcomes: 1 (success) with probability p, and 0 (failure) with probability 1-p.

**PyTorch Implementation:**
```python
import torch

# Bernoulli distribution
p = torch.tensor([0.7])  # Probability of success
bernoulli_dist = torch.distributions.Bernoulli(probs=p)

# Sample from the distribution
samples = bernoulli_dist.sample((10,))
print(f"Bernoulli samples: {samples}")

# Probability mass function (PMF)
x = torch.tensor([0.0, 1.0])
pmf_values = torch.exp(bernoulli_dist.log_prob(x))
print(f"P(X=0): {pmf_values[0].item()}, P(X=1): {pmf_values[1].item()}")

# Expected value and variance
expected_value = p
variance = p * (1 - p)
print(f"Expected value: {expected_value.item()}, Variance: {variance.item()}")
```

**When and where we use the Bernoulli distribution:**
- In binary classification problems
- In modeling coin flips or yes/no decisions
- As a building block for more complex distributions like the Binomial distribution
- In dropout regularization for neural networks

**Real-world examples:**
- Whether a customer will purchase a product or not
- Whether an email is spam or not
- Whether a medical test result is positive or negative

#### Uniform Distribution

**What is the uniform distribution?**
The uniform distribution is a probability distribution where all outcomes are equally likely. For a continuous uniform distribution over the interval [a, b], the probability density function is constant.

**PyTorch Implementation:**
```python
import torch
import matplotlib.pyplot as plt

# Uniform distribution
low = torch.tensor([0.0])
high = torch.tensor([1.0])
uniform_dist = torch.distributions.Uniform(low, high)

# Sample from the distribution
samples = uniform_dist.sample((1000,))

# Compute PDF values
x = torch.linspace(-0.5, 1.5, 100)
pdf_values = torch.exp(uniform_dist.log_prob(x))

# Plot
plt.figure(figsize=(10, 6))
plt.hist(samples.numpy(), bins=30, density=True, alpha=0.6, label='Samples')
plt.plot(x.numpy(), pdf_values.numpy(), 'r-', lw=2, label='PDF')
plt.title('Uniform Distribution [0, 1]')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
```

**When and where we use the uniform distribution:**
- In random sampling and Monte Carlo methods
- As a non-informative prior in Bayesian statistics
- In generating random numbers for simulations
- In weight initialization for neural networks

**Real-world examples:**
- Rolling a fair die
- Randomly selecting a point from a line segment
- Generating random numbers for computer graphics

#### Other Important Distributions

```python
import torch
import matplotlib.pyplot as plt

# Binomial distribution
n = torch.tensor([10])  # Number of trials
p = torch.tensor([0.3])  # Probability of success
binomial_dist = torch.distributions.Binomial(n, p)
binomial_samples = binomial_dist.sample((1000,))
plt.figure(figsize=(10, 6))
plt.hist(binomial_samples.numpy(), bins=11, density=True)
plt.title('Binomial Distribution (n=10, p=0.3)')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.grid(True)
plt.show()

# Exponential distribution
rate = torch.tensor([1.0])
exponential_dist = torch.distributions.Exponential(rate)
exponential_samples = exponential_dist.sample((1000,))
plt.figure(figsize=(10, 6))
plt.hist(exponential_samples.numpy(), bins=30, density=True)
plt.title('Exponential Distribution (rate=1.0)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Poisson distribution
rate = torch.tensor([3.0])
poisson_dist = torch.distributions.Poisson(rate)
poisson_samples = poisson_dist.sample((1000,))
plt.figure(figsize=(10, 6))
plt.hist(poisson_samples.numpy(), bins=15, density=True)
plt.title('Poisson Distribution (rate=3.0)')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
```

### 3. Central Limit Theorem

**What is the Central Limit Theorem?**
The Central Limit Theorem (CLT) states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution. This is one of the most important concepts in statistics and probability.

**PyTorch Implementation:**
```python
import torch
import matplotlib.pyplot as plt

# Demonstrate Central Limit Theorem
# We'll use a non-normal distribution (exponential) as our population
population_dist = torch.distributions.Exponential(torch.tensor([1.0]))

# Sample sizes to try
sample_sizes = [1, 2, 5, 10, 30]
num_samples = 1000  # Number of sample means to compute

plt.figure(figsize=(15, 10))

for i, n in enumerate(sample_sizes):
    # Generate n*num_samples samples and reshape
    samples = population_dist.sample((n * num_samples,))
    samples = samples.reshape(num_samples, n)
    
    # Compute mean of each sample
    sample_means = torch.mean(samples, dim=1)
    
    # Plot histogram of sample means
    plt.subplot(2, 3, i+1)
    plt.hist(sample_means.numpy(), bins=30, density=True)
    plt.title(f'Sample Size = {n}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True)

plt.tight_layout()
plt.show()
```

**When and where we use the Central Limit Theorem:**
- In hypothesis testing and confidence intervals
- In quality control and manufacturing
- In financial modeling and risk assessment
- In justifying the use of normal distributions in various applications

**Real-world examples:**
- Averaging multiple noisy measurements to get a more accurate estimate
- Polling and survey analysis
- Determining the margin of error in statistical estimates

### 4. Maximum Likelihood Estimation

**What is Maximum Likelihood Estimation?**
Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a probability distribution by maximizing a likelihood function. The goal is to find the parameter values that make the observed data most probable.

**PyTorch Implementation:**
```python
import torch
import torch.optim as optim

# Generate some data from a normal distribution with known parameters
true_mean = 5.0
true_std = 2.0
data = torch.distributions.Normal(torch.tensor([true_mean]), torch.tensor([true_std])).sample((1000,))

# Define the log-likelihood function for a normal distribution
def normal_log_likelihood(data, mean, std):
    # Using torch.distributions to compute log probability
    dist = torch.distributions.Normal(mean, std)
    return torch.sum(dist.log_prob(data))

# Initialize parameters to estimate
estimated_mean = torch.tensor([0.0], requires_grad=True)
estimated_std = torch.tensor([1.0], requires_grad=True)

# Use gradient descent to maximize the log-likelihood
optimizer = optim.Adam([estimated_mean, estimated_std], lr=0.01)

num_iterations = 1000
for i in range(num_iterations):
    # We minimize the negative log-likelihood (equivalent to maximizing log-likelihood)
    loss = -normal_log_likelihood(data, estimated_mean, estimated_std)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Ensure std is positive
    with torch.no_grad():
        estimated_std.clamp_(min=0.1)
    
    if (i+1) % 100 == 0:
        print(f"Iteration {i+1}, Loss: {loss.item():.4f}, Mean: {estimated_mean.item():.4f}, Std: {estimated_std.item():.4f}")

print(f"\nTrue parameters: mean={true_mean}, std={true_std}")
print(f"Estimated parameters: mean={estimated_mean.item():.4f}, std={estimated_std.item():.4f}")
```

**When and where we use Maximum Likelihood Estimation:**
- In parameter estimation for statistical models
- In training many machine learning algorithms, including neural networks
- In fitting probability distributions to data
- In econometrics and biostatistics

**Real-world examples:**
- Estimating the parameters of a distribution that best fits observed data
- Training logistic regression models for classification
- Estimating the parameters of a time series model
- Fitting models to biological data

### 5. Bayesian Inference

**What is Bayesian Inference?**
Bayesian inference is a method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available. It combines prior beliefs with observed data to obtain posterior beliefs.

**Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B)**

**PyTorch Implementation:**
```python
import torch
import matplotlib.pyplot as plt

# Example: Coin flip problem
# We want to estimate the probability of heads (p) given some observed data

# Prior belief: p follows a Beta distribution (conjugate prior for binomial)
# Beta(α, β) where α and β are hyperparameters
alpha_prior = 2
beta_prior = 2
prior_dist = torch.distributions.Beta(torch.tensor([alpha_prior]), torch.tensor([beta_prior]))

# Observed data: 7 heads out of 10 flips
num_heads = 7
num_flips = 10

# Likelihood: Binomial distribution
def likelihood(p, heads, flips):
    # Binomial probability mass function
    binomial_coeff = torch.math.factorial(flips) / (torch.math.factorial(heads) * torch.math.factorial(flips - heads))
    return binomial_coeff * (p ** heads) * ((1 - p) ** (flips - heads))

# Posterior is proportional to prior × likelihood
# For Beta-Binomial conjugate pair, posterior is Beta(α + heads, β + tails)
alpha_posterior = alpha_prior + num_heads
beta_posterior = beta_prior + (num_flips - num_heads)
posterior_dist = torch.distributions.Beta(torch.tensor([alpha_posterior]), torch.tensor([beta_posterior]))

# Plot prior, likelihood, and posterior
p_values = torch.linspace(0, 1, 100)

# Prior
prior_pdf = torch.exp(prior_dist.log_prob(p_values))

# Likelihood (normalized)
likelihood_values = likelihood(p_values, num_heads, num_flips)
likelihood_values = likelihood_values / torch.sum(likelihood_values)

# Posterior
posterior_pdf = torch.exp(posterior_dist.log_prob(p_values))

plt.figure(figsize=(12, 6))
plt.plot(p_values.numpy(), prior_pdf.numpy(), 'b-', label='Prior Beta(2,2)')
plt.plot(p_values.numpy(), likelihood_values.numpy(), 'g-', label='Likelihood (7 heads in 10 flips)')
plt.plot(p_values.numpy(), posterior_pdf.numpy(), 'r-', label='Posterior Beta(9,5)')
plt.title('Bayesian Inference for Coin Flip Problem')
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Density/Probability')
plt.legend()
plt.grid(True)
plt.show()

# Maximum a posteriori (MAP) estimate
map_estimate = (alpha_posterior - 1) / (alpha_posterior + beta_posterior - 2)
print(f"MAP estimate of p: {map_estimate.item():.4f}")
```

**When and where we use Bayesian Inference:**
- In situations with limited data where prior knowledge is valuable
- In uncertainty quantification of model predictions
- In model selection and comparison
- In reinforcement learning for policy optimization

**Real-world examples:**
- Spam filtering that updates its beliefs based on user feedback
- Medical diagnosis that combines prior knowledge with test results
- A/B testing in web design and marketing
- Climate modeling that incorporates historical data

## Applications in Deep Learning

### 1. Probabilistic Neural Networks

Neural networks can be designed to output probability distributions rather than single values:

```python
import torch
import torch.nn as nn

class ProbabilisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(hidden_dim, 1)  # Output mean
        self.fc_std = nn.Linear(hidden_dim, 1)   # Output standard deviation
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        mean = self.fc_mean(x)
        # Ensure std is positive
        std = torch.exp(self.fc_std(x))  # Exponential ensures positivity
        return mean, std

# Example usage
model = ProbabilisticRegressionModel(input_dim=10, hidden_dim=64)
x = torch.randn(32, 10)  # Batch of 32 samples with 10 features
mean, std = model(x)

# Create a normal distribution for each sample
dist = torch.distributions.Normal(mean, std)
# Sample from the distribution
samples = dist.sample()
# Compute log probability
log_prob = dist.log_prob(torch.randn_like(mean))
```

### 2. Dropout as Bayesian Approximation

Dropout can be interpreted as a Bayesian approximation:

```python
import torch
import torch.nn as nn

class BayesianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, apply_dropout=True):
        x = self.fc1(x)
        x = torch.relu(x)
        if apply_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict_with_uncertainty(self, x, n_samples=100):
        # Make multiple predictions with dropout enabled
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x, apply_dropout=True)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        return mean, std

# Example usage
model = BayesianNet(input_dim=10, hidden_dim=64, output_dim=1)
x = torch.randn(1, 10)  # Single input sample

# Get prediction with uncertainty
mean, std = model.predict_with_uncertainty(x)
print(f"Prediction: {mean.item():.4f} ± {std.item():.4f}")
```

### 3. Variational Autoencoders (VAEs)

VAEs use probability distributions to model latent variables:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent distribution
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick to allow backpropagation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # Sigmoid for binary data
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (binary cross entropy)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence between learned distribution and standard normal
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD

# Example usage
vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
x = torch.bernoulli(torch.rand(64, 784))  # Random binary data (like MNIST)
recon_x, mu, logvar = vae(x)
loss = vae.loss_function(recon_x, x, mu, logvar)
```

## Common Problems and Advancements

### 1. Problem: Vanishing/Exploding Gradients in Probabilistic Models

**Issue:**
When working with probability distributions, especially in deep generative models, we often compute log probabilities to avoid numerical underflow. However, this can lead to vanishing or exploding gradients during training.

**How advancements solved this:**
- **Numerical stabilization techniques**: Using log-sum-exp trick for stable computations
- **Gradient clipping**: Limiting the maximum value of gradients
- **Careful architecture design**: Using skip connections, batch normalization, etc.
- **Advanced optimization algorithms**: Adam, RMSprop, etc., that adapt learning rates

```python
# Example of log-sum-exp trick for numerical stability
def log_sum_exp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = torch.max(x, dim=-1, keepdim=True)[0]
    return max_x + torch.log(torch.sum(torch.exp(x - max_x), dim=-1, keepdim=True))

# Example usage
x = torch.randn(3, 4)
result = log_sum_exp(x)
print(result)
```

### 2. Problem: Mode Collapse in Generative Models

**Issue:**
In generative models like GANs and VAEs, the model might only capture a subset of the modes in the true data distribution, ignoring other possible modes.

**How advancements solved this:**
- **Improved architectures**: Using Wasserstein GANs, VAE-GAN hybrids
- **Better regularization**: Techniques like minibatch discrimination, feature matching
- **Advanced training strategies**: Using multiple generators, curriculum learning
- **Alternative divergence measures**: Using f-divergences, integral probability metrics

```python
# Example of minibatch discrimination in a GAN discriminator
class MinibatchDiscrimination(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_dim=16):
        super().__init__()
        self.T = nn.Parameter(torch.randn(input_dim, output_dim, kernel_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # T shape: (input_dim, output_dim, kernel_dim)
        
        # Compute M: (batch_size, output_dim, kernel_dim)
        M = torch.einsum('bi,ioj->boj', x, self.T)
        
        # Compute pairwise distances between all samples in the batch
        # Result shape: (batch_size, batch_size, output_dim)
        diffs = M.unsqueeze(1) - M.unsqueeze(0)
        abs_diffs = torch.sum(torch.abs(diffs), dim=-1)
        
        # Sum over all other samples in the batch
        # Result shape: (batch_size, output_dim)
        o = torch.sum(torch.exp(-abs_diffs), dim=1) + self.b.unsqueeze(0)
        
        return o

# Usage in a discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.minibatch_discrim = MinibatchDiscrimination(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.minibatch_discrim(x)
        x = self.fc2(x)
        return x
```

### 3. Problem: Intractable Posterior Distributions

**Issue:**
In Bayesian neural networks and other probabilistic models, the true posterior distribution over parameters is often intractable to compute exactly, making inference challenging.

**How advancements solved this:**
- **Variational inference**: Approximating the true posterior with a simpler, tractable distribution
- **Markov Chain Monte Carlo (MCMC)**: Sampling-based methods for approximating the posterior
- **Expectation Propagation**: Iterative refinement of approximations
- **Normalizing flows**: Using invertible transformations to create more flexible approximate distributions

```python
# Example of a simple normalizing flow
class PlanarFlow(nn.Module):
    """Planar flow as defined in Variational Inference with Normalizing Flows"""
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
        
    def forward(self, z):
        # z shape: (batch_size, dim)
        
        # Compute the flow transformation
        # f(z) = z + u * tanh(w^T * z + b)
        linear = torch.matmul(z, self.w.unsqueeze(1)) + self.b
        h = torch.tanh(linear)
        f_z = z + self.u * h
        
        # Compute the log determinant of the Jacobian
        # log|det J| = log|1 + u^T * diag(1-tanh^2(w^T * z + b)) * w|
        psi = (1 - h**2) * self.w
        log_det_jacobian = torch.log(torch.abs(1 + torch.matmul(psi.unsqueeze(1), self.u.unsqueeze(0)).squeeze()))
        
        return f_z, log_det_jacobian

# Example usage
flow = PlanarFlow(dim=2)
z = torch.randn(10, 2)  # Batch of 10 samples from a 2D distribution
f_z, log_det = flow(z)
```

### 4. Problem: Uncertainty Calibration

**Issue:**
Probabilistic models often produce poorly calibrated uncertainty estimates, where the predicted confidence doesn't match the actual accuracy.

**How advancements solved this:**
- **Temperature scaling**: Adjusting the temperature of the softmax to improve calibration
- **Bayesian neural networks**: Modeling uncertainty in weights
- **Ensemble methods**: Combining predictions from multiple models
- **Direct probability estimation**: Methods that directly optimize for calibration

```python
# Example of temperature scaling for calibration
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=-1)

# Example usage
# Suppose we have a trained model and validation data
model = ...  # Pre-trained model
val_logits = model(val_data)
val_labels = ...  # True labels

# Train temperature scaling on validation data
temp_scaling = TemperatureScaling()
optimizer = torch.optim.LBFGS([temp_scaling.temperature], lr=0.01)

def loss_fn():
    probs = temp_scaling(val_logits)
        loss = F.cross_entropy(probs, val_labels)
        return loss

optimizer.step(loss_fn)

# Now use temperature scaling for better calibrated predictions
test_logits = model(test_data)
calibrated_probs = temp_scaling(test_logits)
```

## Project: Dice & Card Simulator

Now, let's implement a dice and card simulator using PyTorch that demonstrates the probability concepts we've covered:

```python
import torch
import matplotlib.pyplot as plt

class ProbabilitySimulator:
    def __init__(self):
        pass
    
    def roll_die(self, num_sides=6, num_rolls=1000):
        """Simulate rolling a die with given number of sides"""
        # Each outcome has equal probability (uniform distribution)
        outcomes = torch.arange(1, num_sides + 1)
        probabilities = torch.ones(num_sides) / num_sides
        
        # Sample from the distribution
        rolls = torch.multinomial(probabilities, num_rolls, replacement=True) + 1
        
        # Count occurrences of each outcome
        counts = torch.bincount(rolls, minlength=num_sides + 1)[1:]
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(outcomes.numpy(), counts.numpy(), alpha=0.7)
        plt.title(f'Results of Rolling a {num_sides}-Sided Die {num_rolls} Times')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.xticks(outcomes.numpy())
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
        
        # Calculate empirical probabilities
        empirical_probs = counts / num_rolls
        print(f"Theoretical probabilities: {probabilities}")
        print(f"Empirical probabilities: {empirical_probs}")
        
        return rolls
    
    def roll_multiple_dice(self, num_dice=2, num_sides=6, num_rolls=1000):
        """Simulate rolling multiple dice and sum the results"""
        # Roll each die
        all_rolls = []
        for _ in range(num_dice):
            outcomes = torch.arange(1, num_sides + 1)
            probabilities = torch.ones(num_sides) / num_sides
            rolls = torch.multinomial(probabilities, num_rolls, replacement=True) + 1
            all_rolls.append(rolls)
        
        # Sum the results
        sums = torch.stack(all_rolls).sum(dim=0)
        
        # Count occurrences of each sum
        min_sum = num_dice
        max_sum = num_dice * num_sides
        counts = torch.bincount(sums, minlength=max_sum + 1)[min_sum:max_sum + 1]
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(torch.arange(min_sum, max_sum + 1).numpy(), counts.numpy(), alpha=0.7)
        plt.title(f'Results of Rolling {num_dice} {num_sides}-Sided Dice {num_rolls} Times')
        plt.xlabel('Sum')
        plt.ylabel('Count')
        plt.xticks(torch.arange(min_sum, max_sum + 1).numpy())
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
        
        # Calculate empirical probabilities
        empirical_probs = counts / num_rolls
        print(f"Empirical probabilities: {empirical_probs}")
        
        return sums
    
    def draw_cards(self, num_cards=5, num_draws=1000, replacement=False):
        """Simulate drawing cards from a standard deck"""
        # Create a deck of cards (52 cards)
        deck_size = 52
        card_values = torch.arange(1, deck_size + 1)
        
        # Each card has equal probability
        probabilities = torch.ones(deck_size) / deck_size
        
        # Draw cards
        all_draws = []
        for _ in range(num_draws):
            if replacement:
                # With replacement: each draw is independent
                draws = torch.multinomial(probabilities, num_cards, replacement=True) + 1
            else:
                # Without replacement: cards are not returned to the deck
                draws = torch.multinomial(probabilities, num_cards, replacement=False) + 1
            all_draws.append(draws)
        
        all_draws = torch.stack(all_draws)
        
        # Calculate sum of card values for each draw
        sums = all_draws.sum(dim=1)
        
        # Plot histogram of sums
        plt.figure(figsize=(10, 6))
        plt.hist(sums.numpy(), bins=30, alpha=0.7, density=True)
        plt.title(f'Sum of {num_cards} Cards Drawn {num_draws} Times ({"With" if replacement else "Without"} Replacement)')
        plt.xlabel('Sum of Card Values')
        plt.ylabel('Density')
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()
        
        # Calculate mean and variance
        mean = torch.mean(sums.float())
        variance = torch.var(sums.float())
        print(f"Mean sum: {mean.item():.2f}")
        print(f"Variance of sum: {variance.item():.2f}")
        
        return all_draws
    
    def demonstrate_clt(self, distribution="exponential", sample_size=30, num_samples=1000):
        """Demonstrate the Central Limit Theorem"""
        # Generate data from a non-normal distribution
        if distribution == "exponential":
            # Exponential distribution
            data = torch.distributions.Exponential(torch.tensor([1.0])).sample((num_samples * sample_size,))
        elif distribution == "uniform":
            # Uniform distribution
            data = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample((num_samples * sample_size,))
        elif distribution == "bernoulli":
            # Bernoulli distribution
            data = torch.distributions.Bernoulli(torch.tensor([0.3])).sample((num_samples * sample_size,))
        else:
            raise ValueError("Unsupported distribution")
        
        # Reshape and calculate sample means
        data = data.reshape(num_samples, sample_size)
        sample_means = torch.mean(data.float(), dim=1)
        
        # Plot histogram of sample means
        plt.figure(figsize=(10, 6))
        plt.hist(sample_means.numpy(), bins=30, alpha=0.7, density=True)
        plt.title(f'Central Limit Theorem Demonstration ({distribution} distribution)')
        plt.xlabel('Sample Mean')
        plt.ylabel('Density')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Overlay normal distribution for comparison
        x = torch.linspace(torch.min(sample_means), torch.max(sample_means), 100)
        normal_dist = torch.distributions.Normal(torch.mean(sample_means), torch.std(sample_means))
        plt.plot(x.numpy(), torch.exp(normal_dist.log_prob(x)).numpy(), 'r-', lw=2, label='Normal Distribution')
        plt.legend()
        plt.show()
        
        return sample_means

# Example usage
simulator = ProbabilitySimulator()

# Roll a single die
print("Rolling a single die:")
rolls = simulator.roll_die(num_sides=6, num_rolls=1000)

# Roll multiple dice
print("\nRolling multiple dice:")
sums = simulator.roll_multiple_dice(num_dice=2, num_sides=6, num_rolls=1000)

# Draw cards with replacement
print("\nDrawing cards with replacement:")
draws_with_replacement = simulator.draw_cards(num_cards=5, num_draws=1000, replacement=True)

# Draw cards without replacement
print("\nDrawing cards without replacement:")
draws_without_replacement = simulator.draw_cards(num_cards=5, num_draws=1000, replacement=False)

# Demonstrate Central Limit Theorem
print("\nDemonstrating Central Limit Theorem with exponential distribution:")
sample_means = simulator.demonstrate_clt(distribution="exponential", sample_size=30, num_samples=1000)
```

This probability simulator demonstrates several key concepts:

1. **Uniform Distribution**: Rolling a fair die where each outcome has equal probability
2. **Sum of Random Variables**: Rolling multiple dice and observing the distribution of sums
3. **Sampling with and without Replacement**: Drawing cards from a deck
4. **Central Limit Theorem**: Demonstrating how sample means approach a normal distribution regardless of the original distribution

## Summary

Probability and distributions are fundamental concepts in deep learning. In this lesson, we've covered:

1. **Basic Probability Concepts**: Including conditional probability and independence
2. **Common Probability Distributions**: Normal, Bernoulli, Uniform, Binomial, Exponential, and Poisson
3. **Central Limit Theorem**: A fundamental concept explaining why normal distributions are ubiquitous
4. **Maximum Likelihood Estimation**: A method for estimating distribution parameters
5. **Bayesian Inference**: A framework for updating beliefs based on evidence

We've also explored how these concepts are applied in deep learning, common problems encountered, and how advancements in the field have addressed these challenges.

The probability simulator project provides a practical implementation of these concepts using PyTorch, demonstrating how probability distributions behave in practice and how the Central Limit Theorem emerges from random processes.

Understanding probability and distributions is crucial for anyone looking to master deep learning, as it provides the mathematical foundation for understanding uncertainty, model behavior, and the principles behind many deep learning algorithms.