import torch
## implement distribution related functions and classes


# Diffused Gaussian Mixture Model
class DiffusedGaussianMixture:
    def __init__(self, means, vars, weights):
        """
        means: (K, d) tensor
        vars: (K,) tensor — standard deviations for each component (scalar * I)
        weights: (K,) tensor
        """
        self.device = means.device
        self.d = means.shape[1]
        self.K = means.shape[0]

        self.means = means.to(self.device)       # (K, d)
        self.vars = vars.to(self.device)     # (K,)
        self.weights = weights.to(self.device)   # (K,)

    def p_t(self, x, t, alpha_t):
        """Computes log probability density p_t(x) for the diffused GMM."""
        batch_size, d = x.shape

        alpha_t_exp = alpha_t.view(-1, 1, 1)  # (batch_size, 1, 1)
        mean_t = alpha_t_exp * self.means.unsqueeze(0)  # (batch_size, K, d)

        # Compute diffused variance: scalar * I => variance vector of shape (batch_size, K, 1)
        vars_t = (1 - alpha_t.view(-1, 1) ** 2) + (alpha_t.view(-1, 1) ** 2) * self.vars.view(1, -1)  # (batch_size, K)
        vars_t = vars_t.unsqueeze(-1)  # (batch_size, K, 1)

        x_exp = x.unsqueeze(1)  # (batch_size, 1, d)
        diffs = x_exp - mean_t  # (batch_size, K, d)

        exponent = torch.sum(diffs ** 2, dim=-1, keepdim=True) / vars_t  # (batch_size, K, 1)
        log_det = d * torch.log(vars_t)  # (batch_size, K, 1)

        log_probs = -0.5 * (log_det + exponent + d * torch.log(torch.tensor(2 * torch.pi, device=self.device)))  # (batch_size, K, 1)
        log_probs = log_probs.squeeze(-1) + torch.log(self.weights.view(1, -1))  # (batch_size, K)

        return torch.logsumexp(log_probs, dim=1)  # (batch_size,)

    def density(self, x):
        """
        Compute the density p(x) of the original (non-diffused) GMM.

        Parameters:
            x: (batch_size, d) tensor

        Returns:
            densities: (batch_size,) tensor with p(x) for each input point
        """
        batch_size, d = x.shape
        x_exp = x.unsqueeze(1).to(self.device)  # (batch_size, 1, d)
        means = self.means.unsqueeze(0).to(self.device)  # (1, K, d)
        diffs = x_exp - means  # (batch_size, K, d)

        vars = self.vars.view(1, -1, 1)  # (1, K, 1)
        exponent = -0.5 * torch.sum(diffs ** 2 / vars, dim=-1)  # (batch_size, K)

        norm_const = torch.sqrt((2 * torch.pi) ** d * torch.prod(vars, dim=-1))  # (1, K)
        gaussians = torch.exp(exponent) / norm_const  # (batch_size, K)

        weighted = gaussians * self.weights.view(1, -1)  # (batch_size, K)
        return weighted.sum(dim=1)  # (batch_size,)
    
    def score(self, x, t, alpha_t):
        x.requires_grad_(True)
        log_p = self.p_t(x, t, alpha_t)
        score_x = torch.autograd.grad(log_p.sum(), x, create_graph=False)[0]
        x.requires_grad_(False)
        return score_x

    def joint_log_prob_grad(self, x_t, x0, alpha_t):
        sigma_t = torch.sqrt(1 - alpha_t ** 2)
        sigma_t = torch.clamp(sigma_t, min=1e-5)
        grad_log_p_0t = - (1 / sigma_t ** 2) * (x_t - alpha_t * x0)
        return grad_log_p_0t

    def sample(self, n_samples):
        component_ids = torch.multinomial(self.weights, n_samples, replacement=True)  # (n_samples,)
        selected_means = self.means[component_ids]  # (n_samples, d)
        selected_vars = self.vars[component_ids].view(-1, 1)  # (n_samples, 1)
        noise = torch.randn_like(selected_means) * torch.sqrt(selected_vars)  # (n_samples, d)
        return selected_means + noise

    def log_likelihood(self, x):
        """
        Computes the log-likelihood log p(x) of the sample under the original GMM.

        Parameters:
            x: (batch_size, d) tensor

        Returns:
            log_probs: (batch_size,) tensor with log p(x) for each sample
        """
        # Get p(x)
        px = self.density(x)

        # Avoid log(0) by clamping
        px = torch.clamp(px, min=1e-12)

        return torch.log(px)
 