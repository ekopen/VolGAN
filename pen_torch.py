import torch
import torch.optim as optim
import pandas as pd
import Inputs

yield_curve = pd.read_excel("data/usd_sofr_curve_full.xlsx")
forward_swap_tensor = pd.read_excel("data/forward_sofr_swap_full_NEW.xlsx")
tm_tensor = pd.read_excel("data/forward_sofr_swap_full_NEW.xlsx") 

def ns_func(t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    lam = theta[3]
    factor = (1 - torch.exp(-t / lam)) / (t / lam)
    return theta[0] + (theta[1] + theta[2]) * factor - theta[2] * torch.exp(-t / lam)

def ns_fit(times: torch.Tensor, rates: torch.Tensor, lr: float = 1e-2, n_iter: int = 1000) -> torch.Tensor:
    theta = torch.tensor([rates.mean(), -1.0, 1.0, 2.0], device=times.device, dtype=times.dtype, requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)
    for _ in range(n_iter):
        optimizer.zero_grad()
        pred = ns_func(times, theta)
        loss = ((pred - rates)**2).mean()
        loss.backward()
        optimizer.step()
        theta.data[3] = torch.clamp(theta.data[3], min=1e-3)
    return theta.detach()

def option_price_batch(forward: torch.Tensor,
                       strike: torch.Tensor,
                       vol: torch.Tensor,
                       maturity: torch.Tensor,
                       tenor: torch.Tensor,
                       ns_params: torch.Tensor,
                       delta: float = 0.25) -> torch.Tensor:
    d = (forward - strike) / (vol * torch.sqrt(maturity))
    normal = torch.distributions.Normal(torch.tensor(0.0, device=forward.device), torch.tensor(1.0, device=forward.device))
    cdf_d = normal.cdf(d)
    pdf_d = torch.exp(normal.log_prob(d))
    base_term = (forward - strike) * cdf_d + vol * torch.sqrt(maturity) * pdf_d
    steps = (tenor / delta).long()
    batch_size, n_inst = forward.shape
    max_steps = int(steps.max().item())
    j_vec = torch.arange(1, max_steps + 1, device=forward.device, dtype=forward.dtype)
    time_grid = maturity.unsqueeze(2) + delta * j_vec.view(1, 1, -1)
    valid_mask = (torch.arange(1, max_steps + 1, device=forward.device).view(1, 1, -1).expand(batch_size, n_inst, max_steps) <= steps.unsqueeze(2))
    r_interp = ns_func(time_grid, ns_params)
    discount_factors = torch.exp(-r_interp * time_grid)
    discount_sum = (discount_factors * valid_mask.to(discount_factors.dtype)).sum(dim=2)
    price = discount_sum * base_term
    return price

def arbitrage_penalty_batch(price_grids: torch.Tensor) -> torch.Tensor:
    batch = price_grids.shape[0]
    viol_down = price_grids[:, 1:, :] < price_grids[:, :-1, :]
    viol_right = price_grids[:, :, 1:] < price_grids[:, :, :-1]
    viol_combined = torch.cat([viol_down.view(batch, -1), viol_right.view(batch, -1)], dim=1)
    penalty = viol_combined.float().mean(dim=1)
    return penalty
