import torch
import torch.optim as optim
import pandas as pd
import Inputs

yield_curve = pd.read_excel("data/usd_sofr_curve_full.xlsx")
forward_swap_tensor = pd.read_excel("data/forward_sofr_swap_full_NEW.xlsx")
tm_tensor = pd.read_excel("data/forward_sofr_swap_full_NEW.xlsx") 

def ns_func(t, theta0, theta1, theta2, lambda1):
    return theta0 + (theta1 + theta2) * (1 - torch.exp(-t / lambda1)) / (t / lambda1) - theta2 * torch.exp(-t / lambda1)

def ns_fit(times: torch.Tensor, rates: torch.Tensor, lr: float = 1e-2, n_iter: int = 1000) -> torch.Tensor:
    theta = torch.tensor([rates.mean(), -1.0, 1.0, 2.0], device=times.device, dtype=times.dtype, requires_grad=True)
    optimizer = optim.Adam([theta], lr=lr)
    for _ in range(n_iter):
        optimizer.zero_grad()
        pred = ns_func(times, theta[0], theta[1], theta[2], theta[3])
        loss = ((pred - rates) ** 2).mean()
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
    normal = torch.distributions.Normal(torch.tensor(0.0, device=forward.device),
                                          torch.tensor(1.0, device=forward.device))
    cdf_d = normal.cdf(d)
    pdf_d = torch.exp(normal.log_prob(d))
    base_term = (forward - strike) * cdf_d + vol * torch.sqrt(maturity) * pdf_d
    steps = (tenor / delta).long()
    batch, n_inst = forward.shape
    max_steps = int(steps.max().item())
    j_vec = torch.arange(1, max_steps + 1, device=forward.device, dtype=forward.dtype)
    time_grid = maturity.unsqueeze(2) + delta * j_vec.view(1, 1, -1)
    valid_mask = (torch.arange(1, max_steps + 1, device=forward.device).view(1, 1, -1)
                  <= steps.unsqueeze(2)).to(forward.dtype)
    r_interp = ns_func(time_grid, ns_params[0], ns_params[1], ns_params[2], ns_params[3])
    discount = torch.exp(-r_interp * time_grid)
    discount_sum = (discount * valid_mask).sum(dim=2)
    price = base_term * discount_sum
    return price

def arbitrage_penalty_batch(price_grids: torch.Tensor) -> torch.Tensor:
    viol_calendar = (price_grids[:, 1:, :] < price_grids[:, :-1, :]).float()
    viol_tenor = (price_grids[:, :, 1:] < price_grids[:, :, :-1]).float()
    total_viol = torch.cat([viol_calendar.view(price_grids.size(0), -1),
                              viol_tenor.view(price_grids.size(0), -1)], dim=1)
    penalty = total_viol.mean(dim=1)
    return penalty
