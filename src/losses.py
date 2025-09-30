import torch

def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

def geodesic_quat_loss_sign_invariant(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    qp = quat_normalize(q_pred)
    qg = quat_normalize(q_gt)
    dot = torch.sum(qp * qg, dim=-1)
    sign = torch.sign(dot).unsqueeze(-1)
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
    qp_aligned = qp * sign
    dot2 = (qp_aligned * qg).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    ang = 2.0 * torch.acos(dot2)
    return torch.where(torch.isfinite(ang), ang, torch.zeros_like(ang)).mean()