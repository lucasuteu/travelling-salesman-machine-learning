import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class AttentionEncoder(nn.Module):
    """Simple stacked self-attention encoder.

    Inputs:
        coords: (B, N, 2)
    Outputs:
        h: (B, N, d)
        graph: (B, d)
    """

    def __init__(self, d_model=128, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(2, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
            ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))
            ln1 = nn.LayerNorm(d_model)
            ln2 = nn.LayerNorm(d_model)
            self.layers.append(nn.ModuleDict({"attn": attn, "ffn": ffn, "ln1": ln1, "ln2": ln2}))

    def forward(self, coords):
        # coords: (B, N, 2)
        h = self.embed(coords)  # (B,N,d)
        for block in self.layers:
            # Pre-LN style
            _h = block["ln1"](h)
            attn_out, _ = block["attn"](_h, _h, _h)
            h = h + attn_out
            _h2 = block["ln2"](h)
            h = h + block["ffn"](_h2)
        graph = h.mean(dim=1)
        return h, graph


class TSPDecoder(nn.Module):
    """Kool-style decoder: multi-head pointer attention, tanh-clipping and optional random start.

    Matches the Attention Model decoder from Kool et al.:
    - Projects keys/values into heads
    - Projects context to multi-head query
    - Computes per-head scaled dot-product compatibilities and averages across heads
    - Applies tanh clipping to logits to reduce extreme values
    - Supports random starts (uniform) during training
    """

    def __init__(self, d_model=128, n_heads=8, tanh_clipping=10.0, include_first=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.include_first = include_first
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # context: graph (d) + last node (d) [+ first node (d)] -> (2d or 3d)
        ctx_dim = d_model * (3 if include_first else 2)
        self.w_c = nn.Linear(ctx_dim, d_model)
        self.tanh_clipping = tanh_clipping
        self._sqrt_dk = math.sqrt(self.d_k)

    def forward(self, h, graph, decode_type="sample", random_start=True):
        # h: (B,N,d), graph: (B,d)
        B, N, d = h.shape
        device = h.device

        K = self.w_k(h).view(B, N, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (B,H,N,d_k)
        V = self.w_v(h).view(B, N, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (B,H,N,d_k) - unused, kept for completeness

        visited = torch.zeros(B, N, dtype=torch.bool, device=device)

        # choose start
        if random_start and decode_type == "sample":
            cur = torch.randint(0, N, (B,), device=device, dtype=torch.long)
        else:
            cur = torch.zeros(B, dtype=torch.long, device=device)

        tour = torch.zeros(B, N, dtype=torch.long, device=device)
        logp_sum = torch.zeros(B, device=device)

        # set first
        tour[:, 0] = cur
        visited = visited.scatter(1, cur.unsqueeze(1), True)
        if self.include_first:
            h_first = h.gather(1, cur.unsqueeze(1).unsqueeze(2).expand(-1, 1, d)).squeeze(1)

        for t in range(1, N):
            # gather embedding of current node
            h_cur = h.gather(1, cur.unsqueeze(1).unsqueeze(2).expand(-1, 1, d)).squeeze(1)  # (B,d)
            if self.include_first:
                c = torch.cat([graph, h_cur, h_first], dim=1)  # (B, 3d)
            else:
                c = torch.cat([graph, h_cur], dim=1)  # (B, 2d)
            c = self.w_c(c)  # (B,d)
            q = self.w_q(c).view(B, self.n_heads, self.d_k)  # (B,H,d_k)

            # scores per head: (B,H,N)
            # K: (B,H,N,d_k), q: (B,H,d_k)
            scores = (q.unsqueeze(2) * K).sum(-1) / self._sqrt_dk  # (B,H,N)
            scores = scores.mean(dim=1)  # average over heads -> (B,N)

            # tanh clipping as in Kool et al.
            if self.tanh_clipping is not None:
                scores = self.tanh_clipping * torch.tanh(scores)

            scores = scores.masked_fill(visited, float("-inf"))

            if decode_type == "greedy":
                nxt = scores.argmax(dim=1)
                logp = None
            else:
                dist = Categorical(logits=scores)
                nxt = dist.sample()
                logp = dist.log_prob(nxt)

            tour[:, t] = nxt
            if logp is not None:
                logp_sum = logp_sum + logp

            visited = visited.scatter(1, nxt.unsqueeze(1), True)
            cur = nxt

        return tour, logp_sum


class AttentionModel(nn.Module):
    def __init__(self, d_model=128, n_layers=2, n_heads=8, tanh_clipping=10.0, include_first=False):
        super().__init__()
        self.encoder = AttentionEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.decoder = TSPDecoder(d_model=d_model, n_heads=n_heads, tanh_clipping=tanh_clipping, include_first=include_first)

    def forward(self, coords, decode_type="sample", random_start=True):
        # coords: (B,N,2)
        h, graph = self.encoder(coords)
        tour, logp_sum = self.decoder(h, graph, decode_type=decode_type, random_start=random_start)
        return tour, logp_sum


# Helpers

def compute_tour_length(coords, tour, metric="euclidean"):
    """Compute closed-loop tour length.

    coords: (B,N,2) tensor
    tour: (B,N) long tensor of indices

    returns: lengths: (B,) tensor
    """
    B, N, d = coords.shape
    idx = tour.unsqueeze(2).expand(-1, -1, d)  # (B,N,2)
    ordered = torch.gather(coords, 1, idx)
    # pairwise distances
    diff = ordered[:, 1:] - ordered[:, :-1]
    if metric in ("euclidean", "l2"):
        seg = diff.norm(p=2, dim=2).sum(dim=1)
        last_first = (ordered[:, 0] - ordered[:, -1]).norm(p=2, dim=1)
    elif metric in ("manhattan", "l1"):
        seg = diff.abs().sum(dim=2).sum(dim=1)
        last_first = (ordered[:, 0] - ordered[:, -1]).abs().sum(dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    # closing segment
    return seg + last_first


def assert_permutation(tour):
    """Raise if any row of tour is not a permutation 0..N-1."""
    B, N = tour.shape
    s = torch.sort(tour, dim=1)[0]
    expected = torch.arange(N, device=tour.device).unsqueeze(0).expand(B, -1)
    if not torch.equal(s, expected):
        raise AssertionError("Tour is not a valid permutation")


if __name__ == "__main__":
    # tiny sanity check
    torch.manual_seed(0)
    model = AttentionModel(d_model=64, n_layers=1, n_heads=4)
    coords = torch.rand(8, 5, 2)
    # test sampling forward/backward
    tour, logp = model(coords, decode_type="sample", random_start=True)
    assert_permutation(tour)
    L = compute_tour_length(coords, tour)
    print("Sanity check (sample): tours shape", tour.shape, "lengths", L)

    # test greedy decode
    tour_g, _ = model(coords, decode_type="greedy", random_start=False)
    assert_permutation(tour_g)
    Lg = compute_tour_length(coords, tour_g)
    print("Sanity check (greedy): lengths", Lg)

    # backward pass test
    torch.manual_seed(0)
    model.train()
    tour, logp = model(coords, decode_type="sample", random_start=True)
    lengths = compute_tour_length(coords, tour)
    baseline = lengths.mean()
    adv = (lengths - baseline).detach()
    loss = (adv * logp).mean()
    loss.backward()
    print("Backward OK, loss=", loss.item())
