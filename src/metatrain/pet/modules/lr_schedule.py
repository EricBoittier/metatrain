"""Per-group learning-rate schedules (``lr_schedule`` trainer hyper).

The same spec is the JSON file of mlip.cpp's ``--lr-schedule FILE``; the reference
:func:`group_lr` generates ``tests/resources/lr_schedule_golden.json``. Spec::

    {
        "base_lr": 1e-4,
        "min_lr": 1e-6,
        "shape": "cosine" | "linear" | "constant",
        "warmup_epochs": 1,
        "decay_after_epochs": 10,
        "groups": [
            {
                "name": "new_lr",
                "match": ["*_gate", "les."],
                "lr_mult": 1.0,
                "freeze_epochs": 0,
                "warmup_epochs": 2,
                "weight_decay": 0.0,
                "optimizer": "adam" | "muon",
            },
            ...,
        ],
        "default_group": "backbone",
    }

Semantics (``S`` steps per epoch, ``T`` steps in this fine-tune, step ``n`` counted
from 0 at its start):

- base schedule ``b(n)``: ``base_lr * min(1, (n + 1) / W)`` for ``n < W = S
  warmup_epochs`` (``W = 0``: no warmup); ``base_lr`` until ``D = max(W, S
  decay_after_epochs)``; then the shape from ``base_lr`` at ``D`` to ``min_lr`` at
  the last step ``T - 1``, with ``p = (n - D) / max(1, T - 1 - D)``: cosine ``min +
  (base - min) (1 + cos(pi p)) / 2``, linear ``base - (base - min) p``, constant
  ``base``.
- group ``g``: frozen for ``n < F_g = S freeze_epochs`` (lr 0, no update, no
  optimizer state); then ``lr = b(n) lr_mult min(1, (n - F_g + 1) / W_g)`` with
  ``W_g = S warmup_epochs_g`` (``W_g = 0``: ramp 1).
- matching on parameter (state-dict) names, first matching group wins, unmatched
  names go to ``default_group``: a pattern with ``*`` is an ``fnmatch`` glob on
  the whole name (``*_gate``), a pattern without is a substring (``gnn_layers.``).
  ``default_group`` may name a group of ``groups`` or a new one with the defaults.
- group defaults: ``lr_mult 1``, ``freeze_epochs 0``, ``warmup_epochs 0``,
  ``weight_decay 0``, ``optimizer adam``. ``muon`` applies to 2-D weights; other
  tensors of a muon group use AdamW with the same lr.
"""

import fnmatch
import math
from typing import Any, Dict, List, Tuple

import torch


GROUP_DEFAULTS: Dict[str, Any] = {
    "lr_mult": 1.0,
    "freeze_epochs": 0,
    "warmup_epochs": 0,
    "weight_decay": 0.0,
    "optimizer": "adam",
}


def normalize(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The spec with defaults filled in, and the default group added."""
    spec = {
        "min_lr": 0.0,
        "shape": "cosine",
        "warmup_epochs": 0,
        "decay_after_epochs": 0,
        "groups": [],
        "default_group": "default",
        **spec,
    }
    if spec["shape"] not in ("cosine", "linear", "constant"):
        raise ValueError(f"unknown lr_schedule shape '{spec['shape']}'")
    groups = [{**GROUP_DEFAULTS, "match": [], **g} for g in spec["groups"]]
    if spec["default_group"] not in [g["name"] for g in groups]:
        groups.append({**GROUP_DEFAULTS, "name": spec["default_group"], "match": []})
    for g in groups:
        if g["optimizer"] not in ("adam", "muon"):
            raise ValueError(f"unknown optimizer '{g['optimizer']}'")
    spec["groups"] = groups
    return spec


def matches(name: str, pattern: str) -> bool:
    """Whether ``name`` matches ``pattern`` (glob with ``*``, else substring)."""
    if "*" in pattern:
        return fnmatch.fnmatchcase(name, pattern)
    return pattern in name


def group_of(name: str, spec: Dict[str, Any]) -> str:
    """Name of the first group matching ``name``, else the default group."""
    for g in spec["groups"]:
        if any(matches(name, p) for p in g["match"]):
            return g["name"]
    return spec["default_group"]


def base_lr(spec: Dict[str, Any], step: int, steps_per_epoch: int, total: int) -> float:
    """Reference base schedule ``b(step)``."""
    b, m = float(spec["base_lr"]), float(spec["min_lr"])
    W = int(round(spec["warmup_epochs"] * steps_per_epoch))
    D = max(W, int(round(spec["decay_after_epochs"] * steps_per_epoch)))
    if step < W:
        return b * min(1.0, (step + 1) / W)
    if step < D or spec["shape"] == "constant":
        return b
    p = min(1.0, (step - D) / max(1, total - 1 - D))
    if spec["shape"] == "linear":
        return b - (b - m) * p
    return m + (b - m) * 0.5 * (1 + math.cos(math.pi * p))


def group_lr(
    spec: Dict[str, Any], group: Dict[str, Any], step: int, steps_per_epoch: int,
    total: int,
) -> Tuple[float, bool]:  # fmt: skip
    """Reference ``(lr, frozen)`` of ``group`` at ``step``."""
    F = int(round(group["freeze_epochs"] * steps_per_epoch))
    if step < F:
        return 0.0, True
    W = int(round(group["warmup_epochs"] * steps_per_epoch))
    ramp = 1.0 if W == 0 else min(1.0, (step - F + 1) / W)
    return base_lr(spec, step, steps_per_epoch, total) * group["lr_mult"] * ramp, False


def schedule_table(
    spec: Dict[str, Any], steps_per_epoch: int, num_epochs: int
) -> Dict[str, Any]:
    """``lr`` and ``frozen`` per group for every step (the golden file content)."""
    spec = normalize(spec)
    total = steps_per_epoch * num_epochs
    out: Dict[str, Any] = {"lr": {}, "frozen": {}}
    for g in spec["groups"]:
        rows = [group_lr(spec, g, n, steps_per_epoch, total) for n in range(total)]
        out["lr"][g["name"]] = [r[0] for r in rows]
        out["frozen"][g["name"]] = [r[1] for r in rows]
    return out


class GroupOptimizer:
    """One AdamW (and a Muon, if a group asks for it) behind the ``Optimizer``
    methods the trainer uses. ``param_groups`` carry ``name`` and ``frozen``; a
    frozen group's gradients are dropped, so it gets no update and no state."""

    def __init__(self, model: torch.nn.Module, spec: Dict[str, Any]) -> None:
        spec = normalize(spec)
        by_group: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
            g["name"]: [] for g in spec["groups"]
        }
        for name, p in model.named_parameters():
            if p.requires_grad:
                by_group[group_of(name, spec)].append((name, p))
        self.names = {g: [n for n, _ in ps] for g, ps in by_group.items()}
        adam: List[Dict[str, Any]] = []
        muon: List[Dict[str, Any]] = []
        for g in spec["groups"]:
            ps = [p for _, p in by_group[g["name"]]]
            common = {"name": g["name"], "frozen": False, "lr": 0.0}
            common["weight_decay"] = float(g["weight_decay"])
            if g["optimizer"] == "muon":
                mat = [p for p in ps if p.ndim == 2]
                rest = [p for p in ps if p.ndim != 2]
                if mat:
                    muon.append({**common, "params": mat})
                if rest:
                    adam.append({**common, "params": rest})
            elif ps:
                adam.append({**common, "params": ps})
        self.optimizers: List[torch.optim.Optimizer] = []
        if adam:
            self.optimizers.append(torch.optim.AdamW(adam, lr=0.0))
        if muon:
            self.optimizers.append(torch.optim.Muon(muon, lr=0.0))  # type: ignore

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def drop_frozen_grads(self) -> None:
        for g in self.param_groups:
            if g["frozen"]:
                for p in g["params"]:
                    p.grad = None

    def step(self) -> None:
        self.drop_frozen_grads()
        for o in self.optimizers:
            o.step()

    def state_dict(self) -> Dict[str, Any]:
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for o, s in zip(self.optimizers, state["optimizers"], strict=True):
            o.load_state_dict(s)


class GroupLRScheduler:
    """Sets every group's ``lr`` (and ``frozen``) from :func:`group_lr`; ``step()``
    after each optimizer step, like a ``LambdaLR``."""

    def __init__(
        self,
        optimizer: GroupOptimizer,
        spec: Dict[str, Any],
        steps_per_epoch: int,
        num_epochs: int,
        last_step: int = 0,
    ) -> None:
        self.spec = normalize(spec)
        self.groups = {g["name"]: g for g in self.spec["groups"]}
        self.optimizer = optimizer
        self.steps_per_epoch, self.total = steps_per_epoch, steps_per_epoch * num_epochs
        self.n = last_step
        self._apply()

    def _apply(self) -> None:
        for pg in self.optimizer.param_groups:
            lr, frozen = group_lr(
                self.spec, self.groups[pg["name"]], self.n, self.steps_per_epoch,
                self.total,
            )  # fmt: skip
            pg["lr"], pg["frozen"] = lr, frozen

    def step(self) -> None:
        self.n += 1
        self._apply()

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {"step": self.n}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.n = int(state["step"])
        self._apply()


GOLDEN_SPEC: Dict[str, Any] = {
    "base_lr": 1e-4,
    "min_lr": 1e-6,
    "shape": "cosine",
    "warmup_epochs": 1,
    "decay_after_epochs": 6,
    "groups": [
        {"name": "new_lr", "match": ["*_gate", "les.", "mesh.", "rope_omega",
                                     "edge_embedder.vec."],
         "lr_mult": 1.0, "warmup_epochs": 2},
        {"name": "heads", "match": ["node_heads.", "edge_heads.", "last_layers."],
         "lr_mult": 0.5, "freeze_epochs": 2, "warmup_epochs": 1,
         "weight_decay": 0.01},
        {"name": "backbone", "match": ["gnn_layers.", "embedder"], "lr_mult": 0.1},
    ],
    "default_group": "backbone",
}  # fmt: skip

GOLDEN_NAMES = [
    "backend.gnn_layers.0.trans.layers.0.attention.rope_gate",
    "backend.gnn_layers.0.trans.layers.0.attention.rope_omega",
    "backend.gnn_layers.0.trans.layers.0.attention.input_linear.weight",
    "backend.gnn_layers.1.edge_embedder.edge_gate",
    "backend.gnn_layers.1.edge_embedder.vec.weight",
    "backend.gnn_layers.1.edge_embedder.weight",
    "backend.mesh.mesh_gate",
    "backend.mesh.coupling.down.W",
    "backend.les.les_gate",
    "backend.les.charge_head.node.0.0.weight",
    "backend.node_heads.energy.0.0.weight",
    "backend.edge_last_layers.energy.0.energy.bias",
    "backend.node_embedders.0.weight",
    "backend.system_conditioning.project.2.weight",
    "backend.combination_mlps.0.0.weight",
]


def golden(steps_per_epoch: int = 5, num_epochs: int = 12) -> Dict[str, Any]:
    """Content of ``tests/resources/lr_schedule_golden.json``."""
    cases = []
    for shape in ("cosine", "linear", "constant"):
        spec = {**GOLDEN_SPEC, "shape": shape}
        cases.append(
            {
                "spec": spec,
                "steps_per_epoch": steps_per_epoch,
                "num_epochs": num_epochs,
                **schedule_table(spec, steps_per_epoch, num_epochs),
            }
        )
    spec = normalize(GOLDEN_SPEC)
    return {
        "description": __doc__,
        "cases": cases,
        "matching": {
            "spec": GOLDEN_SPEC,
            "groups": {n: group_of(n, spec) for n in GOLDEN_NAMES},
        },
    }


def write_golden(path: str, steps_per_epoch: int = 5, num_epochs: int = 12) -> None:
    """Write :func:`golden` to ``path``."""
    import json

    with open(path, "w") as f:
        json.dump(golden(steps_per_epoch, num_epochs), f, indent=1)
        f.write("\n")


def recommended_spec(base: float = 1e-4, epochs_decay: int = 10) -> Dict[str, Any]:
    """Default PET -> Plato-LR fine-tuning groups, listed before ``backbone`` since
    the new names live under ``gnn_layers.`` (first match wins):

    - ``gates`` (``*_gate``, x1, 1 warmup epoch): the only new tensors with a
      gradient at the start (the ungated terms' values); they open the new terms.
    - ``new`` (LES head, mesh, ``rope_omega``, ``edge_embedder.vec.``; x1, 2 warmup
      epochs): their gradients are proportional to the gates, so they follow.
    - ``heads`` (x0.3): re-fit the readout to the added energy terms.
    - ``backbone`` (everything else incl. conditioning; x0.1, frozen 1 epoch): keep
      the pretrained representation while the new terms fit the residual first
      (linear-probe-then-fine-tune), then adapt slowly to avoid forgetting.
    """
    return {
        "base_lr": base,
        "min_lr": base * 1e-2,
        "shape": "cosine",
        "warmup_epochs": 1,
        "decay_after_epochs": epochs_decay,
        "groups": [
            {"name": "gates", "match": ["*_gate"], "lr_mult": 1.0,
             "warmup_epochs": 1},
            {"name": "new", "match": ["les.", "mesh.", "rope_omega",
                                      "edge_embedder.vec."],
             "lr_mult": 1.0, "warmup_epochs": 2},
            {"name": "heads", "match": ["node_heads.", "edge_heads.",
                                        "last_layers."], "lr_mult": 0.3},
            {"name": "backbone", "match": [], "lr_mult": 0.1, "freeze_epochs": 1},
        ],
        "default_group": "backbone",
    }  # fmt: skip


if __name__ == "__main__":
    import sys

    write_golden(sys.argv[1] if len(sys.argv) > 1 else "lr_schedule_golden.json")
