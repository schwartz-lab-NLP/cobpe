from __future__ import annotations

from dataclasses import dataclass


LR_SCHEDULE_LINEAR_WARMDOWN = "linear_warmdown"
LR_SCHEDULE_DEEPSEEK_80_10_10 = "deepseek_80_10_10"
LR_SCHEDULE_CHOICES = (
    LR_SCHEDULE_LINEAR_WARMDOWN,
    LR_SCHEDULE_DEEPSEEK_80_10_10,
)

DEEPSEEK_STAGE1_END_RATIO = 0.80
DEEPSEEK_STAGE2_END_RATIO = 0.90
DEEPSEEK_STAGE2_LR_FRAC = 10 ** -0.5
DEEPSEEK_STAGE3_LR_FRAC = 0.10
DEFAULT_MUON_FINAL_MOMENTUM = 0.90


@dataclass(frozen=True)
class LRScheduleInfo:
    lr_schedule: str
    num_iterations: int
    warmup_iters: int
    warmdown_iters: int
    final_lr_frac: float
    fixed_lr_start_step: int
    fixed_lr_end_step: int
    decay_step_1: int | None = None
    decay_step_2: int | None = None
    decay_step_1_lr_frac: float | None = None
    decay_step_2_lr_frac: float | None = None


def resolve_warmup_iters(num_iterations: int, warmup_steps: int, warmup_ratio: float = -1.0) -> int:
    if warmup_steps >= 0:
        return int(warmup_steps)
    if warmup_ratio >= 0:
        return round(float(warmup_ratio) * int(num_iterations))
    return 0


def build_lr_schedule_info(
    *,
    lr_schedule: str,
    num_iterations: int,
    warmup_steps: int,
    warmup_ratio: float = -1.0,
    warmdown_ratio: float = 0.0,
    final_lr_frac: float = 0.0,
) -> LRScheduleInfo:
    num_iterations = int(num_iterations)
    warmup_iters = resolve_warmup_iters(
        num_iterations=num_iterations,
        warmup_steps=int(warmup_steps),
        warmup_ratio=float(warmup_ratio),
    )
    fixed_lr_start_step = warmup_iters

    if lr_schedule == LR_SCHEDULE_LINEAR_WARMDOWN:
        warmdown_iters = round(float(warmdown_ratio) * num_iterations)
        fixed_lr_end_step = num_iterations - warmdown_iters
        return LRScheduleInfo(
            lr_schedule=lr_schedule,
            num_iterations=num_iterations,
            warmup_iters=warmup_iters,
            warmdown_iters=warmdown_iters,
            final_lr_frac=float(final_lr_frac),
            fixed_lr_start_step=fixed_lr_start_step,
            fixed_lr_end_step=fixed_lr_end_step,
        )

    if lr_schedule == LR_SCHEDULE_DEEPSEEK_80_10_10:
        decay_step_1 = round(DEEPSEEK_STAGE1_END_RATIO * num_iterations)
        decay_step_2 = round(DEEPSEEK_STAGE2_END_RATIO * num_iterations)
        decay_step_1 = max(warmup_iters, min(decay_step_1, num_iterations))
        decay_step_2 = max(decay_step_1, min(decay_step_2, num_iterations))
        return LRScheduleInfo(
            lr_schedule=lr_schedule,
            num_iterations=num_iterations,
            warmup_iters=warmup_iters,
            warmdown_iters=0,
            final_lr_frac=DEEPSEEK_STAGE3_LR_FRAC,
            fixed_lr_start_step=fixed_lr_start_step,
            fixed_lr_end_step=decay_step_1,
            decay_step_1=decay_step_1,
            decay_step_2=decay_step_2,
            decay_step_1_lr_frac=DEEPSEEK_STAGE2_LR_FRAC,
            decay_step_2_lr_frac=DEEPSEEK_STAGE3_LR_FRAC,
        )

    raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")


def get_lr_multiplier_for_step(it: int, schedule: LRScheduleInfo) -> float:
    if it < schedule.warmup_iters:
        return (it + 1) / schedule.warmup_iters

    if schedule.lr_schedule == LR_SCHEDULE_LINEAR_WARMDOWN:
        if schedule.warmdown_iters <= 0:
            return 1.0
        if it <= schedule.num_iterations - schedule.warmdown_iters:
            return 1.0
        progress = (schedule.num_iterations - it) / schedule.warmdown_iters
        return progress * 1.0 + (1 - progress) * schedule.final_lr_frac

    if schedule.lr_schedule == LR_SCHEDULE_DEEPSEEK_80_10_10:
        assert schedule.decay_step_1 is not None
        assert schedule.decay_step_2 is not None
        assert schedule.decay_step_1_lr_frac is not None
        assert schedule.decay_step_2_lr_frac is not None
        if it <= schedule.decay_step_1:
            return 1.0
        if it <= schedule.decay_step_2:
            return schedule.decay_step_1_lr_frac
        return schedule.decay_step_2_lr_frac

    raise ValueError(f"Unsupported lr_schedule: {schedule.lr_schedule}")


def get_muon_momentum_for_step(
    it: int,
    schedule: LRScheduleInfo,
    *,
    muon_final_momentum: float = DEFAULT_MUON_FINAL_MOMENTUM,
) -> float:
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97

    if schedule.lr_schedule == LR_SCHEDULE_LINEAR_WARMDOWN:
        if schedule.warmdown_iters <= 0:
            return 0.97
        warmdown_start = schedule.num_iterations - schedule.warmdown_iters
        if it >= warmdown_start:
            progress = (it - warmdown_start) / schedule.warmdown_iters
            return 0.97 * (1 - progress) + float(muon_final_momentum) * progress
        return 0.97

    if schedule.lr_schedule == LR_SCHEDULE_DEEPSEEK_80_10_10:
        return 0.97

    raise ValueError(f"Unsupported lr_schedule: {schedule.lr_schedule}")
