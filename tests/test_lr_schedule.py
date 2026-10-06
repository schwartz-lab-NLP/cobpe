from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanochat.lr_schedule import (
    DEEPSEEK_STAGE2_LR_FRAC,
    DEEPSEEK_STAGE3_LR_FRAC,
    LR_SCHEDULE_DEEPSEEK_80_10_10,
    LR_SCHEDULE_LINEAR_WARMDOWN,
    build_lr_schedule_info,
    get_lr_multiplier_for_step,
    get_muon_momentum_for_step,
)


def test_linear_schedule_unchanged_warmup_and_warmdown():
    schedule = build_lr_schedule_info(
        lr_schedule=LR_SCHEDULE_LINEAR_WARMDOWN,
        num_iterations=1000,
        warmup_steps=40,
        warmdown_ratio=0.65,
        final_lr_frac=0.05,
    )
    assert schedule.warmup_iters == 40
    assert schedule.warmdown_iters == 650
    assert schedule.fixed_lr_start_step == 40
    assert schedule.fixed_lr_end_step == 350
    assert get_lr_multiplier_for_step(0, schedule) == 1 / 40
    assert get_lr_multiplier_for_step(40, schedule) == 1.0
    assert get_lr_multiplier_for_step(351, schedule) < 1.0


def test_deepseek_schedule_keeps_existing_warmup_and_uses_80_10_10_decay():
    schedule = build_lr_schedule_info(
        lr_schedule=LR_SCHEDULE_DEEPSEEK_80_10_10,
        num_iterations=1000,
        warmup_steps=40,
        warmdown_ratio=0.65,
        final_lr_frac=0.05,
    )
    assert schedule.warmup_iters == 40
    assert schedule.fixed_lr_start_step == 40
    assert schedule.fixed_lr_end_step == 800
    assert schedule.decay_step_1 == 800
    assert schedule.decay_step_2 == 900
    assert get_lr_multiplier_for_step(0, schedule) == 1 / 40
    assert get_lr_multiplier_for_step(40, schedule) == 1.0
    assert get_lr_multiplier_for_step(800, schedule) == 1.0
    assert get_lr_multiplier_for_step(801, schedule) == DEEPSEEK_STAGE2_LR_FRAC
    assert get_lr_multiplier_for_step(901, schedule) == DEEPSEEK_STAGE3_LR_FRAC


def test_muon_momentum_uses_default_warmdown_endpoint():
    schedule = build_lr_schedule_info(
        lr_schedule=LR_SCHEDULE_LINEAR_WARMDOWN,
        num_iterations=2000,
        warmup_steps=40,
        warmdown_ratio=0.65,
        final_lr_frac=0.05,
    )
    warmdown_start = schedule.num_iterations - schedule.warmdown_iters
    assert get_muon_momentum_for_step(warmdown_start, schedule) == 0.97
    assert get_muon_momentum_for_step(schedule.num_iterations, schedule) == 0.90


def test_muon_momentum_supports_custom_warmdown_endpoint():
    schedule = build_lr_schedule_info(
        lr_schedule=LR_SCHEDULE_LINEAR_WARMDOWN,
        num_iterations=2000,
        warmup_steps=40,
        warmdown_ratio=0.65,
        final_lr_frac=0.05,
    )
    assert get_muon_momentum_for_step(schedule.num_iterations, schedule, muon_final_momentum=0.97) == 0.97
