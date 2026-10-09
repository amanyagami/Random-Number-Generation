import wave

import numpy as np
import pytest

from trng import Generator, HealthTestError, HealthTests, InsufficientEntropyError, min_entropy_mcv
from trng.__main__ import main
from trng.ccml import CCML
from trng.entropy import bit_bias, serial_correlation
from trng.health import apt_cutoff, rct_cutoff
from trng.source import CallableSource, WavSource


def noise_source(seed: int = 0, scale: float = 40.0) -> CallableSource:
    rng = np.random.default_rng(seed)
    return CallableSource(lambda n: np.round(rng.normal(0, scale, n)).astype(np.int16))


def test_output_has_no_stuck_bits_and_high_min_entropy() -> None:
    data = Generator(noise_source()).read(32 * 400)
    mean_bit, stuck = bit_bias(data)
    assert stuck == 0
    assert abs(mean_bit - 0.5) < 0.005
    h = min_entropy_mcv(np.frombuffer(data, dtype=np.uint8))
    assert h > 7.0  # MCV carries a finite-sample penalty at 12.8 kB; the original script scored 3.2
    assert abs(serial_correlation(np.frombuffer(data, dtype=np.uint8))) < 0.02


def test_blocks_are_unique() -> None:
    g = Generator(noise_source(1))
    blocks = {g.block() for _ in range(200)}
    assert len(blocks) == 200


def test_same_input_same_output() -> None:
    assert Generator(noise_source(2)).read(64) == Generator(noise_source(2)).read(64)


def test_constant_source_trips_health_test() -> None:
    src = CallableSource(lambda n: np.zeros(n, dtype=np.int16))
    with pytest.raises(HealthTestError):
        Generator(src).block()


def test_stuck_bit_source_is_rejected() -> None:
    rng = np.random.default_rng(3)
    # Low bits are 0 90% of the time: ~0.15 bits/sample of min-entropy.
    src = CallableSource(lambda n: (rng.integers(0, 4, n) * (rng.random(n) < 0.1)).astype(np.int16))
    with pytest.raises((HealthTestError, InsufficientEntropyError)):
        Generator(src).block()


def test_too_little_noise_is_refused_not_stretched() -> None:
    with pytest.raises(InsufficientEntropyError):
        Generator(noise_source(4), block_samples=64).block()


def test_strongly_dependent_source_is_rejected() -> None:
    rng = np.random.default_rng(6)

    def sticky(n: int) -> np.ndarray:
        out = rng.integers(0, 4, n)
        stay = rng.random(n) < 0.9
        for i in range(1, n):
            if stay[i]:
                out[i] = out[i - 1]
        return out.astype(np.int16)

    with pytest.raises((HealthTestError, InsufficientEntropyError)):
        Generator(CallableSource(sticky)).block()


def test_cutoffs_are_sane() -> None:
    assert rct_cutoff(2.0) == 21
    assert rct_cutoff(1.0) == 41
    assert 128 < apt_cutoff(2.0) < 512
    assert apt_cutoff(1.0) > apt_cutoff(2.0) > apt_cutoff(4.0)


def test_health_tests_pass_uniform_noise() -> None:
    HealthTests().check(np.random.default_rng(0).integers(0, 4, 100_000))


def test_ccml_stays_in_unit_interval_and_never_collapses() -> None:
    c = CCML()
    rng = np.random.default_rng(0)
    for _ in range(20_000):
        c.inject(rng.integers(0, 256, c.size))
        c.step()
        assert np.all((c.x > 0) & (c.x < 1))
    assert c.x.std() > 0.01


def test_ccml_is_sensitive_to_initial_conditions() -> None:
    a, b = CCML(), CCML()
    b.x[0] += 1e-12
    for _ in range(80):
        a.step()
        b.step()
    assert np.abs(a.x - b.x).max() > 0.05


def test_wav_roundtrip_and_cli(tmp_path, capsys) -> None:
    path = tmp_path / "noise.wav"
    rng = np.random.default_rng(5)
    pcm = np.round(rng.normal(0, 40, 4096 * 8)).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(pcm.tobytes())
    assert len(WavSource(str(path)).read(100)) == 100
    assert main(["--source", str(path), "--count", "5", "--hex"]) == 0
    lines = capsys.readouterr().out.split()
    assert len(lines) == 5 and all(int(v, 16) < 2**32 for v in lines)


def test_wav_exhaustion_is_an_error(tmp_path) -> None:
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(np.zeros(10, dtype="<i2").tobytes())
    with pytest.raises(EOFError):
        WavSource(str(path)).read(11)


def test_markov_estimator_flags_predictable_sequences_but_not_noise() -> None:
    from trng.entropy import min_entropy_markov

    rng = np.random.default_rng(0)
    assert min_entropy_markov(rng.integers(0, 4, 50_000), k=4) > 1.9
    cycle = np.tile(np.arange(4), 5000)
    assert min_entropy_markov(cycle, k=4) < 0.2


def test_plan_scales_with_requested_count() -> None:
    g = Generator(noise_source(7))
    p1 = g.plan(1, 32)
    assert (p1.blocks, p1.samples) == (1, 4096)
    assert p1.seconds == pytest.approx(4096 / 44100)
    # 100000 x 32-bit numbers = 400000 bytes = 12500 blocks of 32 bytes
    p = g.plan(100_000, 32)
    assert p.blocks == 12_500
    assert p.seconds == pytest.approx(12_500 * 4096 / 44100)
    for count in (1, 10, 100, 1000, 10000, 100000):
        assert g.plan(count).blocks >= 1
    with pytest.raises(ValueError):
        g.plan(0)


def test_integers_are_in_range_and_unbiased() -> None:
    g = Generator(noise_source(8))
    rolls = g.integers(6000, 1, 6)
    counts = np.bincount(rolls, minlength=7)[1:]
    assert set(rolls) <= {1, 2, 3, 4, 5, 6}
    assert counts.min() > 850 and counts.max() < 1150  # ~1000 each, >6 sigma margin
    assert g.integers(3, 5, 5) == [5, 5, 5]
    with pytest.raises(ValueError):
        g.integers(1, 3, 2)


def test_read_keeps_leftover_bytes_and_actual_use_matches_plan() -> None:
    g = Generator(noise_source(9))
    plan = g.plan(10, 32)  # 40 bytes -> 2 blocks of 32 bytes
    done: list[int] = []
    g.on_block = done.append
    assert len(g.integers(10, 0, 2**32 - 1)) == 10
    assert len(done) == plan.blocks == 2
    g.read(4)  # 64 - 40 - 4 = 20 bytes are left in the pool
    assert len(done) == 2
    assert g.plan(5, 32).blocks == 0  # 20 bytes already recorded cover 5 numbers
    assert g.plan(6, 32).blocks == 1


def test_cli_dry_run_reports_time(capsys) -> None:
    assert main(["--count", "100000", "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "100000 number(s)" in err and "19 min" in err


def _lcg_source() -> CallableSource:
    """Deterministic noise independent of NumPy's random-stream versions."""
    state = [12345]

    def fn(n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int16)
        x = state[0]
        for i in range(n):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            out[i] = (x >> 8) % 201 - 100
        state[0] = x
        return out

    return CallableSource(fn)


def test_golden_digest_pins_the_pipeline() -> None:
    """Any change to CCML, mixing or conditioning changes this digest on purpose."""
    import hashlib

    g = Generator(_lcg_source())
    h = hashlib.sha256()
    for _ in range(5):
        h.update(g.block())
    assert h.hexdigest() == "5f1ecc0e39d0376bbcbe6ac793eea2cf5ea586ed1ee24953503c6fce1f664670"
