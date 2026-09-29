"""Reference functions resolve without torch; their math is checked only when torch is present."""

from __future__ import annotations

import pytest

from vka import reference


def test_resolve_finds_reference_functions() -> None:
    assert reference.resolve("vka.reference:gemm") is reference.gemm


@pytest.mark.parametrize("bad", ["vka.reference", ":gemm", "vka.reference:"])
def test_resolve_rejects_malformed_specs(bad: str) -> None:
    with pytest.raises(ValueError):
        reference.resolve(bad)


def test_references_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)

    a = torch.randn(3, 5, generator=gen)
    b = torch.randn(5, 2, generator=gen)
    torch.testing.assert_close(reference.gemm(a, b), a @ b)

    x = torch.randn(4, 7, generator=gen)
    y = reference.softmax(x)
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(4))

    gamma = torch.randn(7, generator=gen)
    beta = torch.randn(7, generator=gen)
    y = reference.layernorm(x, gamma, beta)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    expected = (x - mean) / torch.sqrt(var + reference.LAYERNORM_EPS) * gamma + beta
    torch.testing.assert_close(y, expected)
