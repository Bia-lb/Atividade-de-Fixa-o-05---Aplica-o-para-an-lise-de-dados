from pathlib import Path

import pandas as pd

from pipeline import (
    adicionar_zscore_por_estado,
    criar_pivot_risco,
    executar_pipeline,
    filtro_setembro_sp_rj_acima_5000,
)


BASE = Path(__file__).resolve().parents[1]


def test_pipeline_remove_duplicadas_e_imputa_nan():
    r = executar_pipeline(BASE / "transacoes.csv", BASE / "cotacoes.csv")
    assert r.duplicadas_removidas == 15
    assert r.transacoes["valor"].isna().sum() == 0
    assert (r.transacoes["plataforma"] == "Mobile").all()
    assert str(r.transacoes["data_transacao"].dtype).endswith("America/Sao_Paulo]")


def test_filtro_bitwise_resultado_valido():
    r = executar_pipeline(BASE / "transacoes.csv", BASE / "cotacoes.csv")
    f = filtro_setembro_sp_rj_acima_5000(r.transacoes)
    assert (f["mes"] == 9).all()
    assert f["estado_cliente"].isin(["SP", "RJ"]).all()
    assert (f["valor"] > 5000).all()


def test_pivot_tem_totais():
    r = executar_pipeline(BASE / "transacoes.csv", BASE / "cotacoes.csv")
    p = criar_pivot_risco(r.transacoes)
    assert "Total" in p.index
    assert "Total" in p.columns


def test_zscore_e_vetorizado_e_finito():
    r = executar_pipeline(BASE / "transacoes.csv", BASE / "cotacoes.csv")
    z = adicionar_zscore_por_estado(r.transacoes)
    assert "z_score" in z.columns
    assert z["z_score"].notna().all()


def test_cotacoes_consolidadas_sem_nan():
    r = executar_pipeline(BASE / "transacoes.csv", BASE / "cotacoes.csv")
    assert r.consolidado["cotacao_usd"].notna().all()
    assert r.consolidado["valor_usd"].notna().all()
