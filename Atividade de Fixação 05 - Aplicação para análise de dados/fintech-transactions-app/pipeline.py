from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RISCO_DICT = {
    "C100": "Baixo",
    "C101": "Alto",
    "C102": "Médio",
    "C103": "Alto",
    "C104": "Baixo",
}


@dataclass
class PipelineResult:
    transacoes: pd.DataFrame
    transacoes_filtradas: pd.DataFrame
    pivot_risco: pd.DataFrame
    anomalias: pd.DataFrame
    serie_diaria: pd.DataFrame
    consolidado: pd.DataFrame
    cotacoes: pd.DataFrame
    duplicadas_removidas: int
    valores_imputados: int


def carregar_transacoes(caminho: str | Path) -> pd.DataFrame:
    """Lê o CSV legado garantindo que latin1 não corrompa caracteres."""
    return pd.read_csv(caminho, encoding="latin1")


def carregar_cotacoes(caminho: str | Path) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8")
    df["data"] = pd.to_datetime(df["data"])

    # As cotações ausentes do arquivo sintético são preenchidas por interpolação
    # temporal para permitir a consolidação diária sem buracos.
    df = df.sort_values("data").set_index("data")
    df["cotacao_usd"] = df["cotacao_usd"].interpolate(method="time").ffill().bfill()
    return df.reset_index()


def preparar_transacoes(
    df: pd.DataFrame,
    risco_dict: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    risco_dict = risco_dict or RISCO_DICT
    df = df.copy()

    # 1) Imputação vetorizada pela mediana de cada estado.
    valores_ausentes = int(df["valor"].isna().sum())
    mediana_estado = df.groupby("estado_cliente")["valor"].transform("median")
    df["valor"] = df["valor"].fillna(mediana_estado)

    # 2) Coluna estática para rastreabilidade do pipeline.
    df["plataforma"] = "Mobile"

    # 3) Conversão temporal + timezone solicitado.
    df["data_transacao"] = pd.to_datetime(df["data_transacao"])
    if df["data_transacao"].dt.tz is None:
        df["data_transacao"] = df["data_transacao"].dt.tz_localize(
            "America/Sao_Paulo",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )

    # 4) Features temporais usando o acessor .dt.
    df["dia_semana"] = df["data_transacao"].dt.day_name(locale="pt_BR.utf8") \
        if _locale_disponivel() else df["data_transacao"].dt.day_name()
    df["mes"] = df["data_transacao"].dt.month

    # 5) Remoção de duplicadas, mantendo a primeira ocorrência.
    antes = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    duplicadas_removidas = antes - len(df)

    # 6) Categoria de risco obrigatoriamente via .map().
    df["nivel_risco"] = df["id_cliente"].map(risco_dict).fillna("Não classificado")

    return df, duplicadas_removidas, valores_ausentes


def _locale_disponivel() -> bool:
    """Evita quebrar a execução em sistemas sem locale pt_BR instalado."""
    try:
        pd.Series(pd.to_datetime(["2026-09-01"])).dt.day_name(locale="pt_BR.utf8")
        return True
    except (locale.Error, ValueError, TypeError, OSError, NameError):
        return False


# Import separado para manter a função acima portável.
import locale  # noqa: E402


def filtro_setembro_sp_rj_acima_5000(df: pd.DataFrame) -> pd.DataFrame:
    """Filtro exigido usando operadores bitwise (& e |), sem apply/loops."""
    mascara = (
        (df["mes"] == 9)
        & ((df["estado_cliente"] == "SP") | (df["estado_cliente"] == "RJ"))
        & (df["valor"] > 5000)
    )
    return df.loc[mascara].copy()


def criar_pivot_risco(df: pd.DataFrame) -> pd.DataFrame:
    return pd.pivot_table(
        df,
        index="mes",
        columns="nivel_risco",
        values="valor",
        aggfunc="sum",
        margins=True,
        margins_name="Total",
        fill_value=0,
    )


def adicionar_zscore_por_estado(df: pd.DataFrame) -> pd.DataFrame:
    """Z-Score por estado, vetorizado, sem laço for."""
    resultado = df.copy()
    grupo = resultado.groupby("estado_cliente")["valor"]
    media = grupo.transform("mean")
    desvio = grupo.transform("std")

    # Evita divisão por zero caso algum estado tenha variância nula.
    resultado["z_score"] = np.where(
        desvio.eq(0) | desvio.isna(),
        0.0,
        (resultado["valor"] - media) / desvio,
    )
    return resultado


def detectar_anomalias(df: pd.DataFrame, limiar: float = 2.5) -> pd.DataFrame:
    com_z = adicionar_zscore_por_estado(df)
    return com_z.loc[com_z["z_score"] > limiar].copy()


def calcular_serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    diario = (
        df.assign(data=df["data_transacao"].dt.tz_localize(None).dt.normalize())
        .groupby("data", as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "valor_total"})
        .sort_values("data")
    )
    diario["media_movel_7d"] = diario["valor_total"].rolling(7).mean()
    return diario


def consolidar_com_cotacoes(df_transacoes: pd.DataFrame, df_cotacoes: pd.DataFrame) -> pd.DataFrame:
    trans = df_transacoes.copy()
    trans["data"] = trans["data_transacao"].dt.tz_localize(None).dt.normalize()

    cot = df_cotacoes.copy()
    cot["data"] = pd.to_datetime(cot["data"]).dt.normalize()

    # merge_asof usa a cotação disponível mais recente para cada transação.
    # Isso também cobre transações posteriores à última data de cotação do arquivo.
    consolidado = pd.merge_asof(
        trans.sort_values("data"),
        cot.sort_values("data"),
        on="data",
        direction="backward",
    ).sort_index()
    consolidado["valor_usd"] = consolidado["valor"] / consolidado["cotacao_usd"]
    return consolidado


def executar_pipeline(
    transacoes_path: str | Path = "transacoes.csv",
    cotacoes_path: str | Path = "cotacoes.csv",
) -> PipelineResult:
    bruto = carregar_transacoes(transacoes_path)
    cotacoes = carregar_cotacoes(cotacoes_path)
    transacoes, duplicadas, imputados = preparar_transacoes(bruto)
    filtradas = filtro_setembro_sp_rj_acima_5000(transacoes)
    pivot = criar_pivot_risco(transacoes)
    anomalias = detectar_anomalias(transacoes)
    serie = calcular_serie_diaria(transacoes)
    consolidado = consolidar_com_cotacoes(transacoes, cotacoes)

    return PipelineResult(
        transacoes=transacoes,
        transacoes_filtradas=filtradas,
        pivot_risco=pivot,
        anomalias=anomalias,
        serie_diaria=serie,
        consolidado=consolidado,
        cotacoes=cotacoes,
        duplicadas_removidas=duplicadas,
        valores_imputados=imputados,
    )
