from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pipeline import RISCO_DICT, executar_pipeline


BASE_DIR = Path(__file__).resolve().parent
TRANSACOES = BASE_DIR / "transacoes.csv"
COTACOES = BASE_DIR / "cotacoes.csv"

st.set_page_config(page_title="Fintech | Monitor de Transações", page_icon="📊", layout="wide")

st.title("📊 Monitor de Transações Financeiras")
st.caption("Pipeline de limpeza, consolidação, análise de risco e detecção estatística de anomalias.")

try:
    resultado = executar_pipeline(TRANSACOES, COTACOES)
except FileNotFoundError:
    st.error("Arquivos CSV não encontrados. Execute primeiro: `python criar_dados.py`.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transações válidas", f"{len(resultado.transacoes):,}".replace(",", "."))
col2.metric("NaNs imputados", resultado.valores_imputados)
col3.metric("Duplicadas removidas", resultado.duplicadas_removidas)
col4.metric("Anomalias Z > 2,5", len(resultado.anomalias))

st.divider()

aba1, aba2, aba3, aba4 = st.tabs([
    "📈 Visão temporal",
    "🚨 Anomalias",
    "🎯 Filtro solicitado",
    "🧩 Risco e consolidação",
])

with aba1:
    st.subheader("Valor diário e média móvel de 7 dias")
    serie = resultado.serie_diaria

    # API orientada a objetos do Matplotlib, conforme exigência.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(serie["data"], serie["valor_total"], label="Valor total diário")
    ax.plot(serie["data"], serie["media_movel_7d"], label="Média móvel (7 dias)")
    ax.set_ylim(bottom=0)
    ax.set_title("Volume financeiro diário e tendência suavizada")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (R$)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

    st.dataframe(serie, use_container_width=True, hide_index=True)

with aba2:
    st.subheader("Potenciais anomalias/fraudes por Z-Score")
    st.write("Critério: transações cujo Z-Score do valor dentro do próprio estado é maior que **2,5**.")

    if resultado.anomalias.empty:
        st.info("Nenhuma transação ultrapassou o limiar Z > 2,5 neste conjunto.")
    else:
        st.dataframe(
            resultado.anomalias.sort_values("z_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

with aba3:
    st.subheader("Setembro • SP ou RJ • valor > R$ 5.000")
    st.caption("Filtro implementado de forma vetorizada usando obrigatoriamente `&` e `|`.")

    filtradas = resultado.transacoes_filtradas
    st.metric("Registros encontrados", len(filtradas))
    st.dataframe(filtradas, use_container_width=True, hide_index=True)

with aba4:
    st.subheader("Tabela dinâmica por mês e nível de risco")
    st.caption(f"Mapa de risco utilizado: {RISCO_DICT}")
    st.dataframe(resultado.pivot_risco, use_container_width=True)

    st.subheader("Consolidação com cotações")
    st.write(
        "As transações são associadas à cotação USD do mesmo dia. As duas cotações ausentes "
        "do arquivo sintético são interpoladas temporalmente para evitar lacunas na consolidação."
    )
    colunas = [
        "id_cliente",
        "data_transacao",
        "estado_cliente",
        "valor",
        "cotacao_usd",
        "valor_usd",
        "nivel_risco",
    ]
    st.dataframe(resultado.consolidado[colunas], use_container_width=True, hide_index=True)

st.divider()
st.caption("Projeto acadêmico • Pandas + NumPy + Matplotlib + Streamlit")
