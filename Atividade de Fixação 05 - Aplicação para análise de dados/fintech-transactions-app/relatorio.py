from pathlib import Path

import matplotlib.pyplot as plt

from pipeline import executar_pipeline


BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

resultado = executar_pipeline(BASE / "transacoes.csv", BASE / "cotacoes.csv")

resultado.transacoes.to_csv(OUT / "transacoes_tratadas.csv", index=False, encoding="utf-8-sig")
resultado.transacoes_filtradas.to_csv(OUT / "transacoes_filtradas.csv", index=False, encoding="utf-8-sig")
resultado.anomalias.to_csv(OUT / "anomalias.csv", index=False, encoding="utf-8-sig")
resultado.consolidado.to_csv(OUT / "transacoes_consolidadas_cotacao.csv", index=False, encoding="utf-8-sig")
resultado.pivot_risco.to_csv(OUT / "pivot_risco.csv", encoding="utf-8-sig")

serie = resultado.serie_diaria
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(serie["data"], serie["valor_total"], label="Valor total diário")
ax.plot(serie["data"], serie["media_movel_7d"], label="Média móvel (7 dias)")
ax.set_ylim(bottom=0)
ax.set_title("Valor total de transações por dia")
ax.set_xlabel("Data")
ax.set_ylabel("Valor (R$)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "transacoes_diarias.png", dpi=160)
plt.close(fig)

print("Pipeline executado com sucesso.")
print(f"Transações após limpeza: {len(resultado.transacoes)}")
print(f"Valores imputados: {resultado.valores_imputados}")
print(f"Duplicadas removidas: {resultado.duplicadas_removidas}")
print(f"Filtro setembro/SP-RJ/>5000: {len(resultado.transacoes_filtradas)}")
print(f"Anomalias Z > 2.5: {len(resultado.anomalias)}")
print(f"Arquivos exportados para: {OUT}")
