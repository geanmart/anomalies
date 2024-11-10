import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Simulação de dados de exemplo
metrics_historic_detector = {
    'α=1.0': [1.0, 0.5, 0.6666666666666666, 0.5, 0.4, 0.3333333333333333, 0.2857142857142857],
    'α=1.2': [0.0, 1.0, 2.0, 3.0, 2.5, 1.8, 1.2],
    # Adicione outros valores conforme necessário
}

X_train = range(7)  # Exemplo de tamanho de treino
idx_drift = [3]  # Índices onde ocorre drift (exemplo)

# Streamlit: Criação do gráfico
st.set_option('deprecation.showPyplotGlobalUse', False)

# Configurações de gráfico
plt.rcParams.update({"font.size": 20})

alpha = 0.6
linewidth = 1.0

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(12, 8),
    sharex=True,
    dpi=300,
    gridspec_kw={"height_ratios": [3, 1]},
)

# Loop para plotar as métricas
for (metric_name, metric_values), linecolor in zip(
    metrics_historic_detector.items(),
    ["#1f77b4", "#ff7f0e", "#2ca02c"],
):
    ax[0].plot(
        metric_values,
        color=linecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=metric_name,
    )

# Marcação de drift e treinamento
drift_color = "red"
drift_linestyle = "--"
warmup_color = "grey"

for idx in range(0, len(X_train)):
    ax[1].axvline(x=idx, color=warmup_color, linewidth=linewidth)

for idx in idx_drift:
    ax[1].axvline(x=idx, color=drift_color, linestyle=drift_linestyle, linewidth=1.5)

# Labels e legendas
ax[0].set_ylabel("Prequential Error")
ax[0].legend(
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.175),
    fancybox=True,
)
ax[1].set_yticks([])
ax[1].set_ylabel("Monitoring")

# Criando patches para a legenda
drift_path = mpatches.Patch(
    color=drift_color, label="Drift detected", linestyle=drift_linestyle
)
warmup_path = mpatches.Patch(color=warmup_color, label="Training phase")

ax[1].legend(
    handles=[warmup_path, drift_path],
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.2),
    fancybox=True,
)

# Ajustando layout
fig.tight_layout()

# Exibindo o gráfico em Streamlit
st.pyplot(fig)
