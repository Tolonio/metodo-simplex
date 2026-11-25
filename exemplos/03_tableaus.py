"""
03 · Os tableaus, iteração por iteração.

O que este exemplo mostra:
  Cada pivotação, no papel: qual coluna entra (mais negativo na linha z),
  qual linha sai (menor razão b/a), e como o tableau fica depois.
  Imprime no terminal e salva uma imagem com todos os tableaus.

Rode:  python exemplos/03_tableaus.py
"""
import matplotlib.pyplot as plt

from estilo import AZUL, LARANJA, VERMELHO, salvar
from simplex import Simplex

c = [6, 4]
A = [[2, 4],
     [4, 2]]
b = [12, 16]

simplex = Simplex(c, A, b)
solucao, valor_otimo = simplex.solve()
historico = simplex.historico

# ---- Terminal -------------------------------------------------------------
for k, passo in enumerate(historico):
    print(f"\n=== Iteração {k} ===")
    simplex.imprimir_tableau(passo["tableau"], passo["base"], passo["pivo"])
    if passo["pivo"]:
        lin, col = passo["pivo"]
        print(f"→ entra {simplex.nomes[col]} (mais negativo na linha z), "
              f"sai {simplex.nomes[passo['base'][lin]]} (menor razão b/a)")
    else:
        print("→ nenhum negativo na linha z: ótimo!")

# ---- Imagem ---------------------------------------------------------------
cabecalho = ["base"] + simplex.nomes + ["b"]
fig, eixos = plt.subplots(len(historico), 1, figsize=(7.5, 1.9 * len(historico)))

for k, (ax, passo) in enumerate(zip(eixos, historico)):
    ax.axis("off")
    T = passo["tableau"]
    linhas = []
    for i in range(T.shape[0]):
        rotulo = simplex.nomes[passo["base"][i]] if i < T.shape[0] - 1 else "z"
        linhas.append([rotulo] + [f"{v:.3g}" for v in T[i]])

    tabela = ax.table(cellText=linhas, colLabels=cabecalho, loc="center", cellLoc="center")
    tabela.scale(1, 1.35)
    for (i, j), celula in tabela.get_celld().items():
        celula.set_edgecolor("#d1d5db")
        if i == 0:
            celula.set_facecolor("#eff6ff")
            celula.set_text_props(weight="bold")
        if i == len(linhas):                   # linha z
            celula.set_facecolor("#f9fafb")

    if passo["pivo"]:
        lin, col = passo["pivo"]
        for i in range(1, len(linhas) + 1):
            tabela[i, col + 1].set_facecolor("#ffedd5")        # coluna que entra
        for j in range(len(cabecalho)):
            tabela[lin + 1, j].set_facecolor("#ffedd5")       # linha que sai
        tabela[lin + 1, col + 1].set_facecolor(LARANJA)
        tabela[lin + 1, col + 1].set_text_props(weight="bold", color="white")
        titulo = (f"Iteração {k}: entra {simplex.nomes[col]}, "
                  f"sai {simplex.nomes[passo['base'][lin]]}   (f = {passo['z']:.2f})")
        cor = AZUL
    else:
        titulo = f"Iteração {k}: ótimo!  x = ({solucao[0]:.2f}, {solucao[1]:.2f}),  f = {valor_otimo:.2f}"
        cor = VERMELHO
    ax.set_title(titulo, color=cor, fontsize=10.5, loc="left")

fig.tight_layout()
salvar(fig, "03_tableaus.png")
plt.show()
