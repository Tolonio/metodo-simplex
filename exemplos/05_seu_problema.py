"""
05 · Sua vez: troque c, A e b pelo seu problema.

Formato aceito:
    max  c^T x
    s.a. A x <= b      (com b >= 0)
         x >= 0

Dicas:
  - Para MINIMIZAR g(x), maximize -g(x) (troque o sinal de c) e troque o sinal do resultado.
  - Restrição com >= : multiplique por -1 para virar <= (mas aí b fica negativo e este
    Simplex simples não aceita; precisaria do método das duas fases ou do Big-M).
  - Com 2 variáveis o gráfico é desenhado automaticamente.

Rode:  python exemplos/05_seu_problema.py
"""
import numpy as np
import matplotlib.pyplot as plt

from estilo import VERMELHO, desenhar_regiao, salvar
from simplex import Simplex

# ----- Edite aqui ----------------------------------------------------------
# Exemplo: fábrica de mesas (x₁) e cadeiras (x₂)
#   lucro: 70 por mesa, 50 por cadeira
#   carpintaria: 4 h por mesa, 3 h por cadeira, 240 h disponíveis
#   pintura:     2 h por mesa, 1 h por cadeira, 100 h disponíveis
c = [70, 50]
A = [[4, 3],
     [2, 1]]
b = [240, 100]
# ---------------------------------------------------------------------------

simplex = Simplex(c, A, b)
solucao, valor_otimo = simplex.solve()

print("Iterações:")
for k, passo in enumerate(simplex.historico):
    print(f"  {k}: x = {np.round(passo['vertice'], 3)}   f = {passo['z']:.3f}")
print(f"\nSolução ótima: x = {np.round(solucao, 4)}")
print(f"Valor ótimo:   f = {valor_otimo:.4f}")

if len(c) == 2:
    fig, ax = plt.subplots(figsize=(7, 6))
    desenhar_regiao(ax, A, b)
    x = np.linspace(0, ax.get_xlim()[1], 200)
    if c[1] != 0:
        ax.plot(x, (valor_otimo - c[0] * x) / c[1], color="black", lw=1.4,
                label=f"f = {valor_otimo:.2f}")
    ax.scatter(*solucao, color=VERMELHO, s=140, zorder=5,
               label=f"ótimo ({solucao[0]:.2f}, {solucao[1]:.2f})")
    ax.set_title("Seu problema")
    ax.legend(fontsize=8.5)
    salvar(fig, "05_seu_problema.png")
    plt.show()
