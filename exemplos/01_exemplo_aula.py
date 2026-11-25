"""
01 · O exemplo da aula: resolver e plotar a solução e o poliedro.

    max  f(x) = 6x₁ + 4x₂
    s.a. 2x₁ + 4x₂ <= 12
         4x₁ + 2x₂ <= 16
         x₁, x₂ >= 0

O que este exemplo mostra:
  A região viável é um polígono (poliedro em 2D). A reta da função objetivo
  "empurrada" o máximo possível para fora encosta no polígono num vértice:
  é ali que está o ótimo, e é exatamente o ponto que o Simplex encontra.

Rode:  python exemplos/01_exemplo_aula.py
"""
import numpy as np
import matplotlib.pyplot as plt

from estilo import CINZA, VERMELHO, desenhar_regiao, salvar
from simplex import Simplex

c = [6, 4]
A = [[2, 4],
     [4, 2]]
b = [12, 16]

simplex = Simplex(c, A, b)
solucao, valor_otimo = simplex.solve()
print(f"Solução ótima: x₁ = {solucao[0]:.4f}, x₂ = {solucao[1]:.4f}")
print(f"Valor ótimo:   f(x) = {valor_otimo:.4f}")

fig, ax = plt.subplots(figsize=(7.5, 6))
vertices = desenhar_regiao(ax, A, b, limites=(6.5, 5))

# Curvas de nível de f: retas 6x₁ + 4x₂ = k. Todas paralelas; a do ótimo só toca o polígono.
x = np.linspace(0, 6.5, 200)
for k in [6, 12, 18]:
    ax.plot(x, (k - c[0] * x) / c[1], color=CINZA, lw=0.8, alpha=0.7)
    ax.annotate(f"f = {k}", (k / c[0] / 2, k / c[1] / 2), color=CINZA, fontsize=8)
ax.plot(x, (valor_otimo - c[0] * x) / c[1], color="black", lw=1.6,
        label=f"f = {valor_otimo:.2f} (nível ótimo)")

# Valor de f em cada vértice: o Simplex só precisa olhar para eles
for v in vertices:
    ax.annotate(f"f = {c[0] * v[0] + c[1] * v[1] + 0.0:.2f}", v, textcoords="offset points",
                xytext=(6, 6), fontsize=8)

ax.scatter(*solucao, color=VERMELHO, s=140, zorder=5,
           label=f"ótimo ({solucao[0]:.2f}, {solucao[1]:.2f})")
ax.set_title("max 6x₁ + 4x₂  —  região viável e solução ótima")
ax.legend(loc="upper right", fontsize=8.5)

salvar(fig, "01_exemplo_aula.png")
plt.show()
