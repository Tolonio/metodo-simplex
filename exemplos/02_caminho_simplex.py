"""
02 · O caminho do Simplex: de vértice em vértice até o ótimo (GIF).

O que este exemplo mostra:
  O Simplex não procura dentro da região. Ele começa na origem (todas as folgas na base)
  e, a cada pivotação, anda por uma aresta até um vértice vizinho com f maior.
  Quando nenhum vizinho melhora f, para: aquele vértice é o ótimo.

Usa um problema com mais restrições para o caminho ficar mais longo:
    max  3x₁ + 5x₂
    s.a. x₁        <= 4
               2x₂ <= 12
         3x₁ + 2x₂ <= 18

Rode:  python exemplos/02_caminho_simplex.py   (gera o GIF em alguns segundos)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from estilo import IMAGENS, RAIZ, VERMELHO, desenhar_regiao
from simplex import Simplex

c = [3, 5]
A = [[1, 0],
     [0, 2],
     [3, 2]]
b = [4, 12, 18]

simplex = Simplex(c, A, b)
solucao, valor_otimo = simplex.solve()
caminho = np.array([p["vertice"] for p in simplex.historico])
valores = [p["z"] for p in simplex.historico]

print("Caminho percorrido pelo Simplex:")
for k, (v, z) in enumerate(zip(caminho, valores)):
    print(f"  iteração {k}: x = ({v[0]:.2f}, {v[1]:.2f})   f = {z:.2f}")

fig, ax = plt.subplots(figsize=(7, 6))
desenhar_regiao(ax, A, b, limites=(7, 9.5))
ax.legend(loc="upper right", fontsize=8.5)
x = np.linspace(0, 7, 200)

linha_caminho, = ax.plot([], [], color=VERMELHO, lw=2.5, zorder=4)
pontos = ax.scatter([], [], color=VERMELHO, s=90, zorder=5)
nivel, = ax.plot([], [], color="black", lw=1.2)
texto = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

QUADROS_POR_PASSO = 12   # quadros para "andar" uma aresta
PAUSA = 10               # quadros parado em cada vértice


def quadro(i):
    passo, resto = divmod(i, QUADROS_POR_PASSO + PAUSA)
    passo = min(passo, len(caminho) - 1)
    visitados = caminho[:passo + 1]

    # Interpola o deslocamento ao longo da aresta até o próximo vértice
    if passo < len(caminho) - 1 and resto < QUADROS_POR_PASSO:
        t = resto / QUADROS_POR_PASSO
        atual = (1 - t) * caminho[passo] + t * caminho[passo + 1]
        trilha = np.vstack([visitados, atual])
    else:
        atual = visitados[-1]
        trilha = visitados

    linha_caminho.set_data(trilha[:, 0], trilha[:, 1])
    pontos.set_offsets(visitados)
    f_atual = c[0] * atual[0] + c[1] * atual[1]
    nivel.set_data(x, (f_atual - c[0] * x) / c[1])
    fim = passo == len(caminho) - 1
    texto.set_text(f"iteração {passo}\nx = ({atual[0]:.2f}, {atual[1]:.2f})\n"
                   f"f = {f_atual:.2f}" + ("   ← ótimo" if fim else ""))
    return linha_caminho, pontos, nivel, texto


total = (len(caminho) - 1) * (QUADROS_POR_PASSO + PAUSA) + PAUSA * 3
ax.set_title("O Simplex anda pelas arestas, sempre aumentando f")
anim = FuncAnimation(fig, quadro, frames=total, interval=60, blit=False)

caminho_gif = IMAGENS / "02_caminho_simplex.gif"
anim.save(caminho_gif, writer=PillowWriter(fps=16))
print("salvo:", caminho_gif.relative_to(RAIZ))
plt.show()
