"""
04 · Três variáveis: o poliedro em 3D.

O que este exemplo mostra:
  O código do Simplex não muda nada com mais variáveis. Em 3D a região viável
  é um poliedro de verdade (com faces), e o caminho continua pulando de vértice em vértice.

Exemplo clássico de livro-texto (Chvátal, "Linear Programming", cap. 2):
    max  5x₁ + 4x₂ + 3x₃
    s.a. 2x₁ + 3x₂ +  x₃ <= 5
         4x₁ +  x₂ + 2x₃ <= 11
         3x₁ + 4x₂ + 2x₃ <= 8
    Resposta conhecida: x = (2, 0, 1), f = 13.

Rode:  python exemplos/04_tres_variaveis.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from estilo import AZUL, VERMELHO, salvar, vertices_poliedro
from simplex import Simplex

c = [5, 4, 3]
A = [[2, 3, 1],
     [4, 1, 2],
     [3, 4, 2]]
b = [5, 11, 8]

simplex = Simplex(c, A, b)
solucao, valor_otimo = simplex.solve()
print(f"Solução ótima: x = {np.round(solucao, 4)}   f = {valor_otimo:.4f}")
assert np.allclose(solucao, [2, 0, 1]) and np.isclose(valor_otimo, 13)
print("Confere com a resposta do livro: x = (2, 0, 1), f = 13.")

# ---- Faces do poliedro ----------------------------------------------------
# Uma face = os vértices que ficam em cima de uma mesma restrição (ativa com igualdade).
A_tot = np.vstack([np.array(A, float), -np.eye(3)])
b_tot = np.concatenate([np.array(b, float), np.zeros(3)])
V = vertices_poliedro(A, b)

faces = []
for a, bk in zip(A_tot, b_tot):
    na_face = V[np.isclose(V @ a, bk, atol=1e-9)]
    if len(na_face) < 3:
        continue
    # ordena os pontos da face em volta do centro, num plano 2D da própria face
    centro = na_face.mean(axis=0)
    normal = a / np.linalg.norm(a)
    u = np.cross(normal, [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    ang = np.arctan2((na_face - centro) @ w, (na_face - centro) @ u)
    faces.append(na_face[np.argsort(ang)])

caminho = np.array([p["vertice"] for p in simplex.historico])

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection="3d")
ax.add_collection3d(Poly3DCollection(faces, facecolor=AZUL, alpha=0.15, edgecolor=AZUL, lw=1))
ax.scatter(*V.T, color=AZUL, s=20)
ax.plot(*caminho.T, color=VERMELHO, lw=2.5, marker="o", label="caminho do Simplex")
ax.scatter(*solucao, color=VERMELHO, s=150, label=f"ótimo (2, 0, 1), f = {valor_otimo:.0f}")
for k, v in enumerate(caminho):
    ax.text(*v, f"  {k}", color=VERMELHO, fontsize=10)

limite = V.max(axis=0) * 1.15
ax.set_xlim(0, limite[0])
ax.set_ylim(0, limite[1])
ax.set_zlim(0, limite[2])
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("x₃")
ax.view_init(elev=22, azim=-60)
ax.set_title("max 5x₁ + 4x₂ + 3x₃ — poliedro viável e caminho até o ótimo")
ax.legend(loc="upper left")
salvar(fig, "04_tres_variaveis.png")
plt.show()
