"""Visual comum a todos os gráficos, pasta de saída e o cálculo do poliedro viável (2 variáveis)."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))          # permite "from simplex import Simplex" de dentro de exemplos/
IMAGENS = RAIZ / "imagens"
IMAGENS.mkdir(exist_ok=True)

AZUL, LARANJA, CINZA, VERDE, VERMELHO = "#2563eb", "#ea580c", "#9ca3af", "#16a34a", "#dc2626"
CORES_RESTRICOES = [LARANJA, VERDE, "#9333ea", "#0891b2", "#ca8a04"]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def salvar(fig, nome):
    """Salva a figura em imagens/<nome> e avisa no terminal."""
    caminho = IMAGENS / nome
    fig.savefig(caminho)
    print("salvo:", caminho.relative_to(RAIZ))


def expressao(coefs):
    """Escreve 3x₁ + 2x₂ de forma limpa: pula termos com 0 e não escreve o coeficiente 1."""
    subscritos = "₁₂₃₄₅₆₇₈₉"
    termos = []
    for j, a in enumerate(coefs):
        if abs(a) < 1e-12:
            continue
        coef = "" if abs(abs(a) - 1) < 1e-12 else f"{abs(a):g}"
        sinal = "-" if a < 0 else "+"
        termos.append((sinal, f"{coef}x{subscritos[j]}"))
    texto = ""
    for k, (sinal, termo) in enumerate(termos):
        texto += (("-" if sinal == "-" else "") if k == 0 else f" {sinal} ") + termo
    return texto or "0"


def vertices_poliedro(A, b):
    """
    Vértices da região {x : Ax <= b, x >= 0}.

    Cada vértice é o cruzamento de n restrições "ativas" (que valem com igualdade).
    Testamos todas as combinações, resolvemos o sistema e ficamos só com os pontos viáveis.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]
    A_tot = np.vstack([A, -np.eye(n)])         # x >= 0  vira  -x <= 0
    b_tot = np.concatenate([b, np.zeros(n)])

    vertices = []
    for linhas in combinations(range(len(b_tot)), n):
        M = A_tot[list(linhas)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue                           # retas/planos paralelos: não se cruzam num ponto
        p = np.linalg.solve(M, b_tot[list(linhas)])
        if np.all(A_tot @ p <= b_tot + 1e-9):
            vertices.append(p)
    return np.unique(np.round(vertices, 9), axis=0)


def ordenar_no_contorno(vertices):
    """Ordena vértices 2D em sentido anti-horário (para desenhar o polígono)."""
    centro = vertices.mean(axis=0)
    angulo = np.arctan2(vertices[:, 1] - centro[1], vertices[:, 0] - centro[0])
    return vertices[np.argsort(angulo)]


def desenhar_regiao(ax, A, b, limites=None):
    """Desenha a região viável e as retas das restrições num eixo 2D. Devolve os vértices."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    vertices = ordenar_no_contorno(vertices_poliedro(A, b))

    if limites is None:
        limites = (vertices[:, 0].max() * 1.35 + 0.5, vertices[:, 1].max() * 1.35 + 0.5)
    xmax, ymax = limites
    x = np.linspace(0, xmax, 400)

    ax.fill(vertices[:, 0], vertices[:, 1], color=AZUL, alpha=0.18, label="região viável")
    ax.plot(*np.vstack([vertices, vertices[0]]).T, color=AZUL, lw=1.5)
    ax.scatter(vertices[:, 0], vertices[:, 1], color=AZUL, s=25, zorder=3)

    for k, (a, bk) in enumerate(zip(A, b)):
        cor = CORES_RESTRICOES[k % len(CORES_RESTRICOES)]
        rotulo = f"{expressao(a)} ≤ {bk:g}"
        if abs(a[1]) > 1e-12:
            ax.plot(x, (bk - a[0] * x) / a[1], "--", color=cor, lw=1.3, label=rotulo)
        else:
            ax.axvline(bk / a[0], linestyle="--", color=cor, lw=1.3, label=rotulo)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_aspect("equal", adjustable="box")
    return vertices
