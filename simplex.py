"""
Método Simplex (forma tableau) para problemas de programação linear:

    max  c^T x
    s.a. A x <= b
         x  >= 0        (com b >= 0, então a origem é viável)

Rode este arquivo direto para resolver o exemplo da aula e conferir o resultado:
    python simplex.py
"""
import numpy as np


class Simplex:
    def __init__(self, c, A, b):
        """
        Inicializa o Simplex.
        :c: Coeficientes da função objetivo.
        :A: Matriz de restrições.
        :b: Lado direito das restrições.
        Input Simplex padrão
        max c^T * x
        s.a. Ax <= b
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = self.A.shape          # m restrições, n variáveis de decisão

        if self.c.shape != (self.n,) or self.b.shape != (self.m,):
            raise ValueError("Dimensões de c, A e b não batem.")
        if np.any(self.b < 0):
            raise ValueError("Este Simplex exige b >= 0 (a origem precisa ser viável).")

        # Nomes das colunas do tableau: x1..xn (decisão) e s1..sm (folgas)
        self.nomes = [f"x{j + 1}" for j in range(self.n)] + [f"s{i + 1}" for i in range(self.m)]

        # No começo, as variáveis básicas são as folgas (colunas n, n+1, ..., n+m-1)
        self.base = list(range(self.n, self.n + self.m))
        self.tableau = self._inicial_tableau()

        # Histórico: um registro por iteração (tableau, base e pivô escolhido)
        self.historico = []

    def _inicial_tableau(self):
        """
        Cria o tableau inicial para o método Simplex.

        Formato (m+1) x (n+m+1):
            [  A | I | b ]   <- uma linha por restrição
            [ -c | 0 | 0 ]   <- linha da função objetivo (linha z)
        """
        tableau = np.zeros((self.m + 1, self.n + self.m + 1))
        tableau[:self.m, :self.n] = self.A                        # coeficientes das restrições
        tableau[:self.m, self.n:self.n + self.m] = np.eye(self.m)  # variáveis de folga
        tableau[:self.m, -1] = self.b                             # lado direito
        tableau[-1, :self.n] = -self.c                            # linha z:  z - c^T x = 0
        return tableau

    def _registrar(self, pivo):
        self.historico.append({
            "tableau": self.tableau.copy(),
            "base": list(self.base),
            "pivo": pivo,                      # (linha, coluna) ou None na última iteração
            "vertice": self.vertice_atual(),
            "z": self.tableau[-1, -1],
        })

    def _pivotacao(self):
        """
        Encontra o pivô e realiza a operação de pivoteamento.
        """
        T = self.tableau
        eps = 1e-12
        while True:
            # Encontra a coluna pivô (entra na base):
            # o coeficiente mais negativo da linha z é a variável que mais aumenta z
            col = int(np.argmin(T[-1, :-1]))
            if T[-1, col] >= -eps:
                self._registrar(pivo=None)
                break                          # nenhum coeficiente negativo: chegamos ao ótimo

            # Encontra a linha pivô (sai da base): teste da razão mínima b_i / a_i,col
            # (só vale para a_i,col > 0; é quem "trava" primeiro quando a variável cresce)
            coluna = T[:-1, col]
            razoes = np.full(self.m, np.inf)
            positivos = coluna > eps
            razoes[positivos] = T[:-1, -1][positivos] / coluna[positivos]
            lin = int(np.argmin(razoes))
            if np.isinf(razoes[lin]):
                raise ValueError("Problema ilimitado: a função objetivo cresce sem limite.")

            self._registrar(pivo=(lin, col))

            # Define pivô igual a 1
            T[lin, :] = T[lin, :] / T[lin, col]

            # Atualiza as outras linhas (zera o resto da coluna pivô)
            for i in range(self.m + 1):
                if i != lin:
                    T[i, :] -= T[i, col] * T[lin, :]

            self.base[lin] = col

    def vertice_atual(self):
        """Valores das variáveis de decisão na solução básica atual."""
        x = np.zeros(self.n + self.m)
        for i, var in enumerate(self.base):
            x[var] = self.tableau[i, -1]
        return x[:self.n]

    def solve(self):
        """
        Resolve o problema de programação linear usando o método Simplex.
        """
        self._pivotacao()
        solucao = self.vertice_atual()
        valor_otimo = self.tableau[-1, -1]
        return solucao, valor_otimo

    def imprimir_tableau(self, tableau=None, base=None, pivo=None):
        """Mostra um tableau em formato de tabela no terminal, marcando o pivô com [ ]."""
        T = self.tableau if tableau is None else tableau
        base = self.base if base is None else base
        cab = ["base"] + self.nomes + ["b"]
        print("  ".join(f"{h:>8}" for h in cab))
        for i in range(self.m + 1):
            rotulo = self.nomes[base[i]] if i < self.m else "z"
            celulas = []
            for j in range(T.shape[1]):
                v = f"{T[i, j]:.3f}"
                celulas.append(f"[{v}]" if pivo == (i, j) else v)
            print("  ".join(f"{s:>8}" for s in [rotulo] + celulas))


if __name__ == "__main__":
    # Exemplo
    # Função objetivo: maximizar f(x) = 6x_1 + 4x_2
    # Restrições:
    # 2x_1 + 4x_2 <= 12
    # 4x_1 + 2x_2 <= 16
    c = [6, 4]
    A = [[2, 4],
         [4, 2]]
    b = [12, 16]

    simplex = Simplex(c, A, b)
    solucao, valor_otimo = simplex.solve()

    for k, passo in enumerate(simplex.historico):
        print(f"\n--- Iteração {k}  (vértice {np.round(passo['vertice'], 3)}, z = {passo['z']:.3f})")
        simplex.imprimir_tableau(passo["tableau"], passo["base"], passo["pivo"])

    print(f"\nSolução ótima: x1 = {solucao[0]:.4f}, x2 = {solucao[1]:.4f}")
    print(f"Valor ótimo:   f(x) = {valor_otimo:.4f}")

    # Confere com a resposta exata: x = (10/3, 4/3), f = 76/3
    assert np.allclose(solucao, [10 / 3, 4 / 3]) and np.isclose(valor_otimo, 76 / 3)
    print("Confere com a solução exata (10/3, 4/3) e f = 76/3.")
