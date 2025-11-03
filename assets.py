import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import CSS4_COLORS, to_rgb
from itertools import permutations
import sys, os, random

plt.ion()

fig, ax = plt.subplots()
linha = None


# Exemplo com números diferentes por linha
dados = [
    {"estagio": "Máquina-01", "atividades": [(0, 3), (3, 2), (5, 1)], "numeros": [1, 2, 3]},
    {"estagio": "Máquina-02", "atividades":[(3, 5), (10, 4), (15, 2)], "numeros": [2, 1, 3, 4]},
    {"estagio": "Máquina-03", "atividades": [(15, 3), (19, 2), (23, 2), (26, 1)], "numeros": [1,  2, 9, 10]},
    {"estagio": "Máquina-04", "atividades": [(21, 3)], "numeros": [105]}
]

# gantt_matplotlib(dados)




# Gerar tempos aleatórios entre 1 e 10 para cada tarefa e máquina
# print('Tempos de processamento de cada tarefa j em cada máquina i')
# print(tempos)

# Gerar a matriz de cálculo de makespan
# makespan = np.zeros((n_maquinas, n_tarefas), dtype=int)

# Uma permutação
#permuta = np.array([0, 1, 2, 3, 4])
#permuta = np.array([1, 3, 2, 0, 4]) #ótimo
# permuta = np.array([0, 3, 4, 1, 2])   #pior solução
# Calcular o makespan
# calcular_makespan(tempos, permuta, makespan)

# print('Makespan, tempo de termino de cada tarefa j em cada máquina i, mas as colunas respeitam a ordem de tarefas em permuta')
# print(makespan)

# dadosGantt = matriz_para_gantt_permutacao_zero(makespan, tempos, permuta)
# gantt_matplotlib(dadosGantt)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Funções de crossover
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def order_crossover(pai1, pai2):
    n = len(pai1)
    a, b = sorted(random.sample(range(n), 2))
    filho = [None] * n
    filho[a:b] = pai1[a:b]

    pos = b
    for x in pai2[b:] + pai2[:b]:
        if x not in filho:
            if pos >= n:
                pos = 0
            filho[pos] = x
            pos += 1
    return filho


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Função que define o valor do makespan para entrar pra população
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def pegar_media_makespan(populacao):
    '''Define qual é o valor necessário para entrar para a populacao'''
    if len(populacao) == 0:
        return float(1000000000000)
    # Definindo pela média
    media = 0
    qtd = 0
    for elemento in populacao:
        media += elemento["mk_value"]
        qtd += 1
    media /= qtd
    # print(f"A média é: {media}")
    return float(media)
    # ultimo = populacao[-1]
    # # print(f"Melhor tempo: {elemento["mk_value"]}")
    # return ultimo["mk_value"]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Função de Makespan
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calcular_makespan(tempos, permutacao, resultado):
    n, m = tempos.shape
    #inicializa o primeiro tempo da primeira tarefa da permutação na máquina 0
    resultado[0, 0] = tempos[0, permutacao[0]]

    for j in range(1, m):
      resultado[0, j] = tempos[0, permutacao[j] ] + resultado[0, j-1]
    #inicializa o tempo em que a primeira tarefa inicia em cada uma das máquinas
    for i in range(1, n):
      resultado[i, 0] = tempos[i, permutacao[0] ] + resultado[i-1, 0]
    #Calcula agora o tempo de término nas demais máquinas para as demais tarefas
    for i in range(1, n):
        for j in range(1, m):
            resultado[i, j] = tempos[i, permutacao[j] ] + max(resultado[i-1, j], resultado[i, j-1])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Funções de geração de indivíduos (retorna uma sequência de indivíduos), geralmente servem de auxiliar para o crossover
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gerar_individuo(n_tarefas):
    random.seed(os.urandom(8))  # entropia real do sistema
    lista = list(range(n_tarefas))
    for i in range(n_tarefas - 1, 0, -1):
        j = random.randint(0, i)
        lista[i], lista[j] = lista[j], lista[i]
    return lista


def selecionar_individuo_populacao(populacao):
    """Seleciona um indivíduo aleatório da população."""
    if not populacao:
        return None

    elemento = random.choice(populacao)
    return elemento.get("solucao", None)


def selecionar_individuo_roleta(populacao):
    """Seleciona um indivíduo via roleta proporcional ao fitness (1/makespan)."""
    if not populacao:
        return None

    mk_values = [p.get("mk_value", float("inf")) for p in populacao]
    fitness = np.array([1.0 / mk for mk in mk_values if mk > 0], dtype=float)

    if len(fitness) < len(populacao):  # Evita erro caso haja mk_value = 0
        fitness = np.pad(fitness, (0, len(populacao) - len(fitness)), constant_values=1e-6)

    prob = fitness / fitness.sum()
    idx = np.random.choice(len(populacao), p=prob)
    return populacao[idx].get("solucao", None)


def selecionar_individuo_torneio(populacao, k=3):
    """Seleciona o melhor indivíduo entre k amostras aleatórias."""
    if not populacao:
        return None

    k = min(k, len(populacao))
    torneio = random.sample(populacao, k)
    vencedor = min(torneio, key=lambda x: x.get("mk_value", float("inf")))
    return vencedor.get("solucao", None)


def selecionar_melhores_e_variedade(populacao, taxa_elitismo=0.1):
    """
    Retorna uma nova população contendo a elite (melhores indivíduos)
    e o restante selecionado aleatoriamente para manter diversidade.
    """
    if not populacao:
        return []

    populacao_ordenada = sorted(populacao, key=lambda x: x.get("mk_value", float("inf")))
    n_elite = max(1, int(len(populacao) * taxa_elitismo))

    elite = populacao_ordenada[:n_elite]
    n_resto = max(0, len(populacao) - n_elite)

    resto = random.sample(populacao, n_resto) if n_resto > 0 else []
    nova_pop = elite + resto

    # Retorna apenas as soluções (permutações)
    return [ind.get("solucao", None) for ind in nova_pop]

def selecionar(populacao, metodo="torneio"):
    if metodo == "roleta":
        return selecionar_individuo_roleta(populacao)
    elif metodo == "populacao":
        return selecionar_individuo_populacao(populacao)
    elif metodo == "melhores":
        return selecionar_melhores_e_variedade(populacao)
    else:
        return selecionar_individuo_torneio(populacao)
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Funções gráficas
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def matriz_para_gantt_permutacao_zero(tempos_termino, duracoes, numeros):
    """
    Converte matriz de tempos de término e duração para gantt_matplotlib
    considerando que a sequência de permutação inclui a tarefa 0.

    tempos_termino: numpy array [n_maquinas x n_tarefas] (termino da permutação)
    duracoes: numpy array [n_maquinas x n_tarefas] (duração original das tarefas)
    numeros: array 1D da permutação [tarefa0, tarefa1, ..., tarefa_n]
    """
    n_maquinas, n_tarefas = tempos_termino.shape
    dados_gantt = []

    for i in range(n_maquinas):
        atividades = []
        for j in range(n_tarefas):
            tarefa_real = numeros[j]  # agora já está correto (0-based)
            duracao = duracoes[i, tarefa_real]
            termino = tempos_termino[i, j]
            inicio = termino - duracao
            atividades.append((inicio, duracao))

        dados_gantt.append({
            "estagio": f"Máquina {i}",
            "atividades": atividades,
            "numeros": numeros  # rótulos da permutação
        })

    return dados_gantt
def cor_contraste(cor):
    r, g, b = to_rgb(cor)
    luminancia = 0.299*r + 0.587*g + 0.114*b
    return "black" if luminancia > 0.5 else "white"
def gerar_100_cores():
    """Gera uma lista de 100 cores nomeadas distintas do CSS4."""
    cores = list(CSS4_COLORS.keys())
    return cores[:100]  # pega as 100 primeiras
def gantt_matplotlib(dados, ax=None, grid=True):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(dados)
    cores_base = gerar_100_cores()  # 100 cores nomeadas
    n = len(df)
    altura_retangulo = 0.6
    fontsize_numero = 9

    # Cria figura apenas se ax não for passado
    if ax is None:
        altura_figura = max(2, n * (altura_retangulo + 0.1))
        fig, ax = plt.subplots(figsize=(10, altura_figura))

    ax.clear()  # limpa dados antigos

    for i, row in df.iterrows():
        y_pos = n - 1 - i
        numeros = row.get("numeros", list(range(1, len(row["atividades"]) + 1)))

        for (start, dur), num in zip(row["atividades"], numeros):
            cor_caixa = cores_base[(num - 1) % len(cores_base)]
            cor_numero = cor_contraste(cor_caixa)

            ax.broken_barh(
                [(start, dur)],
                (y_pos - altura_retangulo/2, altura_retangulo),
                facecolors=cor_caixa,
                edgecolors="black",
                linewidth=1.2
            )

            ax.text(
                start + dur/2,
                y_pos,
                str(num),
                va="center",
                ha="center",
                color=cor_numero,
                fontsize=fontsize_numero,
                weight="bold"
            )

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["estagio"][::-1])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Máquinas")
    ax.set_title("Diagrama de Gantt - Makespan")

    if grid:
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Atualiza a figura sem bloquear
    plt.draw()
    plt.pause(0.001)
def atualizar_tela(m_makespan, tempos, solucao):
    global linha, ax
    
    arr_dados = matriz_para_gantt_permutacao_zero(m_makespan, tempos, solucao)
    
    ax.clear()  # limpa o gráfico atual
    gantt_matplotlib(arr_dados, ax=ax)  # supondo que você adapte a função para receber ax

    plt.draw()      # redesenha o gráfico
    plt.pause(0.001)  # pequena pausa para atualizar a interface
    
