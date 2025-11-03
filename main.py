import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import CSS4_COLORS, to_rgb
from itertools import permutations
from collections import deque
import sys
import time
import matplotlib.pyplot as plt
from assets import *


# np.random.seed(42)  # para reproduzibilidade

n_tarefas = 10
n_maquinas = 4
# Gerar matriz de tempos
tp = np.random.randint(1, 11, size=(n_maquinas, n_tarefas))
# Variável que guarda os melhores makespan, é uma lista de dicionários
global populacao, MAX_POPULACAO
# Utilizando deque para remover o primeiro elemento (fila)
populacao = deque([])
MAX_POPULACAO = 100
melhor_tempo = 10000000000000

def testar_solucoes(n_tarefas, n_maquinas, m_tempos):
    while(True):
        # Gera um indivíduo aleatório
        solucao = gerar_individuo(n_tarefas)
        # Gera a matriz de makespan
        m_makespan = np.zeros((n_maquinas, n_tarefas), dtype=int)
        # Completa a matriz m_makespan com o makespan correspondente
        calcular_makespan(m_tempos, solucao, m_makespan)
        
        fitnees(m_makespan, m_tempos, solucao)
        
        # Precisa fazer o crossover (juntar população com um indivíduo aleatório)
        
        pai1 = selecionar(populacao, "torneio")
        pai2 = selecionar([p for p in populacao if p != pai1], "torneio")
        if(pai2):
            cros_result = order_crossover(pai1, pai2)
            new_makespan = np.zeros((n_maquinas, n_tarefas), dtype=int)
            calcular_makespan(m_tempos, cros_result, new_makespan)
            fitnees(m_makespan, m_tempos, cros_result)

        
        # Depois aplica o fitnees no filho do crosover 
    
    
def fitnees(m_makespan, m_tempos, solucao):
    '''
    Avalia o makespan da solução para saber se é ou não uma boa solução
    Se for uma boa solução, adiciona a populacao
    '''
    # Pega a quantidade de linhas e columnas
    global melhor_tempo
    linhas, colunas = m_makespan.shape
    # O último elemento é o tempo total de processar todas as peças
    ftn = float(m_makespan[linhas-1][colunas-1])
    # Verifica se é uma boa solução, se for, adiciona na populacao para o crossover
    media_tempos = float(pegar_media_makespan(populacao))

    # print(f"Solução: {solucao} | makespan: {ftn}")

    if ftn < media_tempos:
        populacao.append({"solucao": solucao, "mk_value": ftn})
        # print(f"Melhor tempo: {melhores_tempos}")
        print(f"Solução add a população: {solucao} | makespan: {ftn}")
        # if ftn < melhor_tempo:
        atualizar_tela(m_makespan, m_tempos, solucao)
        print(f"Melhor solucao {solucao}\nMelhor Tempo {ftn}")
            
        melhor_tempo = ftn
    # time.sleep(0.1)
    if len(populacao) > MAX_POPULACAO:
        populacao.popleft()
        

    

testar_solucoes(n_tarefas, n_maquinas, tp)
