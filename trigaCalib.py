"""
TrigaCalib is a software multi-system to calibrate the control bars
of Nuclear Reator Triga IPR-R1 by some CSV data file of input.
Copyright (C) 2024 Thalles Campagnani

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.optimize import curve_fit

def calculate_dBar_dRea(caminho_arquivo,name_pdf,janela=10, valor_cruzamento_positivo=2, valor_cruzamento_negativo=-2, tempo_acomodacao=60):
    """
    Ler arquivo CSV
    Descobrir se colunas de tempo do PLC estão disponíveis (PLC_CONV_TIME_ ou PLC_ORIG_TIME_) e importar dados
    Descobrir se coluna de barra de regulação está disponível (PLC_CONV_BarraCon) e importar dados
    Descobrir se coluna do canal logaritimo aquisição logarítima está disponível (PLC_CONV_CLogALog) e importar dados
    Converter colunas de tempo disponíveis em segundos
    Tirar offset de tempo (inicio da aquisição se torna 0s)
    Remover tempos repetidos
    Derivar barra de controle 
    Achar tempo que cruza limiar de velocidade positivo
    Achar tempo que cruza limiar de velocidade negativo 

    Extrair janelas de tempo
        Janela 1: prova de criticalidade
        Janela 2: subida de barra
        Janela 3: acomodação de nêutrons atrasados
        Janela 4: período estável

    Fazer média da posição de barra da janela 1
    Fazer média da posição de barra da janela 3 e 4
    Calcular delta de barra
    Regressão linear na janela 4
    Calcular período
    Calcular reatividade

    Gerar PDF com tabelas e gráficos
    Retornar barra_inicial, barra_final e deltaRearividade

    """
    dados = {}
    
    # Ler arquivo CSV
    df = pd.read_csv(caminho_arquivo, delimiter=';')
    
    # Descobrir se colunas de tempo do PLC estão disponíveis (PLC_CONV_TIME_ ou PLC_ORIG_TIME_) e importar dados
    if 'PLC_CONV_TIME_S' in df.columns:
        tempo = df['PLC_CONV_TIME_S'].values
        if 'PLC_CONV_TIME_H' in df.columns:
            tempo += df['PLC_CONV_TIME_H'].values*3600
        if 'PLC_CONV_TIME_Mi' in df.columns:
            tempo += df['PLC_CONV_TIME_Mi'].values*60
        if 'PLC_CONV_TIME_MS' in df.columns:
            tempo = tempo.astype(float) + df['PLC_CONV_TIME_MS'].values/1000
    elif 'PLC_ORIG_TIME_S' in df.columns:
        tempo = df['PLC_ORIG_TIME_S'].values
        if 'PLC_ORIG_TIME_H' in df.columns:
            tempo += df['PLC_ORIG_TIME_H'].values*3600
        if 'PLC_ORIG_TIME_Mi' in df.columns:
            tempo += df['PLC_ORIG_TIME_Mi'].values*60
        if 'PLC_ORIG_TIME_MS' in df.columns:
            tempo = tempo.astype(float) + df['PLC_ORIG_TIME_MS'].values/1000
    else:
        print("Error: import PLC_CONV_TIME_S or PLC_ORIG_TIME_S from ", caminho_arquivo)
        return None
    
    dados['tempo'] = tempo - tempo[1]
    
    # Descobrir se coluna de barra de regulação está disponível (PLC_CONV_BarraCon) e importar dados
    if 'PLC_CONV_BarraCon' in df.columns:
        dados['posicao'] = df['PLC_CONV_BarraCon'].values
    else:
        print("Error: import PLC_CONV_BarraCon from ", caminho_arquivo)
        return None
    
    # Descobrir se coluna do canal logaritimo aquisição logarítima está disponível (PLC_CONV_CLogALog) e importar dados
    if 'PLC_CONV_CLogALog' in df.columns:
        dados['potencia'] = df['PLC_CONV_CLogALog'].values
    else:
        print("Error: import PLC_CONV_CLogALog from ", caminho_arquivo)
        return None
    
    # Remover tempos repetidos
    tempo = [dados['tempo'][0]]
    posicao = [dados['posicao'][0]]
    potencia = [dados['potencia'][0]]
    for i in range(1, len(dados['tempo'])):
        if dados['tempo'][i] != dados['tempo'][i - 1]:
            tempo.append(dados['tempo'][i])
            posicao.append(dados['posicao'][i])
            potencia.append(dados['potencia'][i])
    dados['tempo'] = np.array(tempo)
    dados['posicao'] = np.array(posicao)
    dados['potencia'] = np.array(potencia)

    # Aplicar filtro de média móvel na posição da barra e derivar
    filtro = np.ones(janela) / janela
    posicao_filtrada = np.convolve(dados['posicao'], filtro, mode='same')
    derivada_posicao = np.diff(posicao_filtrada) / np.diff(dados['tempo'])
    derivada_posicao = np.insert(derivada_posicao, 0, np.nan) # Para que a derivada tenha o mesmo comprimento que os dados originais, podemos adicionar um NaN no início

    # Deleta a quantidade de janela da primeira e ultimas posições
    tempo_cortado = dados['tempo'][janela:-janela]
    derivada_posicao_cortada = derivada_posicao[janela:-janela]

    # Calcular tempos de cruzamento positivo
    cruzamentos_acima_positivo = []
    cruzamentos_abaixo_positivo = []
    for i in range(1, len(derivada_posicao_cortada)):
        if derivada_posicao_cortada[i - 1] <= valor_cruzamento_positivo < derivada_posicao_cortada[i]:
            cruzamentos_acima_positivo.append(tempo_cortado[i])
        elif derivada_posicao_cortada[i - 1] > valor_cruzamento_positivo >= derivada_posicao_cortada[i]:
            cruzamentos_abaixo_positivo.append(tempo_cortado[i])
    
    # Calcular tempos de cruzamento negativo
    cruzamentos_acima_negativo = []
    cruzamentos_abaixo_negativo = []
    for i in range(1, len(derivada_posicao_cortada)):
        if derivada_posicao_cortada[i - 1] <= valor_cruzamento_negativo < derivada_posicao_cortada[i]:
            cruzamentos_acima_negativo.append(tempo_cortado[i])
        elif derivada_posicao_cortada[i - 1] > valor_cruzamento_negativo >= derivada_posicao_cortada[i]:
            cruzamentos_abaixo_negativo.append(tempo_cortado[i])
    
    # Definir as janelas de tempo
    janela1 = {
        'tempo': dados['tempo'][dados['tempo'] <= cruzamentos_acima_positivo[0]],
        'posicao': dados['posicao'][dados['tempo'] <= cruzamentos_acima_positivo[0]],
        'potencia': dados['potencia'][dados['tempo'] <= cruzamentos_acima_positivo[0]]
    }

    janela2 = {
        'tempo': dados['tempo'][(dados['tempo'] > cruzamentos_acima_positivo[0]) & (dados['tempo'] <= cruzamentos_abaixo_positivo[0])],
        'posicao': dados['posicao'][(dados['tempo'] > cruzamentos_acima_positivo[0]) & (dados['tempo'] <= cruzamentos_abaixo_positivo[0])],
        'potencia': dados['potencia'][(dados['tempo'] > cruzamentos_acima_positivo[0]) & (dados['tempo'] <= cruzamentos_abaixo_positivo[0])]
    }

    tempo_apos_cruzamento = cruzamentos_abaixo_positivo[0] + tempo_acomodacao
    indice_mais_proximo = np.abs(dados['tempo'] - tempo_apos_cruzamento).argmin()
    janela3 = {
        'tempo': dados['tempo'][(dados['tempo'] > cruzamentos_abaixo_positivo[0]) & (dados['tempo'] <= dados['tempo'][indice_mais_proximo])],
        'posicao': dados['posicao'][(dados['tempo'] > cruzamentos_abaixo_positivo[0]) & (dados['tempo'] <= dados['tempo'][indice_mais_proximo])],
        'potencia': dados['potencia'][(dados['tempo'] > cruzamentos_abaixo_positivo[0]) & (dados['tempo'] <= dados['tempo'][indice_mais_proximo])]
    }

    if len(cruzamentos_acima_positivo) >= 2 and len(cruzamentos_abaixo_negativo) >= 1:
        if cruzamentos_acima_positivo[1] > cruzamentos_abaixo_negativo[0]:
            final_janela4 = cruzamentos_abaixo_negativo[0]
        else:
            final_janela4 = cruzamentos_acima_positivo[1]
    else:
        if len(cruzamentos_acima_positivo) >= 2:
            final_janela4 = cruzamentos_abaixo_negativo[1]
        elif len(cruzamentos_abaixo_negativo) >= 1:
            final_janela4 = cruzamentos_acima_positivo[0]
        final_janela4 = dados['tempo'][-1]

    janela4 = {
        'tempo': dados['tempo'][(dados['tempo'] > dados['tempo'][indice_mais_proximo]) & (dados['tempo'] <= final_janela4)],
        'posicao': dados['posicao'][(dados['tempo'] > dados['tempo'][indice_mais_proximo]) & (dados['tempo'] <= final_janela4)],
        'potencia': dados['potencia'][(dados['tempo'] > dados['tempo'][indice_mais_proximo]) & (dados['tempo'] <= final_janela4)]
    }
    
    posicao_inicial_barra = np.mean(janela1['posicao']) if len(janela1['posicao']) > 0 else np.nan
    posicao_final_barra = np.mean(janela4['posicao']) if len(janela4['posicao']) > 0 else np.nan


    
    # Regressão exponencial
    def exponencial(x, a, b):
        return a * np.exp(b * x)
    popt, pcov = curve_fit(exponencial, janela4['tempo'], janela4['potencia'], p0=[1, 0.1]) # p0 são os valores iniciais para a otimização
    a, b = popt
    
    # Calcaular periodo
    periodo = 1/b

    # Calular reatividade
    Beff = 0.007
    l    = 0.000073
    B1   = 0.00021
    A1   = 0.01243982736
    B2   = 0.00141
    A2   = 0.03050823858
    B3   = 0.00127
    A3   = 0.1114384535
    B4   = 0.00255
    A4   = 0.3013683394
    B5   = 0.00074
    A5   = 1.136306853
    B6   = 0.00027
    A6   = 3.013683394
    reatividade_PCM  = 100000*((l/periodo)
                               +(B1/(1+A1*periodo))
                               +(B2/(1+A2*periodo))
                               +(B3/(1+A3*periodo))
                               +(B4/(1+A4*periodo))
                               +(B5/(1+A5*periodo))
                               +(B6/(1+A6*periodo)))
    
    # Gerar dados ajustados para visualização
    #x_fit = np.linspace(min(janela2['tempo']), max(janela4['tempo']), 100)
    #y_fit = exponencial(x_fit, *popt)

    # Criação do PDF
    with PdfPages(name_pdf) as pdf:
        
        # Criação da figura
        plt.figure(figsize=(8.27, 11.69))
        
        # Adiciona a tabela na figura
        tabela_dados = [
            ["Posição Inicial da Barra de Regulação:", f"{posicao_inicial_barra:.2f}"],
            ["Posição Final da Barra de Regulação:", f"{posicao_final_barra:.2f}"],
            ["Delta da Posição da Barra de Regulação:", f"{posicao_final_barra-posicao_inicial_barra:.2f}"],
            ["Delta de Reatividade Encontrado (PCM):", reatividade_PCM],
            ["Período encontrado (s):", f"{periodo:.2f}"],
            ["Coeficientes da exponencial ajustada [a, b]:", [float(a),float(b)]],
            ["Janela de tempo 1 (s):", "[" + f"{0:.2f}" + " , " + f"{janela1['tempo'][-1]:.2f}" + "]"],
            ["Janela de tempo 2 (s):", "[" + f"{janela2['tempo'][0]:.2f}" + " , " + f"{janela2['tempo'][-1]:.2f}" + "]"],
            ["Janela de tempo 3 (s):", "[" + f"{janela3['tempo'][0]:.2f}" + " , " + f"{janela3['tempo'][-1]:.2f}" + "]"],
            ["Janela de tempo 4 (s):", "[" + f"{janela4['tempo'][0]:.2f}" + " , " + f"{janela4['tempo'][-1]:.2f}" + "]"],
        ]
        
        # Define os títulos das colunas
        col_labels = ['Descrição', 'Valor']
        # Adiciona a tabela ao gráfico
        plt.subplot(3, 1, 1)
        ax = plt.gca()
        ax.axis('off')  # Remove o eixo
        plt.table(cellText=tabela_dados, colLabels=col_labels, loc='center', cellLoc='left', colColours=['lightgrey', 'lightgrey'])
        
        # Ajusta o layout para a tabela
        plt.title('Relatório n° 1 de Inserção de Reatividade (BarraReg)', fontsize=16)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)  # Ajuste para deixar espaço para a tabela
        
        # Adiciona os gráficos abaixo da tabela
        #plt.figure(figsize=(15, 10))  # Redefine o tamanho da figura para os gráficos
        plt.subplot(3, 1, 2)
        plt.plot(janela1['tempo'], janela1['posicao'], linestyle='-', color='r', label='Janela 1')
        plt.plot(janela2['tempo'], janela2['posicao'], linestyle='-', color='b', label='Janela 2')
        plt.plot(janela3['tempo'], janela3['posicao'], linestyle='-', color='g', label='Janela 3')
        plt.plot(janela4['tempo'], janela4['posicao'], linestyle='-', color='y', label='Janela 4')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posição da Barra de Regulação')
        plt.title('Posição da Barra de Regulação em função do Tempo')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(janela1['tempo'], janela1['potencia'], linestyle='-', color='r', label='Janela 1')
        plt.plot(janela2['tempo'], janela2['potencia'], linestyle='-', color='b', label='Janela 2')
        plt.plot(janela3['tempo'], janela3['potencia'], linestyle='-', color='g', label='Janela 3')
        plt.plot(janela4['tempo'], janela4['potencia'], linestyle='-', color='y', label='Janela 4')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Potência (W)')
        plt.title('Potência em função do Tempo')
        plt.legend()
        plt.grid(True)
        
        # Ajusta o layout para evitar sobreposição
        plt.tight_layout()
        
        # Salva a figura no PDF
        pdf.savefig()  
        plt.close()
    
    return posicao_inicial_barra, posicao_final_barra, reatividade_PCM 

calculate_dBar_dRea('aqui3-100cut.csv','relatorio.pdf')

