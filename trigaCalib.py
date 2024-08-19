#!/bin/python
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
import argparse
import shutil
import sys
import os
from PyPDF2 import PdfMerger


class trigaCalib:
    dBar_dRea = []
    indice_relatorio = 0
    
    def concatenate_pdfs(self, output_filename):
        tmp_dir = './tmp'
        
        # Lista todos os arquivos PDF na pasta tmp
        pdf_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith('.pdf')]
        
        # Ordenar os arquivos com o PDF 'final.pdf' no início e os outros em ordem numérica
        pdf_files.sort(key=lambda x: (not x.endswith('final.pdf'), int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else float('inf')))

        # Cria um PdfMerger para combinar os PDFs
        merger = PdfMerger()

        for pdf in pdf_files:
            merger.append(pdf)

        merger.write(output_filename)
        merger.close()
        shutil.rmtree(tmp_dir)

    def process_file_get_dBar_dRea(self, caminho_arquivo, simples):
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
        dados       = self.import_file(caminho_arquivo)
        self.plot_input_data(dados)
        
        cruzamentos = self.calcula_tempos_cruzamento(dados)
        
        if simples:
            janela1, janela2, janela3, janela4, janela5 = self.gera_janela_simples(dados, cruzamentos, 15)
            barra, reatividade_PCM = self.calculate_dBar_dRea(janela1, janela2, janela3, janela4, janela5)
            self.dBar_dRea.append([barra['inicial'], barra['final'], reatividade_PCM])
        else:
            janela1, janela2, janela3, janela4, janela5 = self.gera_janela_multi(dados, cruzamentos)
            for i in range(0,len(janela1)):
                barra, reatividade_PCM = self.calculate_dBar_dRea(janela1[i], janela2[i], janela3[i], janela4[i], janela5[i])
                self.dBar_dRea.append([barra['inicial'], barra['final'], reatividade_PCM])
        
    def import_file(self, caminho_arquivo):
        
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
            dados['posicao'] = df['PLC_CONV_BarraReg'].values
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
        
        return dados

    def plot_input_data(self,dados):
        # Criar uma figura e eixos com subplots
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # 2 linhas, 1 coluna

        # Primeiro subplot: Posição em função do tempo
        ax[0].plot(dados['tempo'], dados['posicao'], label='Posição', color='blue')
        ax[0].set_xlabel('Tempo (s)')
        ax[0].set_ylabel('Posição')
        ax[0].set_title('Posição em função do tempo')
        ax[0].legend()
        ax[0].grid(True)

        # Segundo subplot: Potência em função do tempo
        ax[1].plot(dados['tempo'], dados['potencia'], label='Potência', color='green')
        ax[1].set_xlabel('Tempo (s)')
        ax[1].set_ylabel('Potência (W)')
        ax[1].set_title('Potência em função do tempo')
        ax[1].set_yscale('log')
        ax[1].legend()
        ax[1].grid(True)

        # Ajustar layout e exibir a figura com os dois gráficos
        plt.tight_layout()
        plt.show()
    
    def calcula_tempos_cruzamento(self, dados, janela_filtro=10, valor_cruzamento_positivo=3, valor_cruzamento_negativo=-3):

        # Aplicar filtro de média móvel na posição da barra e derivar
        filtro = np.ones(janela_filtro) / janela_filtro
        posicao_filtrada = np.convolve(dados['posicao'], filtro, mode='same')
        derivada_posicao = np.diff(posicao_filtrada) / np.diff(dados['tempo'])
        derivada_posicao = np.insert(derivada_posicao, 0, np.nan) # Para que a derivada tenha o mesmo comprimento que os dados originais, podemos adicionar um NaN no início

        # Deleta a quantidade de janela da primeira e ultimas posições
        dados_cortado_tempo = dados['tempo'][janela_filtro:-janela_filtro]
        dados_cortado_derivada = derivada_posicao[janela_filtro:-janela_filtro]

        #Descomentar para exibir grafico da derivada da posição
        #plt.plot(dados_cortado_tempo, dados_cortado_derivada, label='Derivada', color='blue')
        #plt.xlabel('tempo')
        #plt.ylabel('Derivada')
        #plt.title('Derivada da posição da barra')
        #plt.legend()
        #plt.grid(True)
        #plt.show()

        # Calcular tempos de cruzamento positivo
        cruzamentos = {}
        cruzamentos['acima_positivo']  = []
        cruzamentos['abaixo_positivo'] = []
        for i in range(1, len(dados_cortado_derivada)):
            if dados_cortado_derivada[i - 1] <= valor_cruzamento_positivo < dados_cortado_derivada[i]:
                cruzamentos['acima_positivo'].append(dados_cortado_tempo[i])
            elif dados_cortado_derivada[i - 1] > valor_cruzamento_positivo >= dados_cortado_derivada[i]:
                cruzamentos['abaixo_positivo'].append(dados_cortado_tempo[i])
        
        # Calcular tempos de cruzamento negativo
        cruzamentos['acima_negativo']  = []
        cruzamentos['abaixo_negativo'] = []
        for i in range(1, len(dados_cortado_derivada)):
            if dados_cortado_derivada[i - 1] <= valor_cruzamento_negativo < dados_cortado_derivada[i]:
                cruzamentos['acima_negativo'].append(dados_cortado_tempo[i])
            elif dados_cortado_derivada[i - 1] > valor_cruzamento_negativo >= dados_cortado_derivada[i]:
                cruzamentos['abaixo_negativo'].append(dados_cortado_tempo[i])
        
        print(len(cruzamentos['acima_positivo']),  "acima_positivo:\t",  cruzamentos['acima_positivo'])
        print(len(cruzamentos['abaixo_positivo']), "abaixo_positivo:\t", cruzamentos['abaixo_positivo'])
        print(len(cruzamentos['abaixo_negativo']), "abaixo_negativo:\t", cruzamentos['abaixo_negativo'])
        print(len(cruzamentos['acima_negativo']),  "acima_negativo:\t",  cruzamentos['acima_negativo'])
        print()
        
        return cruzamentos

    def gera_janela_simples(self, dados, cruzamentos, tempo_acomodacao=60):
        """
        Gera um único conjunto de janelas
        A partir de uma segunda movimentação da barra, todos os dados são ignorados
        """
        # Verificar se há cruzamentos 'abaixo_negativo' e ajustar dados e cruzamentos
        # (Se a barra foi movimentada para baixo)
        if len(cruzamentos['abaixo_negativo']) >= 1:
            tempo_maximo = cruzamentos['abaixo_negativo'][0]
            dados = {
                'tempo':    dados['tempo'][dados['tempo']    <= tempo_maximo],
                'posicao':  dados['posicao'][dados['tempo']  <= tempo_maximo],
                'potencia': dados['potencia'][dados['tempo'] <= tempo_maximo]
            }
            # Atualizar cruzamentos
            cruzamentos['acima_positivo']  = [t for t in cruzamentos['acima_positivo']  if t <= tempo_maximo]
            cruzamentos['abaixo_positivo'] = [t for t in cruzamentos['abaixo_positivo'] if t <= tempo_maximo]
        
        # remove tempos de cruzamento muito perto um do outro
        i = 0
        while i < len(cruzamentos['acima_positivo']) - 1:
            if (cruzamentos['acima_positivo'][i + 1] - cruzamentos['acima_positivo'][i]) < tempo_acomodacao:
                # Se a diferença é menor que tempo_acomodacao, remover o elemento i
                cruzamentos['acima_positivo'].pop(i+1)#Remove o próximo muito perto
            else:
                i += 1

        i = 0
        while i < len(cruzamentos['abaixo_positivo']) - 1:
            if (cruzamentos['abaixo_positivo'][i + 1] - cruzamentos['abaixo_positivo'][i]) < tempo_acomodacao:
                # Se a diferença é menor que tempo_acomodacao, remover o elemento i
                cruzamentos['abaixo_positivo'].pop(i)#Remove o anterior muito perto
            else:
                i += 1
                
        # Janela 1: 0 segundos até tempo_acomodacao
        janela1 = {
            'tempo':    dados['tempo'][dados['tempo']    <= (tempo_acomodacao + dados['tempo'][0])],
            'posicao':  dados['posicao'][dados['tempo']  <= (tempo_acomodacao + dados['tempo'][0])],
            'potencia': dados['potencia'][dados['tempo'] <= (tempo_acomodacao + dados['tempo'][0])]
        }

        # Janela 2: tempo_acomodacao até cruzamentos['acima_positivo']
        janela2 = {
            'tempo':    dados['tempo'][(dados['tempo']    > tempo_acomodacao + dados['tempo'][0]) & (dados['tempo'] <= cruzamentos['acima_positivo'][0])],
            'posicao':  dados['posicao'][(dados['tempo']  > tempo_acomodacao + dados['tempo'][0]) & (dados['tempo'] <= cruzamentos['acima_positivo'][0])],
            'potencia': dados['potencia'][(dados['tempo'] > tempo_acomodacao + dados['tempo'][0]) & (dados['tempo'] <= cruzamentos['acima_positivo'][0])]
        }

        # Janela 3: cruzamentos['acima_positivo'] até cruzamentos['abaixo_positivo']
        janela3 = {
            'tempo':    dados['tempo'][(dados['tempo']    > cruzamentos['acima_positivo'][0]) & (dados['tempo'] <= cruzamentos['abaixo_positivo'][0])],
            'posicao':  dados['posicao'][(dados['tempo']  > cruzamentos['acima_positivo'][0]) & (dados['tempo'] <= cruzamentos['abaixo_positivo'][0])],
            'potencia': dados['potencia'][(dados['tempo'] > cruzamentos['acima_positivo'][0]) & (dados['tempo'] <= cruzamentos['abaixo_positivo'][0])]
        }

        # Janela 4: cruzamentos['abaixo_positivo'] até cruzamentos['abaixo_positivo'] + tempo_acomodacao
        tempo_apos_cruzamento = cruzamentos['abaixo_positivo'][0] + tempo_acomodacao
        janela4 = {
            'tempo':    dados['tempo'][(dados['tempo']    > cruzamentos['abaixo_positivo'][0]) & (dados['tempo'] <= tempo_apos_cruzamento)],
            'posicao':  dados['posicao'][(dados['tempo']  > cruzamentos['abaixo_positivo'][0]) & (dados['tempo'] <= tempo_apos_cruzamento)],
            'potencia': dados['potencia'][(dados['tempo'] > cruzamentos['abaixo_positivo'][0]) & (dados['tempo'] <= tempo_apos_cruzamento)]
        }

        # Determinando o final da janela 5
        # A partir de qualquer movimentação da barra de regulação, defina o final da janela
        if len(cruzamentos['acima_positivo']) >= 2:
            final_janela5 = cruzamentos['acima_positivo'][1]
        else:
            final_janela5 = dados['tempo'][-1]

        # Janela 5: cruzamentos['abaixo_positivo'] + tempo_acomodacao até final_janela5
        janela5 = {
            'tempo':    dados['tempo'][(dados['tempo']    > tempo_apos_cruzamento) & (dados['tempo'] <= final_janela5)],
            'posicao':  dados['posicao'][(dados['tempo']  > tempo_apos_cruzamento) & (dados['tempo'] <= final_janela5)],
            'potencia': dados['potencia'][(dados['tempo'] > tempo_apos_cruzamento) & (dados['tempo'] <= final_janela5)]
        }

        return janela1, janela2, janela3, janela4, janela5

            
    def gera_janela_multi(self, dados, cruzamentos, tempo_acomodacao=60):
        """
        Gera vários conjuntos de janelas
        Só analiza movimentações positivas, cada uma definindo uma janela
        A partir de alguma movimentação negativa os dados são descartados
        """
        
        # Verificar se há cruzamentos 'abaixo_negativo' e ajustar dados e cruzamentos
        # (Se a barra foi movimentada para baixo)
        if len(cruzamentos['abaixo_negativo']) >= 1:
            tempo_maximo = cruzamentos['abaixo_negativo'][0]
            dados = {
                'tempo':    dados['tempo'][dados['tempo']    <= tempo_maximo],
                'posicao':  dados['posicao'][dados['tempo']  <= tempo_maximo],
                'potencia': dados['potencia'][dados['tempo'] <= tempo_maximo]
            }
            # Atualizar cruzamentos
            cruzamentos['acima_positivo']  = [t for t in cruzamentos['acima_positivo']  if t <= tempo_maximo]
            cruzamentos['abaixo_positivo'] = [t for t in cruzamentos['abaixo_positivo'] if t <= tempo_maximo]
        
        # remove tempos de cruzamento muito perto um do outro
        i = 0
        while i < len(cruzamentos['acima_positivo']) - 1:
            if (cruzamentos['acima_positivo'][i + 1] - cruzamentos['acima_positivo'][i]) < tempo_acomodacao:
                # Se a diferença é menor que tempo_acomodacao, remover o elemento i
                cruzamentos['acima_positivo'].pop(i+1)#Remove o próximo muito perto
            else:
                i += 1

        i = 0
        while i < len(cruzamentos['abaixo_positivo']) - 1:
            if (cruzamentos['abaixo_positivo'][i + 1] - cruzamentos['abaixo_positivo'][i]) < tempo_acomodacao:
                # Se a diferença é menor que tempo_acomodacao, remover o elemento i
                cruzamentos['abaixo_positivo'].pop(i)#Remove o anterior muito perto
            else:
                i += 1
        
        janela1 = []
        janela2 = []
        janela3 = []
        janela4 = []
        janela5 = []
        # Itera para cada cruzamentos positivos
        for i in range(0,len(cruzamentos['acima_positivo'])):
            cruzamento = {}
            cruzamento['acima_positivo']  = [cruzamentos['acima_positivo'][i]]
            cruzamento['abaixo_positivo'] = [cruzamentos['abaixo_positivo'][i]]
            cruzamento['acima_negativo']  = []
            cruzamento['abaixo_negativo'] = []
            
            #print(len(cruzamento['acima_positivo']),  "acima_positivo:\t",  cruzamento['acima_positivo'])
            #print(len(cruzamento['abaixo_positivo']),  "abaixo_positivo:\t",  cruzamento['abaixo_positivo'])
            
            # Cortar Inicio
            if(i>0):
                dados_cortados = {
                    'tempo':    dados['tempo'][dados['tempo']    >= cruzamentos['abaixo_positivo'][i-1]],
                    'posicao':  dados['posicao'][dados['tempo']  >= cruzamentos['abaixo_positivo'][i-1]],
                    'potencia': dados['potencia'][dados['tempo'] >= cruzamentos['abaixo_positivo'][i-1]]
                }
            else:
                dados_cortados = dados
            
            #Cortar final
            if(i<len(cruzamentos['acima_positivo'])-1):
                dados_cortados = {
                    'tempo':    dados_cortados['tempo'][dados_cortados['tempo']    <= cruzamentos['acima_positivo'][i+1]],
                    'posicao':  dados_cortados['posicao'][dados_cortados['tempo']  <= cruzamentos['acima_positivo'][i+1]],
                    'potencia': dados_cortados['potencia'][dados_cortados['tempo'] <= cruzamentos['acima_positivo'][i+1]]
                }
                
            #print("dados_cortados:\t",dados_cortados)
            j1, j2, j3, j4, j5 = self.gera_janela_simples(dados_cortados, cruzamento, tempo_acomodacao)
            #print("j1:\t",j1)
            janela1.append(j1)
            janela2.append(j2)
            janela3.append(j3)
            janela4.append(j4)
            janela5.append(j5)
        return janela1, janela2, janela3, janela4, janela5
        

    def calculate_dBar_dRea(self, janela1, janela2, janela3, janela4, janela5):
        barra = {}
        barra['inicial'] = np.mean(janela2['posicao']) if len(janela2['posicao']) > 0 else np.nan
        barra['final']   = np.mean(janela5['posicao']) if len(janela5['posicao']) > 0 else np.nan
        # Regressão exponencial
        def exponencial(x, a, b):
            return a * np.exp(b * x)
        # Gerar dados ajustados para visualização
        popt2, pcov2 = curve_fit(exponencial, janela2['tempo'], janela2['potencia'], p0=[1, 0.01]) # p0 são os valores iniciais para a otimização
        popt5, pcov5 = curve_fit(exponencial, janela5['tempo'], janela5['potencia'], p0=[1, 0.01]) # p0 são os valores iniciais para a otimização
        x_fit2 = np.linspace(min(janela1['tempo']), max(janela3['tempo']), 100)
        y_fit2 = exponencial(x_fit2, *popt2)
        x_fit5 = np.linspace(min(janela3['tempo']), max(janela5['tempo']), 100)
        y_fit5 = exponencial(x_fit5, *popt5)
                                             #janela2['tempo'] - janela2['tempo'][0]
        popt2, pcov2 = curve_fit(exponencial, janela2['tempo']- janela2['tempo'][0], janela2['potencia'], p0=[1, 0.01]) # p0 são os valores iniciais para a otimização
        popt5, pcov5 = curve_fit(exponencial, janela5['tempo']- janela5['tempo'][0], janela5['potencia'], p0=[1, 0.01]) # p0 são os valores iniciais para a otimização
        a2, b2 = popt2
        a5, b5 = popt5
        
        
        
        # Calcaular periodo
        periodo2 = 1/b2
        periodo5 = 1/b5

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
        reatividade_PCM2  = 100000*((l/periodo2)
                                +(B1/(1+A1*periodo2))
                                +(B2/(1+A2*periodo2))
                                +(B3/(1+A3*periodo2))
                                +(B4/(1+A4*periodo2))
                                +(B5/(1+A5*periodo2))
                                +(B6/(1+A6*periodo2)))
        reatividade_PCM5  = 100000*((l/periodo5)
                                +(B1/(1+A1*periodo5))
                                +(B2/(1+A2*periodo5))
                                +(B3/(1+A3*periodo5))
                                +(B4/(1+A4*periodo5))
                                +(B5/(1+A5*periodo5))
                                +(B6/(1+A6*periodo5)))
        dRea = reatividade_PCM5-reatividade_PCM2
        # Criação do PDF
        self.indice_relatorio += 1
        
        with PdfPages(f'./tmp/{self.indice_relatorio}.pdf') as pdf:
            
            # Criação da figura
            plt.figure(figsize=(8.27, 11.69))
            
            # Adiciona a tabela na figura
            tabela_dados = [
                ["Posição Inicial da Barra de Regulação:",  f"{barra['inicial']:.2f}"],
                ["Posição Final da Barra de Regulação:",    f"{barra['final']:.2f}"],
                ["Delta da Posição da Barra de Regulação:", f"{barra['final'] - barra['inicial']:.2f}"],
                ["Reatividade Inicial Encontrada (PCM):", reatividade_PCM2],
                ["Reatividade Final Encontrada (PCM):", reatividade_PCM5],
                ["Delta de Reatividade Encontrado (PCM):", dRea],
                ["Período Inicial Encontrado (s):", f"{periodo2:.2f}"],
                ["Período Final Encontrado (s):", f"{periodo5:.2f}"],
                ["Coeficientes da exponencial ajustada 2 [a, b]:", [float(a2),float(b2)]],
                ["Coeficientes da exponencial ajustada 5 [a, b]:", [float(a5),float(b5)]],
                ["Janela de tempo 1 (s):", "[" + f"{janela1['tempo'][0]:.2f}" + " , " + f"{janela1['tempo'][-1]:.2f}" + "]"],
                ["Janela de tempo 2 (s):", "[" + f"{janela2['tempo'][0]:.2f}" + " , " + f"{janela2['tempo'][-1]:.2f}" + "]"],
                ["Janela de tempo 3 (s):", "[" + f"{janela3['tempo'][0]:.2f}" + " , " + f"{janela3['tempo'][-1]:.2f}" + "]"],
                ["Janela de tempo 4 (s):", "[" + f"{janela4['tempo'][0]:.2f}" + " , " + f"{janela4['tempo'][-1]:.2f}" + "]"],
                ["Janela de tempo 5 (s):", "[" + f"{janela5['tempo'][0]:.2f}" + " , " + f"{janela5['tempo'][-1]:.2f}" + "]"],
            ]
            
            # Define os títulos das colunas
            col_labels = ['Descrição', 'Valor']
            # Adiciona a tabela ao gráfico
            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.axis('off')  # Remove o eixo
            plt.table(cellText=tabela_dados, colLabels=col_labels, loc='center', cellLoc='left', colColours=['lightgrey', 'lightgrey'])
            
            # Ajusta o layout para a tabela
            plt.title(f'Relatório n° {self.indice_relatorio} de Inserção de Reatividade (BarraReg)', fontsize=16)
            plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)  # Ajuste para deixar espaço para a tabela
            
            # Adiciona os gráficos abaixo da tabela
            #plt.figure(figsize=(15, 10))  # Redefine o tamanho da figura para os gráficos
            plt.subplot(3, 1, 2)
            plt.plot(janela1['tempo'], janela1['posicao'], linestyle='-', color='r', label='Janela 1')
            plt.plot(janela2['tempo'], janela2['posicao'], linestyle='-', color='y', label='Janela 2')
            plt.plot(janela3['tempo'], janela3['posicao'], linestyle='-', color='g', label='Janela 3')
            plt.plot(janela4['tempo'], janela4['posicao'], linestyle='-', color='b', label='Janela 4')
            plt.plot(janela5['tempo'], janela5['posicao'], linestyle='-', color='purple', label='Janela 5')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Posição da Barra de Regulação')
            plt.title('Posição da Barra de Regulação em função do Tempo')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(x_fit2, y_fit2, linestyle='-', color='orange', label='Regressão Exponencial 2')
            plt.plot(x_fit5, y_fit5, linestyle='-', color='orange', label='Regressão Exponencial 5')
            plt.plot(janela1['tempo'], janela1['potencia'], linestyle='-', color='r', label='Janela 1')
            plt.plot(janela2['tempo'], janela2['potencia'], linestyle='-', color='y', label='Janela 2')
            plt.plot(janela3['tempo'], janela3['potencia'], linestyle='-', color='g', label='Janela 3')
            plt.plot(janela4['tempo'], janela4['potencia'], linestyle='-', color='b', label='Janela 4')
            plt.plot(janela5['tempo'], janela5['potencia'], linestyle='-', color='purple', label='Janela 5')
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
        return barra, dRea
    
    
    def gerar_grafico_calibracao(self, dBar_dRea, gpoly):
        # Converter os dados para arrays numpy
        #x = np.array([row[1] for row in dBar_dRea])  # Segunda coluna
        x = np.array([
            row[1] 
            for row in dBar_dRea 
            if row is not None and isinstance(row, (list, tuple)) and len(row) > 1
        ])
        #x = np.array([(row[0] + row[1]) / 2 for row in dBar_dRea])
        dy = np.array([row[2] for row in dBar_dRea])  # Terceira coluna
        y = np.cumsum(dy)
        
        
        
        # Regreção linear com o grau do polinômio especificado
        coeffs = np.polyfit(x, y, gpoly)
        poly = np.poly1d(coeffs)

        # Gerar valores para plotagem
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = poly(x_fit)

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        import matplotlib.backends.backend_pdf

        os.makedirs('./tmp',exist_ok=True)
        c = canvas.Canvas("./tmp/final.pdf", pagesize=A4)
        c.drawString(72, 800, "Página Inicial A4")  # Adiciona um texto simples
        c.showPage()
        c.save()

        with matplotlib.backends.backend_pdf.PdfPages("./tmp/final.pdf") as pdf_pages:
            plt.figure(figsize=(8.27, 11.69))  # Tamanho A4 em polegadas (8.27 x 11.69)
            plt.scatter(x, y, color='red', label='dP/dR')
            plt.plot(x_fit, y_fit, label=f'Ajuste Polinomial ({gpoly}º Grau)', color='blue')
            plt.xlabel('Posição de Barra')
            plt.ylabel('Reatividade')
            plt.title('Reatividade em função da posição de barra')
            plt.legend()
            plt.grid(True)
            pdf_pages.savefig()  # Salva o gráfico no PDF

def main():
    """
    --calib-reg
    --inter-calib
    --test
    """
    
    """
    REG DEL CON REA
    150   0 561 
    250 100 561 34,0
    250   0 555
    350 100 555 58,5 
    350   0 544
    433  83 544 60,8
    433   0 533
    512  79 533 64,0
    512   0 521
    591  79 521 62,4 
    591   0 510 
    681  90 510 55,3
    681   0 499
    771  90 499 38,7
    771   0 491
    904 133 491 30,4 
    904   0 486
    """
    
    calib = trigaCalib()

    # Configurar o argparse para capturar os argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Realizar calibração das barras de controle a partir de arquivos de aquisição de dados CSV')
    parser.add_argument('files', nargs='*', help='Nomes dos arquivos de aquisição de dados .CSV')
    parser.add_argument('-g', '--gpoly', type=int, help='Grau do polinômio a ser usado')
    parser.add_argument('-n', '--name', type=str, help='Nome do PDF que conterá o relatório completo')
    
    # Criar um grupo de argumentos mutuamente exclusivos
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--calib-reg', action='store_true', help='Realizar calibração da barra de regulação')
    group.add_argument('-i', '--inter-calib', action='store_true', help='Realizar intercalibração das barras de segurança e de controle')
    group.add_argument('-t', '--test', action='store_true', help='Executar em modo de teste (não precisa passar arquivos de entrada)')
    group.add_argument('-r', '--report', action='store_true', help='Executar o relatório de reatividade')


    # Parse dos argumentos
    args = parser.parse_args()

    if args.name is not None:
        pdfname = args.name
    else:
        pdfname = "report.pdf"
    
    #Se estiver no modo teste, pegar valores de exemplo
    if args.test:
        # Dados extraidos da calibração de 2023 (reatividade em centavos)
        calib.dBar_dRea = [
            [150, 150,    0],
            [150, 250, 34.0],
            [250, 350, 58.5], 
            [350, 433, 60.8],
            [433, 512, 64.0],
            [512, 591, 62.4], 
            [591, 681, 55.3],
            [681, 771, 38.7],
            [771, 904, 30.4],
            ]
    else:
        #Obter o delta de barra e reatividade a partir dos arquivos de entrada
        if args.files:
            if os.path.exists('./tmp'):
                shutil.rmtree('./tmp')
            os.makedirs('./tmp',exist_ok=True)
            calib.dBar_dRea.append([np.float64(186), np.float64(186),   np.float64(0)])
            for i, name in enumerate(args.files):
                calib.process_file_get_dBar_dRea(caminho_arquivo=name, simples=False)
            # Caso apenas gerar o relatório dos arquivos de entrada
            if args.report: 
                calib.concatenate_pdfs(pdfname)
                sys.exit()
        else:
            parser.print_help()
            sys.exit()
            
    if args.gpoly is not None:
        gpoly = args.gpoly
    else:
        gpoly = 3
    
    print("dBar_dRea: ", calib.dBar_dRea)
    calib.gerar_grafico_calibracao(calib.dBar_dRea, gpoly)
    calib.concatenate_pdfs(pdfname)

if __name__ == '__main__':
    main()

