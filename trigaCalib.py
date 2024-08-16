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

def concatenate_pdfs(output_filename):
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

def calculate_dBar_dRea(caminho_arquivo,indice_relatorio,janela=10, valor_cruzamento_positivo=3, valor_cruzamento_negativo=-3, tempo_acomodacao=60):
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

    # Aplicar filtro de média móvel na posição da barra e derivar
    filtro = np.ones(janela) / janela
    posicao_filtrada = np.convolve(dados['posicao'], filtro, mode='same')
    derivada_posicao = np.diff(posicao_filtrada) / np.diff(dados['tempo'])
    derivada_posicao = np.insert(derivada_posicao, 0, np.nan) # Para que a derivada tenha o mesmo comprimento que os dados originais, podemos adicionar um NaN no início

    # Deleta a quantidade de janela da primeira e ultimas posições
    tempo_cortado = dados['tempo'][janela:-janela]
    derivada_posicao_cortada = derivada_posicao[janela:-janela]





    #plt.plot(tempo_cortado, derivada_posicao_cortada, label='Derivada', color='blue')
    #plt.xlabel('tempo')
    #plt.ylabel('Derivada')
    #plt.title('Derivada da posição da barra')
    #plt.legend()
    #plt.grid(True)
    #plt.show()

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
    
    print(len(cruzamentos_acima_positivo), "\t", cruzamentos_acima_positivo)
    print()
    print(len(cruzamentos_abaixo_positivo), "\t", cruzamentos_abaixo_positivo)
    print()
    print(len(cruzamentos_abaixo_negativo), "\t", cruzamentos_abaixo_negativo)
    print()
    print(len(cruzamentos_acima_negativo), "\t", cruzamentos_acima_negativo)
    print()

    for i in range(0,len(cruzamentos_acima_positivo)):
        # Definir as janelas de tempo
        janela1 = {
            'tempo': dados['tempo'][dados['tempo'] <= cruzamentos_acima_positivo[i]],
            'posicao': dados['posicao'][dados['tempo'] <= cruzamentos_acima_positivo[i]],
            'potencia': dados['potencia'][dados['tempo'] <= cruzamentos_acima_positivo[i]]
        }

        janela2 = {
            'tempo': dados['tempo'][(dados['tempo'] > cruzamentos_acima_positivo[i]) & (dados['tempo'] <= cruzamentos_abaixo_positivo[i])],
            'posicao': dados['posicao'][(dados['tempo'] > cruzamentos_acima_positivo[i]) & (dados['tempo'] <= cruzamentos_abaixo_positivo[i])],
            'potencia': dados['potencia'][(dados['tempo'] > cruzamentos_acima_positivo[i]) & (dados['tempo'] <= cruzamentos_abaixo_positivo[i])]
        }

        tempo_apos_cruzamento = cruzamentos_abaixo_positivo[i] + tempo_acomodacao
        indice_mais_proximo = np.abs(dados['tempo'] - tempo_apos_cruzamento).argmin()
        janela3 = {
            'tempo': dados['tempo'][(dados['tempo'] > cruzamentos_abaixo_positivo[i]) & (dados['tempo'] <= dados['tempo'][indice_mais_proximo])],
            'posicao': dados['posicao'][(dados['tempo'] > cruzamentos_abaixo_positivo[i]) & (dados['tempo'] <= dados['tempo'][indice_mais_proximo])],
            'potencia': dados['potencia'][(dados['tempo'] > cruzamentos_abaixo_positivo[i]) & (dados['tempo'] <= dados['tempo'][indice_mais_proximo])]
        }

        #if len(cruzamentos_acima_positivo) >= 2 and len(cruzamentos_abaixo_negativo) >= 1:
        #    if cruzamentos_acima_positivo[i+1] > cruzamentos_abaixo_negativo[0]:
        #        final_janela4 = cruzamentos_abaixo_negativo[0]
        #    else:
        #        final_janela4 = cruzamentos_acima_positivo[i+1]
        #else:
        #    if len(cruzamentos_acima_positivo) >= 2:
        #        final_janela4 = cruzamentos_acima_positivo[i+1]
        #    elif len(cruzamentos_abaixo_negativo) >= 1:
        #        final_janela4 = cruzamentos_abaixo_negativo[0]
        #    else:
        #        final_janela4 = dados['tempo'][-1]
        
        if len(cruzamentos_acima_positivo) >= 2 and i != (len(cruzamentos_acima_positivo) - 1):
                final_janela4 = cruzamentos_acima_positivo[i+1]
        else:
            final_janela4 = dados['tempo'][-1]

        janela4 = {
            'tempo': dados['tempo'][(dados['tempo'] > dados['tempo'][indice_mais_proximo]) & (dados['tempo'] <= final_janela4)],
            'posicao': dados['posicao'][(dados['tempo'] > dados['tempo'][indice_mais_proximo]) & (dados['tempo'] <= final_janela4)],
            'potencia': dados['potencia'][(dados['tempo'] > dados['tempo'][indice_mais_proximo]) & (dados['tempo'] <= final_janela4)]
        }
        
        posicao_inicial_barra = np.mean(janela1['posicao']) if len(janela1['posicao']) > 0 else np.nan
        posicao_final_barra = np.mean(janela4['posicao']) if len(janela4['posicao']) > 0 else np.nan

        #Deslocar tempo 0 para inicio da janela4
        janela1['tempo'] -= dados['tempo'][indice_mais_proximo]
        janela2['tempo'] -= dados['tempo'][indice_mais_proximo]
        janela3['tempo'] -= dados['tempo'][indice_mais_proximo]
        janela4['tempo'] -= dados['tempo'][indice_mais_proximo]
        
        # Regressão exponencial
        def exponencial(x, a, b):
            return a * np.exp(b * x)
        popt, pcov = curve_fit(exponencial, janela4['tempo'], janela4['potencia'], p0=[1, 0.01]) # p0 são os valores iniciais para a otimização
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
        x_fit = np.linspace(min(janela2['tempo']), max(janela4['tempo']), 100)
        y_fit = exponencial(x_fit, *popt)

        # Criação do PDF
        with PdfPages(f'./tmp/{i}.pdf') as pdf:
            
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
            plt.title(f'Relatório n° {indice_relatorio} de Inserção de Reatividade (BarraReg)', fontsize=16)
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
            plt.plot(x_fit, y_fit, linestyle='-', color='purple', label='Regressão Exponencial')
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

#def calibBarReg():
#    calculate_dBar_dRea('aqui3-100cut.csv','relatorio.pdf')

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
    
    dBar_dRea = [(np.float64(186), np.float64(186),   np.float64(0))]
    #dBar_dRea = []
    #Se estiver no modo teste, pegar valores de exemplo
    if args.test:
        # Dados extraidos da calibração de 2023 (reatividade em centavos)
        dBar_dRea = [
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
    elif args.report:
        #Apenas gerar o relatório dos arquivos de entrada
        if args.files:
            if os.path.exists('./tmp'):
                shutil.rmtree('./tmp')
            os.makedirs('./tmp',exist_ok=True)
            for i, name in enumerate(args.files):
                calculate_dBar_dRea(name,1)
            concatenate_pdfs(pdfname)
            sys.exit()
    else:
        #Obter o delta de barra e reatividade a partir dos arquivos de entrada
        if args.files:
            if os.path.exists('./tmp'):
                shutil.rmtree('./tmp')
            os.makedirs('./tmp',exist_ok=True)
            for i, name in enumerate(args.files):
                dBar_dRea.append(calculate_dBar_dRea(name, i+1))
        else:
            parser.print_help()
            sys.exit()
        
    # Converter os dados para arrays numpy
    #print(dBar_dRea)
    x = np.array([row[1] for row in dBar_dRea])  # Segunda coluna
    #x = np.array([(row[0] + row[1]) / 2 for row in dBar_dRea])
    y = np.array([row[2] for row in dBar_dRea])  # Terceira coluna
    y = np.cumsum(y)
    
    if args.gpoly is not None:
        gpoly = args.gpoly
    else:
        gpoly = 3
    
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

    concatenate_pdfs(pdfname)

if __name__ == '__main__':
    main()

