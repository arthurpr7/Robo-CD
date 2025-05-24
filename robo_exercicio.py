import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import json
import argparse
import multiprocessing as mp
from functools import partial, lru_cache
from itertools import combinations
import time

# =====================================================================
# PARTE 1: ESTRUTURA DA SIMULAÇÃO (NÃO MODIFICAR)
# Esta parte contém a estrutura básica da simulação, incluindo o ambiente,
# o robô e a visualização. Não é recomendado modificar esta parte.
# =====================================================================

class Ambiente:
    def __init__(self, largura=800, altura=600, num_obstaculos=5, num_recursos=5):
        self.largura = largura
        self.altura = altura
        self.obstaculos = []  # Inicializa lista vazia
        self.recursos = []    # Inicializa lista vazia
        self.tempo = 0
        self.max_tempo = 1000  # Tempo máximo de simulação
        self.meta_atingida = False  # Flag para controlar se a meta foi atingida
        
        # Agora gera os elementos na ordem correta
        self.obstaculos = self.gerar_obstaculos(num_obstaculos)
        self.recursos = self.gerar_recursos(num_recursos)
        self.meta = self.gerar_meta()  # Adicionando a meta
    
    def verificar_sobreposicao(self, x, y, raio=0, largura=0, altura=0):
        """Verifica se uma posição está sobrepondo outros elementos"""
        # Verificar sobreposição com obstáculos
        for obstaculo in self.obstaculos:
            if (x + raio > obstaculo['x'] and 
                x - raio < obstaculo['x'] + obstaculo['largura'] and
                y + raio > obstaculo['y'] and 
                y - raio < obstaculo['y'] + obstaculo['altura']):
                return True
        
        # Verificar sobreposição com recursos
        for recurso in self.recursos:
            dist = np.sqrt((x - recurso['x'])**2 + (y - recurso['y'])**2)
            if dist < raio + 20:  # 20 é o raio mínimo entre elementos
                return True
        
        # Verificar sobreposição com meta
        if hasattr(self, 'meta'):
            dist = np.sqrt((x - self.meta['x'])**2 + (y - self.meta['y'])**2)
            if dist < raio + self.meta['raio'] + 20:
                return True
        
        return False
    
    def gerar_obstaculos(self, num_obstaculos):
        obstaculos = []
        max_tentativas = 100
        
        for _ in range(num_obstaculos):
            tentativas = 0
            while tentativas < max_tentativas:
                x = random.randint(50, self.largura - 50)
                y = random.randint(50, self.altura - 50)
                largura = random.randint(20, 100)
                altura = random.randint(20, 100)
                
                # Verificar sobreposição com outros obstáculos
                sobreposicao = False
                for obstaculo in obstaculos:
                    if (x + largura > obstaculo['x'] and 
                        x < obstaculo['x'] + obstaculo['largura'] and
                        y + altura > obstaculo['y'] and 
                        y < obstaculo['y'] + obstaculo['altura']):
                        sobreposicao = True
                        break
                
                if not sobreposicao:
                    obstaculos.append({
                        'x': x,
                        'y': y,
                        'largura': largura,
                        'altura': altura
                    })
                    break
                
                tentativas += 1
        
        return obstaculos
    
    def gerar_recursos(self, num_recursos):
        recursos = []
        max_tentativas = 100
        
        for _ in range(num_recursos):
            tentativas = 0
            while tentativas < max_tentativas:
                x = random.randint(20, self.largura - 20)
                y = random.randint(20, self.altura - 20)
                
                if not self.verificar_sobreposicao(x, y, raio=20):
                    recursos.append({
                        'x': x,
                        'y': y,
                        'coletado': False
                    })
                    break
                
                tentativas += 1
        
        return recursos
    
    def gerar_meta(self):
        max_tentativas = 100
        margem = 50  # Margem das bordas
        
        for _ in range(max_tentativas):
            x = random.randint(margem, self.largura - margem)
            y = random.randint(margem, self.altura - margem)
            
            if not self.verificar_sobreposicao(x, y, raio=30):
                return {
                    'x': x,
                    'y': y,
                    'raio': 30  # Raio da meta
                }
        
        # Se não encontrar uma posição segura, retorna o centro
        return {
            'x': self.largura // 2,
            'y': self.altura // 2,
            'raio': 30
        }
    
    def verificar_colisao(self, x, y, raio):
        # Verificar colisão com as bordas
        if x - raio < 0 or x + raio > self.largura or y - raio < 0 or y + raio > self.altura:
            return True
        
        # Verificar colisão com obstáculos
        for obstaculo in self.obstaculos:
            if (x + raio > obstaculo['x'] and 
                x - raio < obstaculo['x'] + obstaculo['largura'] and
                y + raio > obstaculo['y'] and 
                y - raio < obstaculo['y'] + obstaculo['altura']):
                return True
        
        return False
    
    def verificar_coleta_recursos(self, x, y, raio):
        recursos_coletados = 0
        for recurso in self.recursos:
            if not recurso['coletado']:
                distancia = np.sqrt((x - recurso['x'])**2 + (y - recurso['y'])**2)
                if distancia < raio + 10:  # 10 é o raio do recurso
                    recurso['coletado'] = True
                    recursos_coletados += 1
        return recursos_coletados
    
    def verificar_atingir_meta(self, x, y, raio):
        if not self.meta_atingida:
            distancia = np.sqrt((x - self.meta['x'])**2 + (y - self.meta['y'])**2)
            if distancia < raio + self.meta['raio']:
                self.meta_atingida = True
                return True
        return False
    
    def reset(self):
        self.tempo = 0
        for recurso in self.recursos:
            recurso['coletado'] = False
        self.meta_atingida = False
        return self.get_estado()
    
    def get_estado(self):
        return {
            'tempo': self.tempo,
            'recursos_coletados': sum(1 for r in self.recursos if r['coletado']),
            'recursos_restantes': sum(1 for r in self.recursos if not r['coletado']),
            'meta_atingida': self.meta_atingida
        }
    
    def passo(self):
        self.tempo += 1
        return self.tempo >= self.max_tempo
    
    def posicao_segura(self, raio_robo=15):
        """Encontra uma posição segura para o robô, longe dos obstáculos"""
        max_tentativas = 100
        margem = 50  # Margem das bordas
        
        for _ in range(max_tentativas):
            x = random.randint(margem, self.largura - margem)
            y = random.randint(margem, self.altura - margem)
            
            # Verificar se a posição está longe o suficiente dos obstáculos
            posicao_segura = True
            for obstaculo in self.obstaculos:
                # Calcular a distância até o obstáculo mais próximo
                dist_x = max(obstaculo['x'] - x, 0, x - (obstaculo['x'] + obstaculo['largura']))
                dist_y = max(obstaculo['y'] - y, 0, y - (obstaculo['y'] + obstaculo['altura']))
                dist = np.sqrt(dist_x**2 + dist_y**2)
                
                if dist < raio_robo + 20:  # 20 pixels de margem extra
                    posicao_segura = False
                    break
            
            if posicao_segura:
                return x, y
        
        # Se não encontrar uma posição segura, retorna o centro
        return self.largura // 2, self.altura // 2

class Robo:
    def __init__(self, x, y, raio=15):
        self.x = x
        self.y = y
        self.raio = raio
        self.angulo = 0  # em radianos
        self.velocidade = 0
        self.energia = 100
        self.recursos_coletados = 0
        self.colisoes = 0
        self.distancia_percorrida = 0
        self.tempo_parado = 0  # Novo: contador de tempo parado
        self.ultima_posicao = (x, y)  # Novo: última posição conhecida
        self.meta_atingida = False  # Novo: flag para controlar se a meta foi atingida
    
    def reset(self, x, y):
        self.x = x
        self.y = y
        self.angulo = 0
        self.velocidade = 0
        self.energia = 100
        self.recursos_coletados = 0
        self.colisoes = 0
        self.distancia_percorrida = 0
        self.tempo_parado = 0
        self.ultima_posicao = (x, y)
        self.meta_atingida = False
    
    def mover(self, aceleracao, rotacao, ambiente):
        # Atualizar ângulo
        self.angulo += rotacao
        
        # Verificar se o robô está parado
        distancia_movimento = np.sqrt((self.x - self.ultima_posicao[0])**2 + (self.y - self.ultima_posicao[1])**2)
        if distancia_movimento < 0.1:  # Se moveu menos de 0.1 unidades
            self.tempo_parado += 1
            # Forçar movimento após ficar parado por muito tempo
            if self.tempo_parado > 5:  # Após 5 passos parado
                aceleracao = max(0.2, aceleracao)  # Força aceleração mínima
                rotacao = random.uniform(-0.2, 0.2)  # Pequena rotação aleatória
        else:
            self.tempo_parado = 0
        
        # Atualizar velocidade
        self.velocidade += aceleracao
        self.velocidade = max(0.1, min(5, self.velocidade))  # Velocidade mínima de 0.1
        
        # Calcular nova posição
        novo_x = self.x + self.velocidade * np.cos(self.angulo)
        novo_y = self.y + self.velocidade * np.sin(self.angulo)
        
        # Verificar colisão
        if ambiente.verificar_colisao(novo_x, novo_y, self.raio):
            self.colisoes += 1
            self.velocidade = 0.1  # Reduz velocidade drasticamente
            self.angulo += random.uniform(np.pi/2, np.pi)  # Gira bruscamente para sair da colisão

        else:
            # Atualizar posição
            self.distancia_percorrida += np.sqrt((novo_x - self.x)**2 + (novo_y - self.y)**2)
            self.x = novo_x
            self.y = novo_y
        
        # Atualizar última posição conhecida
        self.ultima_posicao = (self.x, self.y)
        
        # Verificar coleta de recursos
        recursos_coletados = ambiente.verificar_coleta_recursos(self.x, self.y, self.raio)
        self.recursos_coletados += recursos_coletados
        
        # Verificar se atingiu a meta
        if not self.meta_atingida and ambiente.verificar_atingir_meta(self.x, self.y, self.raio):
            self.meta_atingida = True
            # Recuperar energia ao atingir a meta
            self.energia = min(100, self.energia + 50)
        
        # Consumir energia
        self.energia -= 0.1 + 0.05 * self.velocidade + 0.1 * abs(rotacao)
        self.energia = max(0, self.energia)
        
        # Recuperar energia ao coletar recursos
        if recursos_coletados > 0:
            self.energia = min(100, self.energia + 20 * recursos_coletados)
        
        return self.energia <= 0
    
    def get_sensores(self, ambiente):
        # Distância até o recurso mais próximo
        dist_recurso = float('inf')
        for recurso in ambiente.recursos:
            if not recurso['coletado']:
                dist = np.sqrt((self.x - recurso['x'])**2 + (self.y - recurso['y'])**2)
                dist_recurso = min(dist_recurso, dist)
        
        # Distância até o obstáculo mais próximo
        dist_obstaculo = float('inf')
        for obstaculo in ambiente.obstaculos:
            # Simplificação: considerar apenas a distância até o centro do obstáculo
            centro_x = obstaculo['x'] + obstaculo['largura'] / 2
            centro_y = obstaculo['y'] + obstaculo['altura'] / 2
            dist = np.sqrt((self.x - centro_x)**2 + (self.y - centro_y)**2)
            dist_obstaculo = min(dist_obstaculo, dist)
        
        # Distância até a meta
        dist_meta = np.sqrt((self.x - ambiente.meta['x'])**2 + (self.y - ambiente.meta['y'])**2)
        
        # Ângulo até o recurso mais próximo
        angulo_recurso = 0
        if dist_recurso < float('inf'):
            for recurso in ambiente.recursos:
                if not recurso['coletado']:
                    dx = recurso['x'] - self.x
                    dy = recurso['y'] - self.y
                    angulo = np.arctan2(dy, dx)
                    angulo_recurso = angulo - self.angulo
                    # Normalizar para [-pi, pi]
                    while angulo_recurso > np.pi:
                        angulo_recurso -= 2 * np.pi
                    while angulo_recurso < -np.pi:
                        angulo_recurso += 2 * np.pi
                    break
        
        # Ângulo até a meta
        dx_meta = ambiente.meta['x'] - self.x
        dy_meta = ambiente.meta['y'] - self.y
        angulo_meta = np.arctan2(dy_meta, dx_meta) - self.angulo
        # Normalizar para [-pi, pi]
        while angulo_meta > np.pi:
            angulo_meta -= 2 * np.pi
        while angulo_meta < -np.pi:
            angulo_meta += 2 * np.pi
        
        return {
            'dist_recurso': dist_recurso,
            'dist_obstaculo': dist_obstaculo,
            'dist_meta': dist_meta,
            'angulo_recurso': angulo_recurso,
            'angulo_meta': angulo_meta,
            'energia': self.energia,
            'velocidade': self.velocidade,
            'meta_atingida': self.meta_atingida
        }

class Simulador:
    def __init__(self, ambiente, robo, individuo):
        self.ambiente = ambiente
        self.robo = robo
        self.individuo = individuo
        self.frames = []
        
        # Configurar matplotlib para melhor visualização
        plt.style.use('default')
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, ambiente.largura)
        self.ax.set_ylim(0, ambiente.altura)
        self.ax.set_title("Simulador de Robô com Programação Genética", fontsize=14)
        self.ax.set_xlabel("X", fontsize=12)
        self.ax.set_ylabel("Y", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
    
    def simular(self):
        self.ambiente.reset()
        # Encontrar uma posição segura para o robô
        x_inicial, y_inicial = self.ambiente.posicao_segura(self.robo.raio)
        self.robo.reset(x_inicial, y_inicial)
        
        # Limpar a figura atual
        self.ax.clear()
        self.ax.set_xlim(0, self.ambiente.largura)
        self.ax.set_ylim(0, self.ambiente.altura)
        self.ax.set_title("Simulador de Robô com Programação Genética", fontsize=14)
        self.ax.set_xlabel("X", fontsize=12)
        self.ax.set_ylabel("Y", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Desenhar obstáculos
        for obstaculo in self.ambiente.obstaculos:
            rect = patches.Rectangle(
                (obstaculo['x'], obstaculo['y']),
                obstaculo['largura'],
                obstaculo['altura'],
                linewidth=1,
                edgecolor='black',
                facecolor='#FF9999',
                alpha=0.7
            )
            self.ax.add_patch(rect)
        
        # Desenhar recursos
        for recurso in self.ambiente.recursos:
            if not recurso['coletado']:
                circ = patches.Circle(
                    (recurso['x'], recurso['y']),
                    10,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='#99FF99',
                    alpha=0.8
                )
                self.ax.add_patch(circ)
        
        # Desenhar a meta
        meta_circ = patches.Circle(
            (self.ambiente.meta['x'], self.ambiente.meta['y']),
            self.ambiente.meta['raio'],
            linewidth=2,
            edgecolor='black',
            facecolor='#FFFF00',
            alpha=0.8
        )
        self.ax.add_patch(meta_circ)
        
        # Desenhar robô inicial
        robo_circ = patches.Circle(
            (self.robo.x, self.robo.y),
            self.robo.raio,
            linewidth=1,
            edgecolor='black',
            facecolor='#9999FF',
            alpha=0.8
        )
        self.ax.add_patch(robo_circ)
        
        # Desenhar direção inicial do robô
        direcao_x = self.robo.x + self.robo.raio * np.cos(self.robo.angulo)
        direcao_y = self.robo.y + self.robo.raio * np.sin(self.robo.angulo)
        self.ax.plot([self.robo.x, direcao_x], [self.robo.y, direcao_y], 'r-', linewidth=2)
        
        # Adicionar informações iniciais
        info_text = self.ax.text(
            10, self.ambiente.altura - 50,
            f"Tempo: {self.ambiente.tempo}\n"
            f"Recursos: {self.robo.recursos_coletados}\n"
            f"Energia: {self.robo.energia:.1f}\n"
            f"Colisões: {self.robo.colisoes}\n"
            f"Distância: {self.robo.distancia_percorrida:.1f}\n"
            f"Meta atingida: {'Sim' if self.robo.meta_atingida else 'Não'}",
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
        )
        
        # Atualizar a figura
        plt.draw()
        plt.pause(0.01)
        
        try:
            while True:
                # Obter sensores
                sensores = self.robo.get_sensores(self.ambiente)
                
                # Avaliar árvores de decisão
                aceleracao = self.individuo.avaliar(sensores, 'aceleracao')
                rotacao = self.individuo.avaliar(sensores, 'rotacao')
                
                # Limitar valores
                aceleracao = max(-1, min(1, aceleracao))
                rotacao = max(-0.5, min(0.5, rotacao))
                
                # Mover robô
                sem_energia = self.robo.mover(aceleracao, rotacao, self.ambiente)
                
                # Atualizar visualização
                self.ax.clear()
                self.ax.set_xlim(0, self.ambiente.largura)
                self.ax.set_ylim(0, self.ambiente.altura)
                self.ax.set_title("Simulador de Robô com Programação Genética", fontsize=14)
                self.ax.set_xlabel("X", fontsize=12)
                self.ax.set_ylabel("Y", fontsize=12)
                self.ax.grid(True, linestyle='--', alpha=0.7)
                
                # Redesenhar todos os elementos
                for obstaculo in self.ambiente.obstaculos:
                    rect = patches.Rectangle(
                        (obstaculo['x'], obstaculo['y']),
                        obstaculo['largura'],
                        obstaculo['altura'],
                        linewidth=1,
                        edgecolor='black',
                        facecolor='#FF9999',
                        alpha=0.7
                    )
                    self.ax.add_patch(rect)
                
                for recurso in self.ambiente.recursos:
                    if not recurso['coletado']:
                        circ = patches.Circle(
                            (recurso['x'], recurso['y']),
                            10,
                            linewidth=1,
                            edgecolor='black',
                            facecolor='#99FF99',
                            alpha=0.8
                        )
                        self.ax.add_patch(circ)
                
                meta_circ = patches.Circle(
                    (self.ambiente.meta['x'], self.ambiente.meta['y']),
                    self.ambiente.meta['raio'],
                    linewidth=2,
                    edgecolor='black',
                    facecolor='#FFFF00',
                    alpha=0.8
                )
                self.ax.add_patch(meta_circ)
                
                robo_circ = patches.Circle(
                    (self.robo.x, self.robo.y),
                    self.robo.raio,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='#9999FF',
                    alpha=0.8
                )
                self.ax.add_patch(robo_circ)
                
                direcao_x = self.robo.x + self.robo.raio * np.cos(self.robo.angulo)
                direcao_y = self.robo.y + self.robo.raio * np.sin(self.robo.angulo)
                self.ax.plot([self.robo.x, direcao_x], [self.robo.y, direcao_y], 'r-', linewidth=2)
                
                info_text = self.ax.text(
                    10, self.ambiente.altura - 50,
                    f"Tempo: {self.ambiente.tempo}\n"
                    f"Recursos: {self.robo.recursos_coletados}\n"
                    f"Energia: {self.robo.energia:.1f}\n"
                    f"Colisões: {self.robo.colisoes}\n"
                    f"Distância: {self.robo.distancia_percorrida:.1f}\n"
                    f"Meta atingida: {'Sim' if self.robo.meta_atingida else 'Não'}",
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
                )
                
                plt.draw()
                plt.pause(0.05)
                
                if sem_energia or self.ambiente.passo():
                    break
            
            plt.ioff()
            plt.show()
            
        except KeyboardInterrupt:
            plt.close('all')
        
        return self.frames
    
    def animar(self):
        # Desativar o modo interativo antes de criar a animação
        plt.ioff()
        
        # Criar a animação
        anim = animation.FuncAnimation(
            self.fig, self.atualizar_frame,
            frames=len(self.frames),
            interval=50,
            blit=True,
            repeat=True  # Permitir que a animação repita
        )
        
        # Mostrar a animação e manter a janela aberta
        plt.show(block=True)
    
    def atualizar_frame(self, frame_idx):
        return self.frames[frame_idx]

# =====================================================================
# PARTE 2: ALGORITMO GENÉTICO (PARA O VOCÊ MODIFICAR)
# Esta parte contém a implementação do algoritmo genético.
# Deve modificar os parâmetros e a lógica para melhorar o desempenho.
# =====================================================================

# ATUALIZAÇÃO: Classe IndividuoPG com operadores booleanos e comparadores adicionados

class IndividuoPG:
    def __init__(self, profundidade=3):
        self.profundidade = profundidade
        self.arvore_aceleracao = self.criar_arvore_aleatoria()
        self.arvore_rotacao = self.criar_arvore_aleatoria()
        self.fitness = 0
        self.idade = 0  # Novo: idade do indivíduo para controle de diversidade

    def criar_arvore_aleatoria(self):
        if self.profundidade == 0:
            return self.criar_folha()

        # Operadores otimizados para o problema
        operador = random.choice([
            '+', '-', '*', '/', 'max', 'min', 'abs', 'clip', 'if_positivo', 'if_negativo',
            'gt', 'lt', 'eq', 'and', 'or', 'not', 'sin', 'cos', 'tanh', 'log',
            'sigmoid', 'relu', 'softplus'  # Novos operadores
        ])

        if operador in ['+', '-', '*', '/', 'max', 'min', 'if_positivo', 'if_negativo', 'gt', 'lt', 'eq', 'and', 'or', 'clip']:
            return {
                'tipo': 'operador',
                'operador': operador,
                'esquerda': IndividuoPG(self.profundidade - 1).criar_arvore_aleatoria(),
                'direita': IndividuoPG(self.profundidade - 1).criar_arvore_aleatoria()
            }
        elif operador in ['abs', 'not', 'sin', 'cos', 'tanh', 'log', 'sigmoid', 'relu', 'softplus']:
            return {
                'tipo': 'operador',
                'operador': operador,
                'esquerda': IndividuoPG(self.profundidade - 1).criar_arvore_aleatoria(),
                'direita': None
            }

    def criar_folha(self):
        tipo = random.choice([
            'constante', 'dist_recurso', 'dist_obstaculo', 'dist_meta',
            'angulo_recurso', 'angulo_meta', 'energia', 'velocidade', 'meta_atingida'
        ])
        if tipo == 'constante':
            return {
                'tipo': 'folha',
                'valor': random.uniform(-5, 5)
            }
        else:
            return {
                'tipo': 'folha',
                'variavel': tipo
            }

    def avaliar(self, sensores, tipo='aceleracao'):
        arvore = self.arvore_aceleracao if tipo == 'aceleracao' else self.arvore_rotacao
        return self.avaliar_no(arvore, sensores)

    def avaliar_no(self, no, sensores):
        if no is None:
            return 0

        if no['tipo'] == 'folha':
            if 'valor' in no:
                return float(np.clip(no['valor'], -1e6, 1e6))
            elif 'variavel' in no:
                v = sensores[no['variavel']]
                return float(np.clip(v, -1e6, 1e6))

        esquerda = self.avaliar_no(no['esquerda'], sensores)
        direita = self.avaliar_no(no['direita'], sensores) if no['direita'] is not None else 0
        op = no['operador']

        try:
            if op == 'abs':
                return float(np.abs(esquerda))
            elif op == 'not':
                return 1.0 if esquerda <= 0 else 0.0
            elif op == 'if_positivo':
                return direita if esquerda > 0 else 0
            elif op == 'if_negativo':
                return direita if esquerda < 0 else 0
            elif op == '+':
                r = esquerda + direita
                return float(np.clip(r, -1e6, 1e6))
            elif op == '-':
                r = esquerda - direita
                return float(np.clip(r, -1e6, 1e6))
            elif op == '*':
                r = esquerda * direita
                return float(np.clip(r, -1e6, 1e6))
            elif op == '/':
                if direita == 0:
                    return 0.0
                r = esquerda / direita
                return float(np.clip(r, -1e6, 1e6))
            elif op == 'max':
                return float(np.clip(max(esquerda, direita), -1e6, 1e6))
            elif op == 'min':
                return float(np.clip(min(esquerda, direita), -1e6, 1e6))
            elif op == 'gt':
                return 1.0 if esquerda > direita else 0.0
            elif op == 'lt':
                return 1.0 if esquerda < direita else 0.0
            elif op == 'eq':
                return 1.0 if abs(esquerda - direita) < 1e-5 else 0.0
            elif op == 'and':
                return 1.0 if esquerda > 0 and direita > 0 else 0.0
            elif op == 'or':
                return 1.0 if esquerda > 0 or direita > 0 else 0.0
            elif op == 'clip':
                return float(np.clip(esquerda, -1, 1))
            elif op == 'sin':
                return float(np.clip(np.sin(np.clip(esquerda, -1e3, 1e3)), -1, 1))
            elif op == 'cos':
                return float(np.clip(np.cos(np.clip(esquerda, -1e3, 1e3)), -1, 1))
            elif op == 'tanh':
                return float(np.tanh(np.clip(esquerda, -20, 20)))
            elif op == 'log':
                v = np.abs(esquerda)
                if v > 0:
                    return float(np.clip(np.log(v), -1e6, 1e6))
                else:
                    return 0.0
            elif op == 'sigmoid':
                x = np.clip(esquerda, -100, 100)
                return float(1 / (1 + np.exp(-x)))
            elif op == 'relu':
                return float(np.clip(max(0, esquerda), 0, 1e6))
            elif op == 'softplus':
                x = np.clip(esquerda, -100, 100)
                if x > 100:
                    return float(x)
                elif x < -100:
                    return 0.0
                else:
                    return float(np.clip(np.log1p(np.exp(x)), 0, 1e6))
        except Exception as e:
            # Opcional: print(f"Erro matemático: {op}({esquerda}, {direita}) -> {e}")
            return 0.0

    def mutacao(self, probabilidade=0.1):
        # Mutação adaptativa baseada na idade
        prob_mutacao = probabilidade * (1 + self.idade * 0.1)  # Aumenta com a idade
        self.mutacao_no(self.arvore_aceleracao, prob_mutacao)
        self.mutacao_no(self.arvore_rotacao, prob_mutacao)
        self.idade += 1

    def mutacao_no(self, no, probabilidade):
        if random.random() < probabilidade:
            if no['tipo'] == 'folha':
                if 'valor' in no:
                    # Mutação gaussiana para constantes
                    no['valor'] += random.gauss(0, 0.5)
                    no['valor'] = np.clip(no['valor'], -5, 5)
                elif 'variavel' in no:
                    no['variavel'] = random.choice([
                        'dist_recurso', 'dist_obstaculo', 'dist_meta',
                        'angulo_recurso', 'angulo_meta', 'energia', 'velocidade', 'meta_atingida'
                    ])
            else:
                no['operador'] = random.choice([
                    '+', '-', '*', '/', 'max', 'min', 'abs', 'clip',
                    'if_positivo', 'if_negativo', 'gt', 'lt', 'eq', 'and', 'or', 'not',
                    'sin', 'cos', 'tanh', 'log', 'sigmoid', 'relu', 'softplus'
                ])

        if no['tipo'] == 'operador':
            self.mutacao_no(no['esquerda'], probabilidade)
            if no['direita'] is not None:
                self.mutacao_no(no['direita'], probabilidade)

    def crossover(self, outro):
        novo = IndividuoPG(self.profundidade)
        novo.arvore_aceleracao = self.crossover_no(self.arvore_aceleracao, outro.arvore_aceleracao)
        novo.arvore_rotacao = self.crossover_no(self.arvore_rotacao, outro.arvore_rotacao)
        novo.idade = 0  # Reset da idade para novos indivíduos
        return novo

    def crossover_no(self, no1, no2):
        # Crossover com preservação de características boas
        if random.random() < 0.5:
            return json.loads(json.dumps(no1))
        else:
            return json.loads(json.dumps(no2))

    def salvar(self, arquivo):
        with open(arquivo, 'w') as f:
            json.dump({
                'arvore_aceleracao': self.arvore_aceleracao,
                'arvore_rotacao': self.arvore_rotacao,
                'idade': self.idade
            }, f)

    @classmethod
    def carregar(cls, arquivo):
        with open(arquivo, 'r') as f:
            dados = json.load(f)
            individuo = cls()
            individuo.arvore_aceleracao = dados['arvore_aceleracao']
            individuo.arvore_rotacao = dados['arvore_rotacao']
            individuo.idade = dados.get('idade', 0)
            return individuo


class ProgramacaoGenetica:
    def __init__(self, tamanho_populacao=200, profundidade=5):
        self.tamanho_populacao = tamanho_populacao
        self.profundidade = profundidade
        self.populacao = [IndividuoPG(profundidade) for _ in range(tamanho_populacao)]
        self.melhor_individuo = None
        self.melhor_fitness = float('-inf')
        self.historico_fitness = []
        self.historico_diversidade = []
        
        # Parâmetros de evolução otimizados
        self.taxa_elitismo = 0.15
        self.taxa_mutacao = 0.3
        self.tamanho_torneio = 7
        self.pressao_seletiva = 0.9
        self.taxa_crossover = 0.9
        
        # Número de processos para paralelização
        self.num_processos = max(1, mp.cpu_count() - 1)
        
        # Cache para ambiente e robô
        self._ambiente_cache = None
        self._robo_cache = None
        
        # Constantes de fitness pré-calculadas
        self._fitness_constants = {
            'penalidade_sem_meta': 2000,
            'penalidade_sem_energia': 800,
            'penalidade_sem_recursos': 500,
            'penalidade_poucos_recursos': 1000,
            'bonus_coleta': 1500,
            'bonus_meta': 800,
            'bonus_recurso': 500,
            'bonus_energia': 15,
            'bonus_todos_recursos': 3000,
            'bonus_meta_com_recursos': 2000,
            'bonus_completude': 5000,
            'penalidade_colisao': 300,
            'penalidade_sem_progresso': 15
        }
    
    @property
    def ambiente(self):
        """Lazy initialization of environment"""
        if self._ambiente_cache is None:
            self._ambiente_cache = Ambiente()
        return self._ambiente_cache
    
    @property
    def robo(self):
        """Lazy initialization of robot"""
        if self._robo_cache is None:
            self._robo_cache = Robo(self.ambiente.largura // 2, self.ambiente.altura // 2)
        return self._robo_cache
    
    @lru_cache(maxsize=1024)
    def distancia_arvores(self, arvore1_str, arvore2_str):
        """Versão otimizada do cálculo de distância entre árvores usando cache"""
        arvore1 = json.loads(arvore1_str)
        arvore2 = json.loads(arvore2_str)
        
        if arvore1['tipo'] != arvore2['tipo']:
            return 1.0
        
        if arvore1['tipo'] == 'folha':
            if 'valor' in arvore1 and 'valor' in arvore2:
                return abs(arvore1['valor'] - arvore2['valor']) / 10
            elif 'variavel' in arvore1 and 'variavel' in arvore2:
                return 0.0 if arvore1['variavel'] == arvore2['variavel'] else 1.0
            return 1.0
        
        dist_esq = self.distancia_arvores(
            json.dumps(arvore1['esquerda']),
            json.dumps(arvore2['esquerda'])
        )
        
        if arvore1['direita'] is None and arvore2['direita'] is None:
            dist_dir = 0.0
        elif arvore1['direita'] is None or arvore2['direita'] is None:
            dist_dir = 1.0
        else:
            dist_dir = self.distancia_arvores(
                json.dumps(arvore1['direita']),
                json.dumps(arvore2['direita'])
            )
        
        return (dist_esq + dist_dir) / 2 + (0.0 if arvore1['operador'] == arvore2['operador'] else 0.5)
    
    def calcular_diversidade_paralela(self, chunk):
        """Calcula diversidade para um chunk da população em paralelo"""
        diversidade = 0
        for i, j in combinations(chunk, 2):
            dist_acel = self.distancia_arvores(
                json.dumps(i.arvore_aceleracao),
                json.dumps(j.arvore_aceleracao)
            )
            dist_rot = self.distancia_arvores(
                json.dumps(i.arvore_rotacao),
                json.dumps(j.arvore_rotacao)
            )
            diversidade += (dist_acel + dist_rot) / 2
        return diversidade
    
    def calcular_diversidade(self):
        """Calcula a diversidade da população usando paralelização"""
        # Dividir população em chunks para processamento paralelo
        chunk_size = max(1, len(self.populacao) // self.num_processos)
        chunks = [self.populacao[i:i + chunk_size] for i in range(0, len(self.populacao), chunk_size)]
        
        with mp.Pool(processes=self.num_processos) as pool:
            diversidades = pool.map(self.calcular_diversidade_paralela, chunks)
        
        total_diversidade = sum(diversidades)
        total_comparacoes = len(self.populacao) * (len(self.populacao) - 1) / 2
        return total_diversidade / total_comparacoes
    
    def avaliar_individuo(self, individuo, num_tentativas=8):
        """Versão otimizada da avaliação de indivíduo"""
        fitness_total = 0
        recursos_totais = len(self.ambiente.recursos)
        fitness_array = np.zeros(num_tentativas)
        penalidade_parede = 1000  # penalidade extra para colisão com parede
        bonus_coleta_incremental = 2000  # bônus por cada coleta nova
        bonus_meta_final = 8000  # bônus maior para atingir meta após coletar todos recursos
        for tentativa in range(num_tentativas):
            self.ambiente.reset()
            x_inicial, y_inicial = self.ambiente.posicao_segura(self.robo.raio)
            self.robo.reset(x_inicial, y_inicial)
            ultima_distancia_meta = float('inf')
            ultima_distancia_recurso = float('inf')
            tempo_sem_progresso = 0
            recursos_coletados_anterior = 0
            recursos_coletados_set = set()
            meta_atingida_valida = False
            penalidade_colisao_parede = 0
            while True:
                sensores = self.robo.get_sensores(self.ambiente)
                aceleracao = np.clip(individuo.avaliar(sensores, 'aceleracao'), -1, 1)
                rotacao = np.clip(individuo.avaliar(sensores, 'rotacao'), -0.5, 0.5)
                distancia_meta_atual = sensores['dist_meta']
                distancia_recurso_atual = sensores['dist_recurso']
                if np.all([distancia_meta_atual >= ultima_distancia_meta, distancia_recurso_atual >= ultima_distancia_recurso]):
                    tempo_sem_progresso += 1
                else:
                    tempo_sem_progresso = 0
                ultima_distancia_meta = distancia_meta_atual
                ultima_distancia_recurso = distancia_recurso_atual
                # Detectar colisão com parede
                colidiu_parede = False
                novo_x = self.robo.x + self.robo.velocidade * np.cos(self.robo.angulo)
                novo_y = self.robo.y + self.robo.velocidade * np.sin(self.robo.angulo)
                if (novo_x - self.robo.raio < 0 or novo_x + self.robo.raio > self.ambiente.largura or
                    novo_y - self.robo.raio < 0 or novo_y + self.robo.raio > self.ambiente.altura):
                    colidiu_parede = True
                sem_energia = self.robo.mover(aceleracao, rotacao, self.ambiente)
                # Recompensa incremental por cada coleta nova
                if self.robo.recursos_coletados > recursos_coletados_anterior:
                    fitness_incremento = (self.robo.recursos_coletados - recursos_coletados_anterior) * bonus_coleta_incremental
                else:
                    fitness_incremento = 0
                recursos_coletados_anterior = self.robo.recursos_coletados
                # Penalidade extra para colisão com parede
                if colidiu_parede:
                    penalidade_colisao_parede += penalidade_parede
                # Só considerar meta atingida se todos recursos foram coletados
                if (not meta_atingida_valida and self.robo.meta_atingida and self.robo.recursos_coletados == recursos_totais):
                    meta_atingida_valida = True
                # Cálculo de fitness
                fitness_tentativa = (
                    self.robo.recursos_coletados * 2000 +
                    fitness_incremento +
                    (bonus_meta_final if meta_atingida_valida else 0) +
                    (self._fitness_constants['bonus_meta'] if distancia_meta_atual < ultima_distancia_meta else 0) +
                    (self._fitness_constants['bonus_recurso'] if distancia_recurso_atual < ultima_distancia_recurso else 0) +
                    (self.robo.energia * self._fitness_constants['bonus_energia']) +
                    (self._fitness_constants['bonus_todos_recursos'] if self.robo.recursos_coletados == recursos_totais else 0) +
                    (self._fitness_constants['bonus_meta_com_recursos'] if meta_atingida_valida and self.robo.recursos_coletados > 0 else 0) +
                    (self._fitness_constants['bonus_completude'] if self.robo.recursos_coletados == recursos_totais and meta_atingida_valida else 0) -
                    (self.robo.colisoes * self._fitness_constants['penalidade_colisao']) -
                    (tempo_sem_progresso * self._fitness_constants['penalidade_sem_progresso']) -
                    (self._fitness_constants['penalidade_sem_meta'] if not meta_atingida_valida else 0) -
                    (self._fitness_constants['penalidade_sem_energia'] if self.robo.energia <= 0 else 0) -
                    (self._fitness_constants['penalidade_sem_recursos'] if self.robo.recursos_coletados == 0 else 0) -
                    (self._fitness_constants['penalidade_poucos_recursos'] if self.robo.recursos_coletados < recursos_totais/2 else 0) -
                    penalidade_colisao_parede
                )
                fitness_tentativa = float(np.nan_to_num(fitness_tentativa, nan=0.0, posinf=1e6, neginf=-1e6))
                fitness_array[tentativa] = max(0, fitness_tentativa)
                if sem_energia or self.ambiente.passo() or tempo_sem_progresso > 50 or meta_atingida_valida:
                    break
        return np.mean(fitness_array)
    
    def avaliar_populacao(self):
        """Avalia a população em paralelo com melhor gerenciamento de recursos"""
        # Usar um pool de processos com melhor gerenciamento de memória
        with mp.Pool(processes=self.num_processos, maxtasksperchild=10) as pool:
            # Avaliar indivíduos em paralelo com chunks para melhor performance
            chunk_size = max(1, len(self.populacao) // self.num_processos)
            fitness_values = []
            
            for chunk in np.array_split(self.populacao, len(self.populacao) // chunk_size + 1):
                chunk_fitness = pool.map(self.avaliar_individuo, chunk)
                fitness_values.extend(chunk_fitness)
        
        # Atualizar fitness dos indivíduos de forma vetorizada
        for individuo, fitness in zip(self.populacao, fitness_values):
            individuo.fitness = fitness
            if fitness > self.melhor_fitness:
                self.melhor_fitness = fitness
                self.melhor_individuo = individuo
    
    def selecionar(self):
        # Seleção por torneio com pressão seletiva
        selecionados = []
        
        # Preservar elite
        elite = sorted(self.populacao, key=lambda x: x.fitness, reverse=True)
        n_elite = int(self.tamanho_populacao * self.taxa_elitismo)
        selecionados.extend(elite[:n_elite])
        
        # Seleção por torneio para o resto da população
        while len(selecionados) < self.tamanho_populacao:
            # Selecionar candidatos para o torneio
            candidatos = random.sample(self.populacao, self.tamanho_torneio)
            
            # Ordenar candidatos por fitness
            candidatos.sort(key=lambda x: x.fitness, reverse=True)
            
            # Selecionar vencedor com base na pressão seletiva
            if random.random() < self.pressao_seletiva:
                vencedor = candidatos[0]  # Melhor candidato
            else:
                vencedor = random.choice(candidatos[1:])  # Outro candidato aleatório
            
            selecionados.append(vencedor)
        
        return selecionados
    
    def evoluir(self, n_geracoes=50):
        for geracao in range(n_geracoes):
            random.seed()
            np.random.seed()
            
            print(f"Geração {geracao + 1}/{n_geracoes}")
            
            self.avaliar_populacao()
            diversidade = self.calcular_diversidade()
            
            self.historico_fitness.append(self.melhor_fitness)
            self.historico_diversidade.append(diversidade)
            
            print(f"Melhor fitness: {self.melhor_fitness:.2f}")
            print(f"Diversidade: {diversidade:.2f}")
            
            # Ajuste dinâmico da taxa de mutação baseado na diversidade
            if diversidade < 0.3:  # Se a diversidade estiver muito baixa
                self.taxa_mutacao = min(0.4, self.taxa_mutacao * 1.1)  # Aumenta mutação
            elif diversidade > 0.7:  # Se a diversidade estiver muito alta
                self.taxa_mutacao = max(0.1, self.taxa_mutacao * 0.9)  # Diminui mutação
            
            selecionados = self.selecionar()
            nova_populacao = []
            
            # Preservar elite
            elite = sorted(selecionados, key=lambda x: x.fitness, reverse=True)
            n_elite = int(self.tamanho_populacao * self.taxa_elitismo)
            nova_populacao.extend(elite[:n_elite])
            
            # Gerar resto da população
            while len(nova_populacao) < self.tamanho_populacao:
                if random.random() < self.taxa_crossover:
                    pai1, pai2 = random.sample(selecionados, 2)
                    filho = pai1.crossover(pai2)
                else:
                    filho = IndividuoPG(self.profundidade)
                
                if filho.fitness < self.melhor_fitness * 0.5:
                    filho.mutacao(probabilidade=self.taxa_mutacao * 1.5)  # Mais mutação para indivíduos ruins
                else:
                    filho.mutacao(probabilidade=self.taxa_mutacao)
                
                nova_populacao.append(filho)
            
            self.populacao = nova_populacao
        
        return self.melhor_individuo, self.historico_fitness, self.historico_diversidade

# =====================================================================
# PARTE 3: EXECUÇÃO DO PROGRAMA
# =====================================================================

if __name__ == "__main__":
    # Parser de argumentos
    parser = argparse.ArgumentParser(description="Simulação de robô com programação genética")
    parser.add_argument('--populacao', type=int, default=100, help='Tamanho da população (padrão: 10)')
    parser.add_argument('--profundidade', type=int, default=4, help='Profundidade da árvore (padrão: 4)')
    parser.add_argument('--geracoes', type=int, default=10, help='Número de gerações (padrão: 10)')
    args = parser.parse_args()

    # Usar argumentos ou valores default
    TAMANHO_POPULACAO = args.populacao
    PROFUNDIDADE = args.profundidade
    NUM_GERACOES = args.geracoes

    print("Iniciando simulação de robô com programação genética...")
    print(f"Configurações: População={TAMANHO_POPULACAO}, Profundidade={PROFUNDIDADE}, Gerações={NUM_GERACOES}")

    # Criar ambiente e robô
    ambiente = Ambiente()
    robo = Robo(ambiente.largura // 2, ambiente.altura // 2)

    # Criar e executar a programação genética
    pg = ProgramacaoGenetica(tamanho_populacao=TAMANHO_POPULACAO, profundidade=PROFUNDIDADE)
    melhor_individuo, historico_fitness, historico_diversidade = pg.evoluir(n_geracoes=NUM_GERACOES)

    print("\nTreinamento concluído!")
    print(f"Melhor fitness alcançado: {pg.melhor_fitness:.2f}")

    # Salvar o melhor indivíduo
    print("Salvando o melhor indivíduo...")
    melhor_individuo.salvar('melhor_robo.json')

    # Plotar gráficos de evolução
    plt.figure(figsize=(15, 5))

    # Gráfico de fitness
    plt.subplot(1, 2, 1)
    plt.plot(historico_fitness, 'b-', label='Melhor Fitness')
    plt.title('Evolução do Fitness', fontsize=12)
    plt.xlabel('Geração', fontsize=10)
    plt.ylabel('Fitness', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Gráfico de diversidade
    plt.subplot(1, 2, 2)
    plt.plot(historico_diversidade, 'r-', label='Diversidade')
    plt.title('Evolução da Diversidade', fontsize=12)
    plt.xlabel('Geração', fontsize=10)
    plt.ylabel('Diversidade', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('evolucao_fitness_robo.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Simular o melhor indivíduo
    print("\nSimulando o melhor indivíduo...")

    # Resetar ambiente e robô
    ambiente.reset()
    x_inicial, y_inicial = ambiente.posicao_segura(robo.raio)
    robo.reset(x_inicial, y_inicial)

    # Criar e executar simulação
    simulador = Simulador(ambiente, robo, melhor_individuo)
    simulador.simular()
