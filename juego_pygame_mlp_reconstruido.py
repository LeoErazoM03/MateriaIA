import csv
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pygame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# JUEGO DE BALA Y SALTO CON MLP
# Reconstrucción funcional basada en la descripción de la página
# https://ealcaraz85.github.io/IA.io/
# ============================================================


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    tiempo_impacto: float
    salto: int


class JuegoMLP:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.BASE_W = 1080
        self.BASE_H = 720
        self.screen = pygame.display.set_mode((self.BASE_W, self.BASE_H))
        pygame.display.set_caption("Juego Bala y Salto con MLP")

        self.clock = pygame.time.Clock()
        self.running = True
        self.fullscreen = False

        # Colores
        self.CIELO = (135, 206, 235)
        self.SUELO = (95, 184, 92)
        self.TEXTO = (20, 20, 20)
        self.BLANCO = (255, 255, 255)
        self.ROJO = (220, 70, 70)
        self.AMARILLO = (245, 210, 70)
        self.AZUL = (50, 100, 220)
        self.VERDE = (50, 170, 90)
        self.GRIS = (70, 70, 70)
        self.MORADO = (135, 90, 190)
        self.NEGRO = (0, 0, 0)

        # Fuentes
        self.font_small = pygame.font.SysFont("arial", 22)
        self.font_medium = pygame.font.SysFont("arial", 30)
        self.font_large = pygame.font.SysFont("arial", 44, bold=True)

        # Estado del juego
        self.estado = "menu"   # menu / manual / auto / game_over
        self.modo = None        # manual / auto
        self.score = 0
        self.best_score = 0
        self.mensaje = ""
        self.mensaje_timer = 0

        # Jugador
        self.player_w = 70
        self.player_h = 90
        self.ground_y = self.BASE_H - 120
        self.player_x = 140
        self.player_y = self.ground_y - self.player_h
        self.player_vel_y = 0.0
        self.saltando = False
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0

        # Bala
        self.bala_w = 34
        self.bala_h = 18
        self.bala_x = self.BASE_W + 120
        self.bala_y = self.ground_y - self.bala_h - 12
        self.bala_vel = -8.0
        self.bala_activa = False
        self.frames_para_siguiente_bala = 0

        # Fondo visual
        self.fondo_offset = 0
        self.fondo_speed = 3

        # Dataset y modelo
        self.datos_modelo = []
        self.nuevas_muestras = []
        self.modelo = None
        self.scaler = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.last_proba_salto = 0.0
        self.min_samples = 80

        # Archivo principal de datos
        self.csv_filename = "datos_mlp.csv"

        # Para registrar el salto por frame
        self.salto_solicitado_este_frame = 0

        self.reset_bala()
        self.cargar_csv_automatico()

    # --------------------------------------------------------
    # Utilidades de estado
    # --------------------------------------------------------
    def set_message(self, texto, segundos=2.5):
        self.mensaje = texto
        self.mensaje_timer = pygame.time.get_ticks() + int(segundos * 1000)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.BASE_W, self.BASE_H))

    def reset_modelo(self):
        self.modelo = None
        self.scaler = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.last_proba_salto = 0.0

    def reset_juego(self):
        self.score = 0
        self.player_y = self.ground_y - self.player_h
        self.player_vel_y = 0.0
        self.saltando = False
        self.salto_solicitado_este_frame = 0
        self.reset_bala()

    def reset_bala(self):
        self.bala_x = self.BASE_W + random.randint(30, 220)
        self.bala_y = self.ground_y - self.bala_h - 12
        self.bala_vel = random.uniform(-12.0, -6.0)
        self.bala_activa = True
        self.frames_para_siguiente_bala = random.randint(35, 90)

    def iniciar_manual(self):
        self.estado = "manual"
        self.modo = "manual"
        self.reset_juego()
        self.set_message(f"Modo manual: dataset actual = {len(self.datos_modelo)} muestras")

    def iniciar_auto(self):
        self.estado = "auto"
        self.modo = "auto"
        self.reset_juego()
        if self.modelo is None:
            self.set_message("Modo auto sin modelo: no va a saltar")
        else:
            self.set_message("Modo auto activado")

    def volver_menu(self):
        self.estado = "menu"
        self.modo = None
        self.reset_juego()

    # --------------------------------------------------------
    # Lógica de juego
    # --------------------------------------------------------
    def iniciar_salto(self):
        if not self.saltando:
            self.saltando = True
            self.player_vel_y = -self.salto_vel_inicial
            self.salto_solicitado_este_frame = 1

    def manejar_salto(self):
        if self.saltando:
            self.player_vel_y += self.gravedad
            self.player_y += self.player_vel_y

            piso = self.ground_y - self.player_h
            if self.player_y >= piso:
                self.player_y = piso
                self.player_vel_y = 0.0
                self.saltando = False

    def actualizar_bala(self):
        if self.bala_activa:
            self.bala_x += self.bala_vel
            if self.bala_x + self.bala_w < 0:
                self.score += 1
                self.best_score = max(self.best_score, self.score)
                self.reset_bala()

    def get_player_rect(self):
        return pygame.Rect(int(self.player_x), int(self.player_y), self.player_w, self.player_h)

    def get_bala_rect(self):
        return pygame.Rect(int(self.bala_x), int(self.bala_y), self.bala_w, self.bala_h)

    def checar_colision(self):
        if self.get_player_rect().colliderect(self.get_bala_rect()):
            self.estado = "game_over"
            self.best_score = max(self.best_score, self.score)
            self.set_message("Game over")

    # --------------------------------------------------------
    # Dataset manual
    # --------------------------------------------------------
    def distancia_jugador_bala(self):
        return float(self.bala_x - (self.player_x + self.player_w))

    def registrar_decision_manual(self):
        if not self.bala_activa:
            return

        distancia = float(self.distancia_jugador_bala())

        # Solo registrar cuando la bala ya es relevante.
        # Esto evita miles de frames inútiles donde la respuesta correcta
        # casi siempre es "no saltar".
        if distancia < -20 or distancia > 220:
            return

        velocidad_abs = max(abs(float(self.bala_vel)), 0.001)
        tiempo_impacto = distancia / velocidad_abs

        muestra = Sample(
            velocidad_bala=float(self.bala_vel),
            distancia=distancia,
            tiempo_impacto=tiempo_impacto,
            salto=int(self.salto_solicitado_este_frame),
        )
        self.datos_modelo.append(muestra)
        self.nuevas_muestras.append(muestra)

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------
    def entrenar_modelo(self):
        if len(self.datos_modelo) < self.min_samples:
            self.set_message(f"Muy pocas muestras: {len(self.datos_modelo)}/{self.min_samples}")
            return

        X = np.array([[s.velocidad_bala, s.distancia, s.tiempo_impacto] for s in self.datos_modelo], dtype=np.float32)
        y = np.array([s.salto for s in self.datos_modelo], dtype=np.int32)

        clases = np.unique(y)
        if len(clases) < 2:
            self.set_message("Necesitas ambas clases: saltar y no saltar")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        rng = np.random.RandomState(42)

        if len(idx_0) > len(idx_1):
            extra = rng.choice(idx_1, size=len(idx_0) - len(idx_1), replace=True)
            idx_balanceados = np.concatenate([idx_0, idx_1, extra])
        elif len(idx_1) > len(idx_0):
            extra = rng.choice(idx_0, size=len(idx_1) - len(idx_0), replace=True)
            idx_balanceados = np.concatenate([idx_0, idx_1, extra])
        else:
            idx_balanceados = np.concatenate([idx_0, idx_1])

        rng.shuffle(idx_balanceados)
        X_train_bal = X_train[idx_balanceados]
        y_train_bal = y_train[idx_balanceados]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)

        modelo = MLPClassifier(
            hidden_layer_sizes=(24, 12),
            activation="relu",
            solver="adam",
            max_iter=3000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        modelo.fit(X_train_scaled, y_train_bal)

        y_pred = modelo.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self.modelo = modelo
        self.scaler = scaler
        self.accuracy = acc
        self.precision = prec
        self.recall = rec
        self.f1 = f1
        self.set_message(f"MLP entrenado | acc={acc:.3f} rec={rec:.3f} f1={f1:.3f}")

    def decision_auto_saltar(self):
        self.last_proba_salto = 0.0
        if self.modelo is None or self.scaler is None or not self.bala_activa:
            return

        distancia = float(self.distancia_jugador_bala())
        velocidad = float(self.bala_vel)
        velocidad_abs = max(abs(velocidad), 0.001)
        tiempo_impacto = distancia / velocidad_abs
        X = np.array([[velocidad, distancia, tiempo_impacto]], dtype=np.float32)
        Xs = self.scaler.transform(X)

        proba = self.modelo.predict_proba(Xs)[0]

        if len(self.modelo.classes_) == 2 and 1 in self.modelo.classes_:
            idx_clase_1 = list(self.modelo.classes_).index(1)
            p_salto = float(proba[idx_clase_1])
        else:
            pred = self.modelo.predict(Xs)[0]
            p_salto = 1.0 if pred == 1 else 0.0

        self.last_proba_salto = p_salto

        # Ajuste dinámico: si la bala viene más lenta, esperamos menos distancia
        # pero también bajamos el umbral del modelo para que no se quede pasivo.
        velocidad_abs = abs(velocidad)
        if velocidad_abs <= 7.0:
            umbral_modelo = 0.28
            umbral_distancia = 70
        elif velocidad_abs <= 9.0:
            umbral_modelo = 0.33
            umbral_distancia = 85
        else:
            umbral_modelo = 0.38
            umbral_distancia = 100

        if not self.saltando:
            if p_salto >= umbral_modelo:
                self.iniciar_salto()
            elif distancia <= umbral_distancia:
                self.iniciar_salto()

    # --------------------------------------------------------
    # Exportar CSV
    # --------------------------------------------------------
    def guardar_datos_csv(self, filename=None):
        if filename is None:
            filename = self.csv_filename

        if not self.nuevas_muestras:
            self.set_message("No hay muestras nuevas para guardar")
            return

        existentes = []
        existentes_keys = set()

        if os.path.exists(filename):
            try:
                with open(filename, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tiempo_impacto = float(row["tiempo_impacto"]) if "tiempo_impacto" in row and row["tiempo_impacto"] != "" else float(row["distancia"]) / max(abs(float(row["velocidad_bala"])), 0.001)
                        s = Sample(
                            velocidad_bala=float(row["velocidad_bala"]),
                            distancia=float(row["distancia"]),
                            tiempo_impacto=tiempo_impacto,
                            salto=int(float(row["salto"])),
                        )
                        existentes.append(s)
                        key = (round(s.velocidad_bala, 3), round(s.distancia, 3), round(s.tiempo_impacto, 3), int(s.salto))
                        existentes_keys.add(key)
            except Exception:
                existentes = []
                existentes_keys = set()

        agregadas = 0
        for s in self.nuevas_muestras:
            key = (round(s.velocidad_bala, 3), round(s.distancia, 3), round(s.tiempo_impacto, 3), int(s.salto))
            if key not in existentes_keys:
                existentes.append(s)
                existentes_keys.add(key)
                agregadas += 1

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["velocidad_bala", "distancia", "tiempo_impacto", "salto"])
            for s in existentes:
                writer.writerow([s.velocidad_bala, s.distancia, s.tiempo_impacto, s.salto])

        self.datos_modelo = existentes
        self.nuevas_muestras = []
        self.set_message(f"G: {agregadas} muestras nuevas guardadas | total={len(existentes)}")

    def cargar_csv_automatico(self, filename=None):
        if filename is None:
            filename = self.csv_filename

        if not os.path.exists(filename):
            return

        try:
            datos = []
            with open(filename, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tiempo_impacto = float(row["tiempo_impacto"]) if "tiempo_impacto" in row and row["tiempo_impacto"] != "" else float(row["distancia"]) / max(abs(float(row["velocidad_bala"])), 0.001)
                    datos.append(
                        Sample(
                            velocidad_bala=float(row["velocidad_bala"]),
                            distancia=float(row["distancia"]),
                            tiempo_impacto=tiempo_impacto,
                            salto=int(float(row["salto"])),
                        )
                    )
            self.datos_modelo = datos
            self.nuevas_muestras = []
            self.set_message(f"CSV cargado automáticamente: {len(datos)} muestras", 2.0)
        except Exception as e:
            self.set_message(f"Error cargando CSV: {e}")

    def draw_background(self):
        w, h = self.screen.get_size()
        self.screen.fill(self.CIELO)

        # Nubes simples
        for i in range(4):
            x = (i * 270 + 80 - (self.fondo_offset // 2)) % (w + 200) - 100
            y = 70 + (i % 2) * 35
            pygame.draw.ellipse(self.screen, self.BLANCO, (x, y, 90, 40))
            pygame.draw.ellipse(self.screen, self.BLANCO, (x + 30, y - 15, 80, 45))
            pygame.draw.ellipse(self.screen, self.BLANCO, (x + 55, y, 90, 40))

        # Suelo
        pygame.draw.rect(self.screen, self.SUELO, (0, self.ground_y, w, h - self.ground_y))
        pygame.draw.line(self.screen, (70, 130, 60), (0, self.ground_y), (w, self.ground_y), 4)

        # Líneas del piso para sensación de movimiento
        for x in range(-40, w + 80, 70):
            xx = (x - self.fondo_offset) % (w + 70) - 35
            pygame.draw.line(self.screen, (120, 100, 60), (xx, self.ground_y + 55), (xx + 35, self.ground_y + 55), 4)

    def draw_player(self):
        rect = self.get_player_rect()
        pygame.draw.rect(self.screen, self.AZUL, rect, border_radius=10)
        pygame.draw.rect(self.screen, self.NEGRO, rect, 2, border_radius=10)
        # Ojo
        pygame.draw.circle(self.screen, self.BLANCO, (rect.x + 50, rect.y + 25), 8)
        pygame.draw.circle(self.screen, self.NEGRO, (rect.x + 53, rect.y + 25), 3)

    def draw_bala(self):
        rect = self.get_bala_rect()
        pygame.draw.ellipse(self.screen, self.ROJO, rect)
        pygame.draw.ellipse(self.screen, self.NEGRO, rect, 2)
        # cola
        pygame.draw.line(self.screen, self.AMARILLO, (rect.right, rect.centery), (rect.right + 20, rect.centery), 4)

    def draw_hud(self):
        y = 18
        line_gap = 28

        saltos_1 = sum(1 for s in self.datos_modelo if s.salto == 1)
        saltos_0 = sum(1 for s in self.datos_modelo if s.salto == 0)

        textos = [
            f"Estado: {self.estado.upper()}",
            f"Modo: {self.modo if self.modo else '-'}",
            f"Score: {self.score}",
            f"Best: {self.best_score}",
            f"Muestras: {len(self.datos_modelo)}",
            f"Nuevas sin guardar: {len(self.nuevas_muestras)}",
            f"Clase 0 (no salto): {saltos_0}",
            f"Clase 1 (salto): {saltos_1}",
            f"Accuracy: {self.accuracy:.3f}" if self.accuracy is not None else "Accuracy: -",
            f"Recall salto: {self.recall:.3f}" if self.recall is not None else "Recall salto: -",
            f"F1 salto: {self.f1:.3f}" if self.f1 is not None else "F1 salto: -",
            f"proba_salto ≈ {self.last_proba_salto:.2f}",
            f"tti ≈ {max(self.distancia_jugador_bala(), 0) / max(abs(self.bala_vel), 0.001):.2f}",
        ]

        for t in textos:
            surf = self.font_small.render(t, True, self.TEXTO)
            self.screen.blit(surf, (18, y))
            y += line_gap

        distancia = self.distancia_jugador_bala()
        info = f"vel_bala={self.bala_vel:.2f}  distancia={distancia:.2f}"
        surf = self.font_small.render(info, True, self.TEXTO)
        self.screen.blit(surf, (18, y + 8))

        if self.mensaje and pygame.time.get_ticks() < self.mensaje_timer:
            msg = self.font_medium.render(self.mensaje, True, self.MORADO)
            self.screen.blit(msg, (18, self.screen.get_height() - 46))

    def draw_menu(self):
        self.draw_background()

        title = self.font_large.render("Juego Bala y Salto con MLP", True, self.TEXTO)
        subtitle = self.font_medium.render("Recolecta datos en manual, entrena y luego prueba AUTO", True, self.GRIS)

        self.screen.blit(title, (70, 70))
        self.screen.blit(subtitle, (70, 130))

        opciones = [
            "M = Manual (sin reiniciar dataset)",
            "A = Automático",
            "T = Entrenar MLP",
            "G = Guardar solo muestras nuevas en CSV",
            "ESPACIO = Saltar en manual",
            "ESC = Menú",
            "F = Fullscreen | Q = Salir",
        ]

        y = 210
        for op in opciones:
            surf = self.font_medium.render(op, True, self.TEXTO)
            self.screen.blit(surf, (90, y))
            y += 48

        resumen = [
            f"Muestras cargadas: {len(self.datos_modelo)}",
            f"Modelo entrenado: {'sí' if self.modelo is not None else 'no'}",
            f"Accuracy: {self.accuracy:.3f}" if self.accuracy is not None else "Accuracy: -",
            "Tip: ahora el MLP también usa tiempo estimado de impacto.",
        ]

        y += 25
        for r in resumen:
            surf = self.font_small.render(r, True, self.MORADO)
            self.screen.blit(surf, (90, y))
            y += 32

        if self.mensaje and pygame.time.get_ticks() < self.mensaje_timer:
            msg = self.font_medium.render(self.mensaje, True, self.MORADO)
            self.screen.blit(msg, (90, y + 20))

    def draw_game_over(self):
        self.draw_background()
        self.draw_player()
        self.draw_bala()
        self.draw_hud()

        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        self.screen.blit(overlay, (0, 0))

        msg1 = self.font_large.render("GAME OVER", True, self.BLANCO)
        msg2 = self.font_medium.render("R = Reintentar | ESC = Menú | Q = Salir", True, self.BLANCO)
        msg3 = self.font_medium.render(f"Score final: {self.score}", True, self.BLANCO)

        w, h = self.screen.get_size()
        self.screen.blit(msg1, (w // 2 - msg1.get_width() // 2, h // 2 - 90))
        self.screen.blit(msg3, (w // 2 - msg3.get_width() // 2, h // 2 - 25))
        self.screen.blit(msg2, (w // 2 - msg2.get_width() // 2, h // 2 + 30))

    def draw_game(self):
        self.draw_background()
        self.draw_player()
        self.draw_bala()
        self.draw_hud()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------
    def manejar_eventos(self):
        self.salto_solicitado_este_frame = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False

                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()

                elif event.key == pygame.K_t:
                    self.entrenar_modelo()

                elif event.key == pygame.K_g:
                    self.guardar_datos_csv()

                if self.estado == "menu":
                    if event.key == pygame.K_m:
                        self.iniciar_manual()
                    elif event.key == pygame.K_a:
                        self.iniciar_auto()

                elif self.estado in ("manual", "auto"):
                    if event.key in (pygame.K_ESCAPE, pygame.K_p):
                        self.volver_menu()
                    elif event.key == pygame.K_SPACE and self.estado == "manual":
                        self.iniciar_salto()

                elif self.estado == "game_over":
                    if event.key == pygame.K_r:
                        if self.modo == "manual":
                            self.estado = "manual"
                        elif self.modo == "auto":
                            self.estado = "auto"
                        self.reset_juego()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_p):
                        self.volver_menu()

    # --------------------------------------------------------
    # Update principal
    # --------------------------------------------------------
    def update(self):
        if self.estado not in ("manual", "auto"):
            return

        self.fondo_offset = (self.fondo_offset + self.fondo_speed) % self.BASE_W

        if self.estado == "auto":
            self.decision_auto_saltar()

        self.manejar_salto()

        if self.estado == "manual":
            self.registrar_decision_manual()

        self.actualizar_bala()
        self.checar_colision()

    # --------------------------------------------------------
    # Render principal
    # --------------------------------------------------------
    def draw(self):
        if self.estado == "menu":
            self.draw_menu()
        elif self.estado in ("manual", "auto"):
            self.draw_game()
        elif self.estado == "game_over":
            self.draw_game_over()

        pygame.display.flip()

    # --------------------------------------------------------
    # Loop principal
    # --------------------------------------------------------
    def run(self):
        while self.running:
            self.manejar_eventos()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    juego = JuegoMLP()
    juego.run()
