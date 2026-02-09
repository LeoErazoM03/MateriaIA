import tkinter as tk
from tkinter import messagebox
import math

# ---------- Lógica del juego ----------

def check_winner(board):
    combos = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in combos:
        if board[a] == board[b] == board[c] != "":
            return board[a]
    return None


def board_full(board):
    return "" not in board


def minimax(b, is_max):
    winner = check_winner(b)
    if winner == "O": return 1
    if winner == "X": return -1
    if board_full(b): return 0

    if is_max:
        best = -math.inf
        for i in range(9):
            if b[i] == "":
                b[i] = "O"
                val = minimax(b, False)
                b[i] = ""
                best = max(best, val)
        return best
    else:
        best = math.inf
        for i in range(9):
            if b[i] == "":
                b[i] = "X"
                val = minimax(b, True)
                b[i] = ""
                best = min(best, val)
        return best


def best_move():
    best_val = -math.inf
    move = None
    for i in range(9):
        if board[i] == "":
            board[i] = "O"
            val = minimax(board, False)
            board[i] = ""
            if val > best_val:
                best_val = val
                move = i
    return move


def click(i):
    global game_over
    if board[i] == "" and not game_over:
        board[i] = "X"
        buttons[i].config(text="X", fg="#1e88e5")
        if check_end(): return

        ai = best_move()
        if ai is not None:
            board[ai] = "O"
            buttons[ai].config(text="O", fg="#e53935")
            check_end()


def check_end():
    global game_over
    winner = check_winner(board)
    if winner:
        game_over = True
        messagebox.showinfo("Resultado", f"¡Gana {winner}! aprende a jugar humano manco.")
        return True
    if board_full(board):
        game_over = True
        messagebox.showinfo("Resultado", "¡Empate!")
        return True
    return False


def reset():
    global board, game_over
    board = [""] * 9
    game_over = False
    for b in buttons:
        b.config(text="", fg="black")

# ---------- Interfaz Moderna ----------

root = tk.Tk()
root.title("Tic Tac Toe vs IA")
root.geometry("360x460")
root.configure(bg="#1f2933")
root.resizable(False, False)

board = [""] * 9
game_over = False

# Título

title = tk.Label(root, text="Tic Tac Toe", font=("Segoe UI", 22, "bold"), bg="#1f2933", fg="white")
title.pack(pady=10)

subtitle = tk.Label(root, text="Jugador vs Máquina", font=("Segoe UI", 12), bg="#1f2933", fg="#9ca3af")
subtitle.pack(pady=5)

# Marco del tablero

frame = tk.Frame(root, bg="#1f2933")
frame.pack(pady=20)

buttons = []

for i in range(9):
    btn = tk.Button(frame, text="", font=("Segoe UI", 28, "bold"), width=3, height=1,
                    bg="#111827", fg="white", relief="flat",
                    activebackground="#374151",
                    command=lambda i=i: click(i))
    btn.grid(row=i//3, column=i%3, padx=5, pady=5)
    buttons.append(btn)

# Botón reinicio

reset_btn = tk.Button(root, text="Reiniciar Juego", font=("Segoe UI", 12, "bold"),
                      bg="#2563eb", fg="white", relief="flat",
                      activebackground="#1d4ed8",
                      command=reset)
reset_btn.pack(fill="x", padx=40, pady=20)

root.mainloop()