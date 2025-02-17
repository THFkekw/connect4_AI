import numpy as np
import tkinter as tk
from kaggle_environments import make

# Erstelle eine ConnectX-Umgebung (4-Gewinnt)
env = make("connectx", debug=True)
trainer = env.train([None, "random"])  # Spieler gegen Zufallsgegner

# Spielfeld-Parameter
ROWS, COLS = 6, 7
board = np.array(trainer.reset()["board"]).reshape(ROWS, COLS)

# Tkinter Setup
root = tk.Tk()
root.title("4 Gewinnt - Kaggle Environments")
canvas = tk.Canvas(root, width=COLS * 100, height=ROWS * 100, bg="blue")
canvas.pack()

def draw_board():
    """Zeichnet das Spielfeld mit aktuellem Board-Zustand"""
    canvas.delete("all")
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * 100 + 50, r * 100 + 50
            color = "white"
            if board[r, c] == 1:
                color = "red"
            elif board[r, c] == 2:
                color = "yellow"
            canvas.create_oval(x-40, y-40, x+40, y+40, fill=color, outline="black")

def drop_piece(col):
    """Setzt einen Stein in die gewählte Spalte und aktualisiert das Board"""
    global board
    new_state, _, done, _ = trainer.step(col)  # Führe den Zug aus
    board = np.array(new_state["board"]).reshape(ROWS, COLS)  # Aktualisiere das Board
    draw_board()
    
    if done:  # Falls das Spiel vorbei ist
        tk.messagebox.showinfo("Spielende", "Spiel vorbei!")
        root.quit()

def on_click(event):
    """Fängt Mausklicks ab und bestimmt die Spalte"""
    col = event.x // 100
    drop_piece(col)

# Starte das Spiel
canvas.bind("<Button-1>", on_click)
draw_board()
root.mainloop()
