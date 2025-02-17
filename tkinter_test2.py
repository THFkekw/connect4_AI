import numpy as np
import tkinter as tk
from tkinter import messagebox
from kaggle_environments import make



# Erstelle eine ConnectX-Umgebung (4-Gewinnt)
env = make("connectx", debug=True)
trainer = None  # Wird nach Spielerwahl initialisiert

# Spielfeld-Parameter
ROWS, COLS = 6, 7
board = np.zeros((ROWS, COLS), dtype=int)  # Start-Board
current_player = 1  # Standard: Mensch beginnt (1 = Mensch, 2 = KI)
game_active = False  # Verhindert ungewollte Züge vor Spielstart

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
    global board, current_player, game_active
    if not game_active:  # Verhindere falsche Züge
        return

    new_state, _, done, _ = trainer.step(col)  # Mensch macht einen Zug
    board[:] = np.array(new_state["board"]).reshape(ROWS, COLS)  # Board aktualisieren
    draw_board()

    if done:  # Falls das Spiel vorbei ist
        messagebox.showinfo("Spielende", "Spiel vorbei!")
        root.quit()
        return

    #current_player = 1  # Wechsel zur KI  # KI führt nach 500ms ihren Zug aus


def on_click(event):
    """Fängt Mausklicks ab und bestimmt die Spalte"""
    col = event.x // 100
    drop_piece(col)

def start_game(player):
    """Initialisiert das Spiel basierend auf der Spielerwahl"""
    global trainer, current_player, game_active
    trainer = env.play(["random",None ])  # KI spielt mit Zufallsstrategie
    #trainer.reset()  # Reset das Spiel
    current_player = 1
    if player == "Mensch":
        current_player =  2  # Setze Startspieler
    game_active = True  # Das Spiel ist nun aktiv
    if current_player == 1:
        pass

    draw_board()  # Starte das Spielbrett

# Auswahl-Menü für den Startspieler
menu = tk.Toplevel(root)
menu.title("Wer soll anfangen?")
menu.geometry("250x100")

tk.Label(menu, text="Wähle den Startspieler:").pack(pady=10)
tk.Button(menu, text="Mensch beginnt", command=lambda: [menu.destroy(), start_game("Mensch")]).pack()
tk.Button(menu, text="KI beginnt", command=lambda: [menu.destroy(), start_game("KI")]).pack()

# Starte das Spiel
canvas.bind("<Button-1>", on_click)
draw_board()
root.mainloop()
