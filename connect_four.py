# connect_four.py
import sys
import math
import random
import time

import numpy as np
import pygame

# -----------------------
# Basic game settings
# -----------------------
ROWS = 6
COLS = 7
CONNECT_N = 4  # how many in a row to win

EMPTY = 0
PLAYER = 1
AI = 2

# Visual/grid sizes
SQUARE = 100
RADIUS = int(SQUARE / 2 - 5)
WIDTH = COLS * SQUARE
HEIGHT = (ROWS + 1) * SQUARE  # extra top row for hover display
WINDOW_SIZE = (WIDTH, HEIGHT)

# Colours (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BOARD_BLUE = (10, 80, 170)
BOARD_BLUE_LIGHT = (40, 110, 210)
PLAYER_RED = (255, 80, 80)
AI_YELLOW = (255, 215, 60)
HIGHLIGHT_GREEN = (0, 200, 0)
DIM_GREY = (30, 30, 30)

# Sound sample rate for generated tones
SAMPLE_RATE = 44100

# Difficulty mapping (Kid, Teen, Adult)
AI_DEPTH_MAP = {
    "Kid": 2,    # Easy
    "Teen": 3,   # Medium
    "Adult": 5   # Hard
}

# -----------------------
# Utility: board helpers
# -----------------------
def make_board():
    """Create a blank game board as a numpy array. 0 means empty."""
    return np.zeros((ROWS, COLS), dtype=int)


def can_drop(board, col):
    """Is there space to drop a piece in column `col`?"""
    return board[ROWS - 1][col] == EMPTY


def next_open_row(board, col):
    """Return the lowest empty row index in `col`, or None if the column is full."""
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None


def place_piece(board, row, col, piece):
    """Place a piece (PLAYER or AI) into the board at (row, col)."""
    board[row][col] = piece


def print_board(board):
    """Console-friendly print of the board (flipped so bottom row prints last)."""
    print(np.flip(board, 0))


# -----------------------
# Winning detection (with coordinates)
# -----------------------
def find_winning_line(board, piece):
    """
    If `piece` has CONNECT_N in a row, return list of coordinates [(r,c), ...].
    Otherwise return None.
    """
    # Horizontal
    for c in range(COLS - (CONNECT_N - 1)):
        for r in range(ROWS):
            coords = [(r, c + i) for i in range(CONNECT_N)]
            if all(board[r][c + i] == piece for i in range(CONNECT_N)):
                return coords

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - (CONNECT_N - 1)):
            coords = [(r + i, c) for i in range(CONNECT_N)]
            if all(board[r + i][c] == piece for i in range(CONNECT_N)):
                return coords

    # Positive slope diagonal
    for c in range(COLS - (CONNECT_N - 1)):
        for r in range(ROWS - (CONNECT_N - 1)):
            coords = [(r + i, c + i) for i in range(CONNECT_N)]
            if all(board[r + i][c + i] == piece for i in range(CONNECT_N)):
                return coords

    # Negative slope diagonal
    for c in range(COLS - (CONNECT_N - 1)):
        for r in range(CONNECT_N - 1, ROWS):
            coords = [(r - i, c + i) for i in range(CONNECT_N)]
            if all(board[r - i][c + i] == piece for i in range(CONNECT_N)):
                return coords

    return None


def has_winner(board, piece):
    """Boolean wrapper for find_winning_line."""
    return find_winning_line(board, piece) is not None


# -----------------------
# Heuristic / evaluation
# -----------------------
def evaluate_window(window, piece):
    """
    Score a window (list of values) from the perspective of `piece`.
    This is intentionally simple but works well enough for Connect Four.
    """
    score = 0
    opp = PLAYER if piece == AI else AI

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score


def board_score(board, piece):
    """
    Heuristic board evaluation for `piece`.
    Center control + window evaluations across directions.
    """
    score = 0

    # center column preference
    center_col = list(board[:, COLS // 2])
    center_count = center_col.count(piece)
    score += center_count * 3

    # horizontal
    for r in range(ROWS):
        row_vals = list(board[r, :])
        for c in range(COLS - (CONNECT_N - 1)):
            window = row_vals[c:c + CONNECT_N]
            score += evaluate_window(window, piece)

    # vertical
    for c in range(COLS):
        col_vals = list(board[:, c])
        for r in range(ROWS - (CONNECT_N - 1)):
            window = col_vals[r:r + CONNECT_N]
            score += evaluate_window(window, piece)

    # positive diagonal
    for r in range(ROWS - (CONNECT_N - 1)):
        for c in range(COLS - (CONNECT_N - 1)):
            window = [board[r + i][c + i] for i in range(CONNECT_N)]
            score += evaluate_window(window, piece)

    # negative diagonal
    for r in range(ROWS - (CONNECT_N - 1)):
        for c in range(COLS - (CONNECT_N - 1)):
            window = [board[r + (CONNECT_N - 1) - i][c + i] for i in range(CONNECT_N)]
            score += evaluate_window(window, piece)

    return score


def board_is_terminal(board):
    """True if the game is over (win for anyone or draw)."""
    return has_winner(board, PLAYER) or has_winner(board, AI) or len(valid_moves(board)) == 0


# -----------------------
# Minimax with alpha-beta
# -----------------------
def valid_moves(board):
    """Return a list of columns that are not full."""
    return [c for c in range(COLS) if can_drop(board, c)]


def minimax(board, depth, alpha, beta, maximizing):
    """
    Minimax search with alpha-beta pruning.
    Returns tuple (best_col, score). best_col may be None at leaf nodes.
    Move ordering tries center columns first for better pruning.
    """
    valid_cols = valid_moves(board)
    terminal = board_is_terminal(board)

    if depth == 0 or terminal:
        if terminal:
            if has_winner(board, AI):
                return (None, float('inf'))
            elif has_winner(board, PLAYER):
                return (None, -float('inf'))
            else:
                return (None, 0)
        else:
            return (None, board_score(board, AI))

    if maximizing:
        value = -math.inf
        best_col = random.choice(valid_cols) if valid_cols else None
        ordered = sorted(valid_cols, key=lambda c: abs(c - COLS // 2))
        for col in ordered:
            row = next_open_row(board, col)
            if row is None:
                continue
            board_copy = board.copy()
            place_piece(board_copy, row, col, AI)
            new_score = minimax(board_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(valid_cols) if valid_cols else None
        ordered = sorted(valid_cols, key=lambda c: abs(c - COLS // 2))
        for col in ordered:
            row = next_open_row(board, col)
            if row is None:
                continue
            board_copy = board.copy()
            place_piece(board_copy, row, col, PLAYER)
            new_score = minimax(board_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value


def quick_best_move(board, piece):
    """Quick heuristic move (fallback). Good for tie-breaking and ordering."""
    valid_cols = valid_moves(board)
    best_score = -1e9
    best_col = random.choice(valid_cols) if valid_cols else None
    for col in valid_cols:
        row = next_open_row(board, col)
        if row is None:
            continue
        copy_board = board.copy()
        place_piece(copy_board, row, col, piece)
        sc = board_score(copy_board, piece)
        if sc > best_score:
            best_score = sc
            best_col = col
    return best_col


# -----------------------
# Graphics helpers: drawing + animation
# -----------------------
def draw_board(screen, board, last_move=None, dim_amount=0.0):
    """
    Draw the whole board and the UI top bar.
    dim_amount: 0.0 means normal. 0.5 means half-dim overlay on top for focus effect.
    last_move: tuple (row, col, piece) to highlight
    """
    # top info background
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARE))

    # board grid and empty circles
    for c in range(COLS):
        for r in range(ROWS):
            rect = (c * SQUARE, r * SQUARE + SQUARE, SQUARE, SQUARE)
            pygame.draw.rect(screen, BOARD_BLUE, rect)
            center = (int(c * SQUARE + SQUARE / 2), int(r * SQUARE + SQUARE + SQUARE / 2))
            pygame.draw.circle(screen, BLACK, center, RADIUS)

    # pieces (note: board row 0 is bottom in our coordinate scheme printing but we draw accordingly)
    for c in range(COLS):
        for r in range(ROWS):
            center = (int(c * SQUARE + SQUARE / 2), HEIGHT - int(r * SQUARE + SQUARE / 2))
            if board[r][c] == PLAYER:
                pygame.draw.circle(screen, PLAYER_RED, center, RADIUS)
            elif board[r][c] == AI:
                pygame.draw.circle(screen, AI_YELLOW, center, RADIUS)

    # highlight last move with a thin ring
    if last_move is not None:
        lr, lc, lp = last_move
        cx = int(lc * SQUARE + SQUARE / 2)
        cy = HEIGHT - int(lr * SQUARE + SQUARE / 2)
        color = HIGHLIGHT_GREEN if lp == PLAYER else DIM_GREY
        pygame.draw.circle(screen, color, (cx, cy), RADIUS + 4, 4)

    # draw restart button on the top right
    btn_w, btn_h = 140, 40
    btn_x, btn_y = WIDTH - btn_w - 10, 10
    pygame.draw.rect(screen, (200, 200, 200), (btn_x, btn_y, btn_w, btn_h), border_radius=6)
    small_font = pygame.font.SysFont("monospace", 20)
    txt = small_font.render("Restart", True, BLACK)
    screen.blit(txt, (btn_x + (btn_w - txt.get_width()) // 2, btn_y + (btn_h - txt.get_height()) // 2))

    # optional dim overlay for focus on win animation
    if dim_amount > 0.0:
        # clamp 0.0..1.0
        d = max(0.0, min(1.0, dim_amount))
        overlay = pygame.Surface((WIDTH, HEIGHT), flags=pygame.SRCALPHA)
        overlay.fill((0, 0, 0, int(180 * d)))  # alpha overlay
        screen.blit(overlay, (0, 0))

    pygame.display.update()


def animate_drop(screen, board, col, piece, last_move_holder):
    """
    Animate a piece falling into column `col`.
    last_move_holder is a small dict {'last_move': (r,c,piece)} so we can update outside.
    Returns the landed row or None if cancelled by restart/quit.
    """
    if not can_drop(board, col):
        return None

    target_row = next_open_row(board, col)
    if target_row is None:
        return None

    start_y = int(SQUARE / 2)
    final_y = HEIGHT - int(target_row * SQUARE + SQUARE / 2)
    cx = int(col * SQUARE + SQUARE / 2)

    y = start_y
    speed = 18  # pixels per frame, tweak for feeling
    clock = pygame.time.Clock()

    while y < final_y:
        # handle events to stay responsive during animation
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                # check restart button click while animating
                btn_w, btn_h = 140, 40
                btn_x, btn_y = WIDTH - btn_w - 10, 10
                if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                    return None

        # redraw the board and draw the falling piece on top
        draw_board(screen, board, last_move_holder.get('last_move'))
        if piece == PLAYER:
            pygame.draw.circle(screen, PLAYER_RED, (cx, int(y)), RADIUS)
        else:
            pygame.draw.circle(screen, AI_YELLOW, (cx, int(y)), RADIUS)

        pygame.display.update()
        y += speed
        clock.tick(60)

    # final placement
    place_piece(board, target_row, col, piece)
    last_move_holder['last_move'] = (target_row, col, piece)
    draw_board(screen, board, last_move_holder.get('last_move'))
    return target_row


# -----------------------
# Win/Lose flashy animations + sounds
# -----------------------
def generate_sine_wave(freq, duration, volume=0.5):
    """
    Return a numpy int16 wave array for given frequency & duration.
    volume between 0.0 and 1.0.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * 2 * np.pi * t)
    # scale to int16
    audio = (wave * (volume * 32767)).astype(np.int16)
    # make stereo by stacking columns
    stereo = np.column_stack((audio, audio))
    return stereo


def play_tone(freq, duration=0.25, volume=0.5):
    """Play a quick tone using pygame.sndarray (non-blocking)."""
    try:
        arr = generate_sine_wave(freq, duration, volume)
        sound = pygame.sndarray.make_sound(arr)
        sound.play()
    except Exception:
        # In environments without audio available, ignore sound errors
        pass


def flash_win_animation(screen, board, winning_coords, winner_piece):
    """
    Flash the winning four with pulsing glow, dim the rest of the board,
    show a pulsing message, and play a victory/defeat tone.
    """
    # precompute screen centers for the winning cells
    centers = []
    for (r, c) in winning_coords:
        cx = int(c * SQUARE + SQUARE / 2)
        cy = HEIGHT - int(r * SQUARE + SQUARE / 2)
        centers.append((cx, cy))

    # animation parameters
    frames = 40  # number of frames (~2 seconds at 20 fps)
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("monospace", 64)

    # play sound once at start
    if winner_piece == PLAYER:
        # happy chime sequence
        play_tone(880, 0.14, 0.6)
        pygame.time.wait(80)
        play_tone(1100, 0.14, 0.6)
    else:
        # low defeat tone sequence
        play_tone(200, 0.4, 0.6)
        pygame.time.wait(120)
        play_tone(140, 0.35, 0.6)

    # animate: pulse brightness and dim background
    for i in range(frames):
        # pulse factor: 0..1..0
        t = math.sin(math.pi * i / frames) ** 2  # nice ease in/out pulse
        draw_board(screen, board, dim_amount=0.55)  # dim the whole board for emphasis

        # draw pulsing glow ring around each winning piece
        for center in centers:
            cx, cy = center
            pulse_radius = int(RADIUS + 6 + t * 20)
            color = PLAYER_RED if winner_piece == PLAYER else AI_YELLOW
            # outer glow (faded)
            s = pygame.Surface((pulse_radius * 2 + 6, pulse_radius * 2 + 6), pygame.SRCALPHA)
            glow_color = (color[0], color[1], color[2], int(80 + t * 120))
            pygame.draw.circle(s, glow_color, (pulse_radius + 3, pulse_radius + 3), pulse_radius)
            screen.blit(s, (cx - pulse_radius - 3, cy - pulse_radius - 3))
            # inner ring
            pygame.draw.circle(screen, WHITE, (cx, cy), int(RADIUS + 2 + t * 6), 3)

        # pulsing message
        if winner_piece == PLAYER:
            message = "🔥 YOU WIN!"
        else:
            message = "💀 AI WINS!"

        # pulsing text scale
        scale = 1.0 + 0.12 * t
        txt_surf = font_big.render(message, True, WHITE)
        txt_rect = txt_surf.get_rect(center=(WIDTH // 2, SQUARE // 2))
        # scale manually using transform.smoothscale
        surf = pygame.transform.smoothscale(txt_surf, (int(txt_rect.width * scale), int(txt_rect.height * scale)))
        rect = surf.get_rect(center=(WIDTH // 2, SQUARE // 2))
        screen.blit(surf, rect)

        pygame.display.update()
        clock.tick(20)

    # short pause so user sees the result
    pygame.time.wait(600)


# -----------------------
# Menu icons (generated at runtime)
# -----------------------
def make_icon_surface(diameter, color, two_dots=False):
    """
    Create a circular icon surface of given diameter and color.
    If two_dots=True, draws two small white dots (used for Local).
    """
    surf = pygame.Surface((diameter, diameter), flags=pygame.SRCALPHA)
    pygame.draw.circle(surf, color, (diameter // 2, diameter // 2), diameter // 2)
    if two_dots:
        dot_r = max(3, diameter // 8)
        offset = diameter // 5
        pygame.draw.circle(surf, WHITE, (diameter // 2 - offset, diameter // 2), dot_r)
        pygame.draw.circle(surf, WHITE, (diameter // 2 + offset, diameter // 2), dot_r)
    return surf


# -----------------------
# MENU + UI helpers (style B: board blue themed)
# -----------------------
def draw_menu_button_with_icon(screen, text, icon_surf, x, y, w, h, font, hover=False):
    # background
    bg = BOARD_BLUE_LIGHT if hover else BOARD_BLUE
    border_col = WHITE if hover else (180, 200, 230)
    pygame.draw.rect(screen, bg, (x, y, w, h), border_radius=10)
    pygame.draw.rect(screen, border_col, (x + 2, y + 2, w - 4, h - 4), width=2, border_radius=8)
    # icon
    if icon_surf:
        icon_rect = icon_surf.get_rect()
        icon_x = x + 12
        icon_y = y + (h - icon_rect.height) // 2
        screen.blit(icon_surf, (icon_x, icon_y))
        text_x = icon_x + icon_rect.width + 12
    else:
        text_x = x + 20
    # text
    label = font.render(text, True, WHITE if not hover else BLACK)
    screen.blit(label, (text_x, y + (h - label.get_height()) // 2))


def main_menu(screen):
    font_title = pygame.font.SysFont("monospace", 64, bold=True)
    font_btn = pygame.font.SysFont("monospace", 32)
    clock = pygame.time.Clock()

    # create icon surfaces once
    icon_size = 44
    kid_icon = make_icon_surface(icon_size, (120, 180, 255))   # baby-blue
    teen_icon = make_icon_surface(icon_size, (120, 255, 120))  # green
    adult_icon = make_icon_surface(icon_size, (255, 150, 60))  # orange
    local_icon = make_icon_surface(icon_size, (180, 120, 255), two_dots=True)  # purple w/ two dots

    buttons = [
        ("Kid   ",   "Kid", kid_icon),
        ("Teen  ",   "Teen", teen_icon),
        ("Adult ",   "Adult", adult_icon),
        ("Local ",   "PVP",  local_icon),
        ("Quit ",    "QUIT", None),
    ]

    while True:
        screen.fill(BLACK)

        # Draw a large board-like panel
        panel_w, panel_h = WIDTH - 160, HEIGHT - 160
        panel_x, panel_y = 80, 40
        pygame.draw.rect(screen, BOARD_BLUE, (panel_x, panel_y, panel_w, panel_h), border_radius=18)
        pygame.draw.rect(screen, BOARD_BLUE_LIGHT, (panel_x + 6, panel_y + 6, panel_w - 12, panel_h - 12), width=2, border_radius=14)

        # Title
        title = font_title.render("CONNECT FOUR", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, panel_y + 18))

        mx, my = pygame.mouse.get_pos()

        # Draw buttons in panel
        btn_w, btn_h = 420, 64
        start_y = panel_y + 120
        spacing = 20
        for i, (label, tag, icon) in enumerate(buttons):
            x = WIDTH // 2 - btn_w // 2
            y = start_y + i * (btn_h + spacing)
            hover = (x <= mx <= x + btn_w and y <= my <= y + btn_h)
            draw_menu_button_with_icon(screen, label, icon, x, y, btn_w, btn_h, font_btn, hover)

        pygame.display.update()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for i, (label, tag, icon) in enumerate(buttons):
                    x = WIDTH // 2 - btn_w // 2
                    y = start_y + i * (btn_h + spacing)
                    if x <= mx <= x + btn_w and y <= my <= y + btn_h:
                        if tag == "QUIT":
                            pygame.quit()
                            sys.exit()
                        if tag == "PVP":
                            return ("PVP", None)
                        else:
                            return ("AI", AI_DEPTH_MAP[tag])

        clock.tick(60)


# -----------------------
# Main game loop
# -----------------------
def main():
    pygame.init()
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
    except Exception:
        # continue without audio if init fails
        pass
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Saleh's Connect Four")
    font_big = pygame.font.SysFont("monospace", 48)
    font_small = pygame.font.SysFont("monospace", 22)

    # choose mode
    mode, chosen_depth = main_menu(screen)
    current_ai_depth = chosen_depth if mode == "AI" else None

    board = make_board()
    last_move_holder = {'last_move': None}  # small mutable holder to pass between functions
    game_over = False
    turn = PLAYER  # player goes first
    clock = pygame.time.Clock()
    ai_thinking = False
    ai_start_time = None

    draw_board(screen, board, None)

    while True:
        # process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if game_over:
                # Click anywhere to restart (goes back to menu)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return main()  # restart by returning to menu
                continue

            if event.type == pygame.MOUSEMOTION:
                # draw hover piece on the top row while it's player's turn
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARE))
                posx = event.pos[0]
                if mode == "PVP":
                    color = PLAYER_RED if turn == PLAYER else AI_YELLOW
                    pygame.draw.circle(screen, color, (posx, int(SQUARE / 2)), RADIUS)
                else:
                    if turn == PLAYER:
                        pygame.draw.circle(screen, PLAYER_RED, (posx, int(SQUARE / 2)), RADIUS)
                screen.blit(font_small.render("Click Restart to reset (or click after game to return to menu)", True, WHITE), (10, 10))
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # check restart button
                btn_w, btn_h = 140, 40
                btn_x, btn_y = WIDTH - btn_w - 10, 10
                if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                    return main()

                # player's move when it's their turn
                if (mode == "PVP") or (mode == "AI" and turn == PLAYER):
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARE))
                    if col < 0 or col >= COLS:
                        continue
                    if can_drop(board, col):
                        piece_type = PLAYER if turn == PLAYER else AI
                        landed = animate_drop(screen, board, col, piece_type, last_move_holder)
                        if landed is None:
                            # animation cancelled (restart clicked) -> skip
                            continue

                        # check for win
                        win_coords = find_winning_line(board, piece_type)
                        if win_coords:
                            flash_win_animation(screen, board, win_coords, piece_type)
                            game_over = True

                        # switch turn
                        if mode == "PVP":
                            turn = PLAYER if turn == AI else AI
                        else:
                            # vs AI, switch to AI after player's move
                            turn = AI

                        draw_board(screen, board, last_move_holder.get('last_move'))

        # AI turn — we do it outside of events so animations & UI are responsive
        if not game_over and mode == "AI" and turn == AI:
            if not ai_thinking:
                ai_thinking = True
                ai_start_time = time.time()
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARE))
                screen.blit(font_small.render("AI thinking...", True, WHITE), (10, 10))
                pygame.display.update()

            # compute move using the selected depth
            col, score = minimax(board, current_ai_depth, -math.inf, math.inf, True)
            if col is None:
                col = quick_best_move(board, AI)

            # tiny delay for UX and to let "AI thinking..." show
            elapsed = time.time() - (ai_start_time or time.time())
            if elapsed < 0.12:
                pygame.time.wait(int((0.12 - elapsed) * 1000))

            # animate AI drop
            landed = animate_drop(screen, board, col, AI, last_move_holder)
            if landed is None:
                # animation might have been cancelled by restart click
                ai_thinking = False
                continue

            # check for AI win
            win_coords = find_winning_line(board, AI)
            if win_coords:
                flash_win_animation(screen, board, win_coords, AI)
                game_over = True

            draw_board(screen, board, last_move_holder.get('last_move'))
            turn = PLAYER
            ai_thinking = False

        # draw draw-check
        if not game_over and len(valid_moves(board)) == 0:
            label = font_big.render("Draw!", True, WHITE)
            screen.blit(label, (40, 10))
            pygame.display.update()
            game_over = True

        clock.tick(60)


if __name__ == "__main__":
    main()

