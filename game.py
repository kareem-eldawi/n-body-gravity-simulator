import random

import numpy as np
import pygame

from body import Body
from physics import G
from barnes_hut import step_verlet_bh

# ── Window layout ─────────────────────────────────────────────────────────────
SIM_SIZE = 700                 # simulation area is a square
PANEL_W = 300                  # control panel on the right
WIDTH = SIM_SIZE + PANEL_W
HEIGHT = SIM_SIZE
CENTER = SIM_SIZE // 2

SIM_RECT = pygame.Rect(0, 0, SIM_SIZE, SIM_SIZE)
PANEL_RECT = pygame.Rect(SIM_SIZE, 0, PANEL_W, HEIGHT)

# ── Catalogue of bodies ───────────────────────────────────────────────────────
# distance = mean orbital radius in metres, speed = mean orbital speed in m/s.
# Every planet starts on the +x axis moving in +y, so the full position and
# velocity vectors can be built from these two scalars.
SOLAR_SYSTEM = {
    "Sun":     {"mass": 1.989e30, "distance": 0.0,      "speed": 0.0,   "color": (251, 191, 36),  "size": 14},
    "Mercury": {"mass": 3.301e23, "distance": 5.791e10, "speed": 47360, "color": (168, 162, 158), "size": 3},
    "Venus":   {"mass": 4.867e24, "distance": 1.082e11, "speed": 35020, "color": (232, 180, 110), "size": 5},
    "Earth":   {"mass": 5.972e24, "distance": 1.496e11, "speed": 29780, "color": (96, 165, 250),  "size": 5},
    "Mars":    {"mass": 6.417e23, "distance": 2.279e11, "speed": 24070, "color": (248, 113, 113), "size": 4},
    "Jupiter": {"mass": 1.898e27, "distance": 7.785e11, "speed": 13070, "color": (217, 164, 118), "size": 10},
    "Saturn":  {"mass": 5.683e26, "distance": 1.434e12, "speed": 9680,  "color": (234, 214, 160), "size": 9},
    "Uranus":  {"mass": 8.681e25, "distance": 2.871e12, "speed": 6800,  "color": (147, 212, 222), "size": 7},
    "Neptune": {"mass": 1.024e26, "distance": 4.495e12, "speed": 5430,  "color": (99, 134, 230),  "size": 7},
}

DEFAULT_COLOR = (170, 176, 190)
DEFAULT_SIZE = 2

# Scale that just fits Neptune's orbit inside the simulation square, used as
# the zoom-out end of the slider.
BASE_SCALE = (CENTER - 10) / SOLAR_SYSTEM["Neptune"]["distance"]
ZOOM_MIN = 1.0
ZOOM_MAX = 40.0

THETA = 0.5

# Panel colours
BG = (15, 17, 23)
EDGE = (51, 65, 85)
FILL_ON = (30, 41, 59)
LINE_ON = (71, 85, 105)
LINE_OFF = (40, 48, 62)
TEXT = (226, 232, 240)
TEXT_DIM = (100, 116, 139)
TEXT_OFF = (90, 98, 112)


def build_bodies(enabled_names):
    """Fresh Body objects for the named bodies, at their true starting state."""
    bodies = []
    for name in SOLAR_SYSTEM:              # dict order keeps Sun first
        if name in enabled_names:
            d = SOLAR_SYSTEM[name]
            bodies.append(
                Body(d["mass"], [0.0, d["speed"]], [d["distance"], 0.0], name)
            )
    return bodies


def spawn_cluster(bodies, count, scale, start_index):
    """Add random bodies on roughly circular orbits around the heaviest body."""
    central_mass = max((b.mass for b in bodies), default=1.989e30)
    half_view = (CENTER - 20) / scale          # metres currently visible
    for i in range(count):
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(half_view * 0.15, half_view * 0.95)
        position = [dist * np.cos(angle), dist * np.sin(angle)]

        # Circular-orbit speed at this radius, directed tangentially
        speed = np.sqrt(G * central_mass / dist)
        velocity = [-np.sin(angle) * speed, np.cos(angle) * speed]

        bodies.append(Body(random.uniform(1e23, 1e25), velocity, position,
                           f"Extra{start_index + i + 1}"))
    return start_index + count


def to_screen(position, scale):
    return position * scale + CENTER


def to_world(screen_pos, scale):
    return (np.array(screen_pos, dtype=float) - CENTER) / scale


# ── Setup ─────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("N-Body Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Helvetica", 14)
small_font = pygame.font.SysFont("Helvetica", 12)

enabled = {"Sun", "Mercury", "Venus", "Earth", "Mars"}
bodies = build_bodies(enabled)
extra_count = 0                 # names spawned bodies, keeps them unique

zoom = 6.0
scale = BASE_SCALE * zoom
dt = 52560.0                    # simulated seconds per frame (~14.6 hours)
paused = False

fade = pygame.Surface((SIM_SIZE, SIM_SIZE))
fade.set_alpha(22)
fade.fill((0, 0, 0))
screen.fill((0, 0, 0))

# ── Panel geometry ────────────────────────────────────────────────────────────
PAD = 16
BTN_W = PANEL_W - 2 * PAD

toggle_rects = {}
y = PAD + 22
for name in SOLAR_SYSTEM:
    toggle_rects[name] = pygame.Rect(SIM_SIZE + PAD, y, BTN_W, 24)
    y += 28

slider_rect = pygame.Rect(SIM_SIZE + PAD, y + 34, BTN_W, 6)

# Spawn buttons sit side by side on one row
spawn_y = slider_rect.bottom + 38
half = (BTN_W - 8) // 2
spawn_50_rect = pygame.Rect(SIM_SIZE + PAD, spawn_y, half, 28)
spawn_200_rect = pygame.Rect(SIM_SIZE + PAD + half + 8, spawn_y, half, 28)

clear_rect = pygame.Rect(SIM_SIZE + PAD, spawn_y + 36, BTN_W, 28)
reset_rect = pygame.Rect(SIM_SIZE + PAD, spawn_y + 72, BTN_W, 30)

dragging_slider = False


def zoom_from_mouse(mx):
    t = (mx - slider_rect.x) / slider_rect.width
    t = min(max(t, 0.0), 1.0)
    return ZOOM_MIN + t * (ZOOM_MAX - ZOOM_MIN)


def draw_button(rect, label, strong=False):
    pygame.draw.rect(screen, FILL_ON, rect, border_radius=6)
    pygame.draw.rect(screen, LINE_ON if strong else LINE_OFF, rect, 1, border_radius=6)
    text = font.render(label, True, TEXT)
    screen.blit(text, (rect.centerx - text.get_width() // 2,
                       rect.centery - text.get_height() // 2))


def draw_panel():
    pygame.draw.rect(screen, BG, PANEL_RECT)
    pygame.draw.line(screen, EDGE, (SIM_SIZE, 0), (SIM_SIZE, HEIGHT))

    screen.blit(small_font.render("BODIES  (click to toggle)", True, TEXT_DIM),
                (SIM_SIZE + PAD, PAD))

    for name, rect in toggle_rects.items():
        on = name in enabled
        pygame.draw.rect(screen, FILL_ON if on else BG, rect, border_radius=5)
        pygame.draw.rect(screen, LINE_ON if on else LINE_OFF, rect, 1, border_radius=5)
        dot = SOLAR_SYSTEM[name]["color"] if on else (60, 66, 78)
        pygame.draw.circle(screen, dot, (rect.x + 14, rect.centery), 5)
        screen.blit(font.render(name, True, TEXT if on else TEXT_OFF),
                    (rect.x + 30, rect.centery - 8))

    screen.blit(small_font.render(f"ZOOM  {zoom:.1f}x", True, TEXT_DIM),
                (slider_rect.x, slider_rect.y - 22))
    pygame.draw.rect(screen, LINE_OFF, slider_rect, border_radius=3)
    t = (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN)
    knob_x = int(slider_rect.x + t * slider_rect.width)
    pygame.draw.circle(screen, (148, 163, 184), (knob_x, slider_rect.centery), 7)

    screen.blit(small_font.render("SPAWN", True, TEXT_DIM),
                (spawn_50_rect.x, spawn_50_rect.y - 20))
    draw_button(spawn_50_rect, "+50")
    draw_button(spawn_200_rect, "+200")
    draw_button(clear_rect, "Clear spawned")
    draw_button(reset_rect, "Reset simulation", strong=True)

    stats = [
        f"{len(bodies)} bodies      {clock.get_fps():.0f} fps",
        f"{dt / 3600:.1f} simulated hours per frame",
        "SPACE  pause   (" + ("paused" if paused else "running") + ")",
        "UP / DOWN  speed",
        "Click the black area to drop a body",
    ]
    ly = reset_rect.bottom + 24
    for line in stats:
        screen.blit(small_font.render(line, True, TEXT_DIM), (SIM_SIZE + PAD, ly))
        ly += 18


# ── Main loop ─────────────────────────────────────────────────────────────────
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            if slider_rect.inflate(16, 20).collidepoint(mx, my):
                dragging_slider = True
                zoom = zoom_from_mouse(mx)
                scale = BASE_SCALE * zoom

            elif spawn_50_rect.collidepoint(mx, my):
                extra_count = spawn_cluster(bodies, 50, scale, extra_count)

            elif spawn_200_rect.collidepoint(mx, my):
                extra_count = spawn_cluster(bodies, 200, scale, extra_count)

            elif clear_rect.collidepoint(mx, my):
                bodies = [b for b in bodies if b.name in SOLAR_SYSTEM]
                extra_count = 0
                screen.fill((0, 0, 0))

            elif reset_rect.collidepoint(mx, my):
                bodies = build_bodies(enabled)
                extra_count = 0
                screen.fill((0, 0, 0))          # wipe old trails

            elif PANEL_RECT.collidepoint(mx, my):
                for name, rect in toggle_rects.items():
                    if rect.collidepoint(mx, my):
                        if name in enabled:
                            enabled.remove(name)
                        else:
                            enabled.add(name)
                        bodies = build_bodies(enabled)
                        extra_count = 0
                        screen.fill((0, 0, 0))
                        break

            elif SIM_RECT.collidepoint(mx, my):
                extra_count += 1
                bodies.append(
                    Body(5e24, [0.0, 0.0], to_world(event.pos, scale), f"Extra{extra_count}")
                )

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_slider = False

        elif event.type == pygame.MOUSEMOTION and dragging_slider:
            zoom = zoom_from_mouse(event.pos[0])
            scale = BASE_SCALE * zoom

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                dt *= 1.5
            elif event.key == pygame.K_DOWN:
                dt /= 1.5
            elif event.key == pygame.K_SPACE:
                paused = not paused

    if not paused and bodies:
        step_verlet_bh(bodies, dt, THETA)

    # Simulation area: fade instead of clear, so trails linger
    screen.set_clip(SIM_RECT)
    screen.blit(fade, (0, 0))

    for b in bodies:
        info = SOLAR_SYSTEM.get(b.name)
        color = info["color"] if info else DEFAULT_COLOR
        size = info["size"] if info else DEFAULT_SIZE
        sx, sy = to_screen(b.position, scale)
        if abs(sx) < 1e6 and abs(sy) < 1e6:        # skip anything flung far off-screen
            pygame.draw.circle(screen, color, (int(sx), int(sy)), size)

    screen.set_clip(None)
    draw_panel()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
