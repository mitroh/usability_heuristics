import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Racing Mountain")

# Colors
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 200, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)

# Car properties
car_width = 50
car_height = 100
car_x = WIDTH // 2 - car_width // 2
car_y = HEIGHT - car_height - 20
car_speed = 5

# Obstacle properties
obstacle_width = 50
obstacle_height = 100
obstacle_speed = 7
obstacles = []

# Mountain properties
mountain_points = [(0, HEIGHT), (WIDTH // 4, HEIGHT // 3), (WIDTH // 2, HEIGHT // 5),
                   (3 * WIDTH // 4, HEIGHT // 3), (WIDTH, HEIGHT)]

# Game variables
score = 0
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

def draw_car(x, y):
    pygame.draw.rect(screen, RED, (x, y, car_width, car_height))

def draw_obstacle(x, y):
    pygame.draw.rect(screen, GRAY, (x, y, obstacle_width, obstacle_height))

def draw_mountain():
    pygame.draw.polygon(screen, GREEN, mountain_points)
    pygame.draw.polygon(screen, BROWN, [(0, HEIGHT), (WIDTH // 4, HEIGHT // 3 + 20), (WIDTH // 2, HEIGHT // 5 + 20), (3 * WIDTH // 4, HEIGHT // 3 + 20), (WIDTH, HEIGHT)])

def generate_obstacle():
    x = random.randint(0, WIDTH - obstacle_width)
    y = -obstacle_height
    obstacles.append([x, y])

def check_collision(car_x, car_y, obstacle_x, obstacle_y):
    if car_x < obstacle_x + obstacle_width and \
       car_x + car_width > obstacle_x and \
       car_y < obstacle_y + obstacle_height and \
       car_y + car_height > obstacle_y:
        return True
    return False

# Game loop
running = True
obstacle_timer = pygame.time.get_ticks()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and car_x > 0:
        car_x -= car_speed
    if keys[pygame.K_RIGHT] and car_x < WIDTH - car_width:
        car_x += car_speed

    # Generate obstacles
    current_time = pygame.time.get_ticks()
    if current_time - obstacle_timer > 1500:  # Generate every 1.5 seconds
        generate_obstacle()
        obstacle_timer = current_time

    # Move obstacles
    for obstacle in obstacles:
        obstacle[1] += obstacle_speed

    # Remove off-screen obstacles
    obstacles = [obstacle for obstacle in obstacles if obstacle[1] < HEIGHT]

    # Check for collisions
    for obstacle in obstacles:
        if check_collision(car_x, car_y, obstacle[0], obstacle[1]):
            running = False  # Game over

    # Update score
    score += 1

    # Draw everything
    screen.fill(WHITE)
    draw_mountain()
    draw_car(car_x, car_y)
    for obstacle in obstacles:
        draw_obstacle(obstacle[0], obstacle[1])

    # Display score
    score_text = font.render("Score: " + str(score), True, GRAY)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()