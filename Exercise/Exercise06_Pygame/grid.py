import pygame, sys
import random

# Define Game parameters
# screen size
screen_width = 1200
screen_height = 780
# grid selection
xSel = 0
ySel = 0
#Boolean for showing cells
ShowCell = False

# initialise the game window
pygame.init()
pygame.display.set_caption('first game')

# set the game surface
screen = pygame.display.set_mode((screen_width, screen_height))

# a clock to keep track of the game progress
clock = pygame.time.Clock()

# Any commands that draw the initial state of the game
screen.fill(pygame.Color('red'))
pygame.draw.rect(screen, (255,255,255), pygame.Rect(100,100,300,200))

# Use existing images
backgrImage = pygame.image.load('population_density.jpg')
backgrImage = pygame.transform.scale(backgrImage, (screen_width, screen_height))
screen.blit(backgrImage, [0,0])



# Update before the first frame
pygame.display.update()

# Any objects you might need/use
rectangles = []
for i in range(10):
    row = []
    for j in range(10):
        r = pygame.Rect(i * (screen_width/10), j * (screen_height/7),
                        (screen_width/10), (screen_height/7))
        row.append(r)
    rectangles.append(row)

# The game loop, in here the behaviour of the game is defined.
# This loop is executed every frame.
while True:
    screen.blit(backgrImage, [0,0])

    # Get key events to check if something is going on through some form of input.
    events = pygame.event.get()
    for event in events:
        # Check exit through x button window
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Go through all key presses, which you can use to controll the game.
        elif event.type == pygame.KEYDOWN:
            # Check exit through esc
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_RIGHT:
                if xSel < 9:
                    xSel +=1
            if event.key == pygame.K_LEFT:
                if xSel > 0:
                    xSel -=1
            if event.key == pygame.K_UP:
                if ySel > 0:
                    ySel -= 1
            if event.key == pygame.K_DOWN:
                if ySel < 9:
                    ySel += 1
            if event.key == pygame.K_RETURN:
                if ShowCell:
                    ShowCell = False
                else:
                    ShowCell = True


    # define any drawings
    pygame.draw.rect(screen, pygame.Color(100,100,100), rectangles[xSel][ySel])

    if ShowCell:
        try:
            cellImage = pygame.image.load('GridCells/' + str(xSel) + '_' + str(ySel) + '.png')
            imWidth, imHeight = cellImage.get_size()
            dx = screen_width / imWidth
            dy = screen_height / imHeight
            f = min(dx,dy)
            cellImage = pygame.transform.scale(cellImage, (int(f * imWidth), int(f * imHeight)))
            screen.blit(cellImage, [0,0])
        except FileNotFoundError:
            ShowCell = False

    # At the end of the loop update the screen and game time.
    pygame.display.update()
    clock.tick()
