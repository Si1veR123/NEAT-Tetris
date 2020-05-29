import pygame, os, json, random, csv
pygame.init()


def remove_row(row_num):
    """
    :param row_num: number of row to remove
    removes a row
    """
    stopped_blocks.remove(stopped_blocks[row_num])
    stopped_blocks.append([4 for _ in range(GRID_WIDTH//SPACING-1)])


def find_stopped_blocks_pos(stopped_blocks):
    """
    :param stopped_blocks: list of stopped blocks
    :return: positions of blocks that have stopped
    """
    found_blocks = []
    for row_count, row in enumerate(stopped_blocks):
        for col_count, col in enumerate(row):
            if col != 4:
                found_blocks.append([col, (col_count+1, 15-row_count+1)])
    return found_blocks


def check_block_hit(blocks, others, direc='above'):
    """
    :param blocks: current block in play
    :param others: blocks that have stopped
    :param direc: direction to check if they are next to each other
    :return: True if found next to each other in specified direction, None if not
    """
    others_pos = [block[1] for block in find_stopped_blocks_pos(others)]
    for other_block in others_pos:
        for block in blocks:
            if direc == 'above':
                if block[1] + 1 == other_block[1] and block[0] == other_block[0]:
                    return True
            elif direc == 'left':
                if block[1] == other_block[1] and block[0] - 1 == other_block[0]:
                    return True
            elif direc == 'right':
                if block[1] == other_block[1] and block[0] + 1 == other_block[0]:
                    return True


def get_all_block_pos(block):
    """
    :param block: shape of block as 2D list
    :return: grid positions of all blocks
    """
    positions = []
    for row_count, row in enumerate(block):
        for col_count, col in enumerate(row):
            if col:
                block_x = col_count+current_pos[0]
                block_y = row_count+current_pos[1]
                positions.append((block_x, block_y))
    return positions


def return_pixel_coordinates(block_coords):
    """
    converts block coords to pixel coordinates
    """
    bx, by = block_coords
    return (bx*SPACING-SPACING+Y_PAD, by*SPACING-SPACING+X_TOP_PAD)


def rotate_matrix(matrix):
    """
    Takes a matrix e.g.
    [
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
    ]
    and rotates it e.g.
    [
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0]
    ]
    """
    all = []
    current = []

    for x in range(len(matrix)):
        for row in matrix:
            current.append(row[x])
        all.append(current)
        current = []

    for row in all:
        row.reverse()

    return all

# centre video
os.environ['SDL_VIDEO_CENTERED'] = '1'

# create window
root = pygame.display.set_mode((600, 900))

# open shapes json file
with open('shapes.json', 'r') as f:
    shapes = json.load(f)['shapes']

# load block images
green_cube = pygame.image.load('assets/green_cube.png')
blue_cube = pygame.image.load('assets/blue_cube.png')
red_cube = pygame.image.load('assets/red_cube.png')
blocks = [green_cube, blue_cube, red_cube]

cube_no_rotate = [
      [1, 1, 0],
      [1, 1, 0],
      [0, 0, 0]
    ]

score = 0
font = pygame.font.SysFont(pygame.font.get_default_font(), 30, True)

# create the first block
random.shuffle(shapes)
current = shapes[0]
current_pos = (6, 1)
current_block_pos = get_all_block_pos(current)
current_im = random.randint(0, 2)

# values to create board
SPACING = 40
GRID_HEIGHT = 680
GRID_WIDTH = 560
X_TOP_PAD = 10
Y_PAD = 40
x_axis_grid = [num+X_TOP_PAD for num in range(GRID_HEIGHT) if num % SPACING == 0]
y_axis_grid = [num+Y_PAD for num in range(GRID_WIDTH) if num % SPACING == 0]

# blocks fall at this speed
default_drop_speed = 600
block_drop_speed = default_drop_speed
fast_blocks_dropped = 0
BLOCK_FAST_SPEED = 100

# 2D list, where each list is one row. Each list is full of 4s, which get replaced for each block that lands.
stopped_blocks = [[4 for _ in range(GRID_WIDTH//SPACING-1)] for _ in range(GRID_HEIGHT//SPACING-1)]

# time leveller allows program to get time since last block dropped
time_leveller = 0
fast_time_leveller = 0

# max time for number of iterations per second
fps = 20
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(fps)

    # get time since last block dropped
    current_time = pygame.time.get_ticks() - time_leveller + fast_time_leveller

    # fill background black
    root.fill((0, 0, 0))

    for event in pygame.event.get():
        # quit when exit button pressed
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if not check_block_hit(current_block_pos, stopped_blocks, direc='left'):
                    """
                    If trying to go left, get leftmost block and check it isn't next to wall 
                    """
                    col1, col2, col3 = zip(current[0], current[1], current[2])
                    if any(col1):
                        leftmost_pos = (current_pos[0] - 1, current_pos[1])
                    elif any(col2) or any(col3):
                        leftmost_pos = current_pos
                    if leftmost_pos[0] > 0:
                        # move left
                        current_pos = (current_pos[0] - 1, current_pos[1])

            elif event.key == pygame.K_RIGHT:
                if not check_block_hit(current_block_pos, stopped_blocks, direc='right'):
                    """
                    If trying to go right, get rightmost block and check it isn't next to wall 
                    """

                    col1, col2, col3 = zip(current[0], current[1], current[2])
                    if any(col3):
                        rightmost_pos = (current_pos[0]+1, current_pos[1])
                    elif any(col2) or any(col1):
                        rightmost_pos = current_pos
                    if rightmost_pos[0] < 12:
                        # move right
                        current_pos = (current_pos[0]+1, current_pos[1])

            elif event.key == pygame.K_DOWN:
                block_drop_speed = BLOCK_FAST_SPEED
                fast_blocks_dropped = 0

            elif event.key == pygame.K_r:
                if not current == cube_no_rotate:
                    # rotate block with rotate_matrix function
                    current = rotate_matrix(current)
                    current_block_pos = get_all_block_pos(current)
                    for block_pos in current_block_pos:
                        if block_pos[0] <= 1:
                            current_pos = (current_pos[0]+1, current_pos[1])
                            break
                        elif block_pos[0] >= 12:
                            current_pos = (current_pos[0]-1, current_pos[1])
                            break
                        elif block_pos[1] > 16:
                            current_pos = (current_pos[0], current_pos[1]-1)
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                block_drop_speed = default_drop_speed
                fast_time_leveller += fast_blocks_dropped*default_drop_speed

    current_block_pos = get_all_block_pos(current)

    # Draw gridlines
    for line in x_axis_grid:
        pygame.draw.line(root, (255, 255, 255), (y_axis_grid[0], line), (y_axis_grid[-1], line))
    for line in y_axis_grid:
        pygame.draw.line(root, (255, 255, 255), (line, x_axis_grid[0]), (line, x_axis_grid[-1]))

    # draw all blocks at correct positions
    for row_count, row in enumerate(current):
        for col_count, space in enumerate(row):
            if space:
                block_coords = (current_pos[0]+col_count, current_pos[1]+row_count)
                root.blit(blocks[current_im], return_pixel_coordinates(block_coords))

    # draw landed blocks
    for im, stopped_block_pos in find_stopped_blocks_pos(stopped_blocks):
        root.blit(blocks[im], return_pixel_coordinates(stopped_block_pos))

    score_text = f'Score: {score}'
    score_surf = font.render(score_text, False, (0, 255, 0))
    root.blit(score_surf, (200, 700))

    # get lowest block
    for row_count, row in enumerate(current[::-1]):
        if any(row):
            lowest_pos = 2 - row_count
            lowest_block = current_pos[1] + lowest_pos + 2
            break

    # check for complete line
    for row_count, row in enumerate(stopped_blocks):
        for block in row:
            if block == 4:
                break
        else:
            remove_row(row_count)
            score += 1

    for col in stopped_blocks[-1]:
        if col != 4:
            running = False

    # drops block 1 row
    if current_time > block_drop_speed*current_pos[1]:
        current_pos = (current_pos[0], current_pos[1]+1)
        blocks_dropped = True
        if block_drop_speed == BLOCK_FAST_SPEED:
            fast_blocks_dropped += 1

    # creates new block if current touches bottom
    if lowest_block > GRID_HEIGHT/SPACING or check_block_hit(current_block_pos, stopped_blocks):
        for block in current_block_pos:
            """
            Replaces the 4 in the stopped_blocks 2D list with the current image number.
            16-block[1] is the row to change
            block[0]-1 is the position of the 4 to change
            """
            stopped_blocks[16-block[1]][block[0]-1] = current_im


        random.shuffle(shapes)
        current = shapes[0]
        current_pos = (6, 1)
        current_im = random.randint(0, 2)
        time_leveller += pygame.time.get_ticks() - time_leveller
        fast_blocks_dropped = 0
        fast_time_leveller = 0

    pygame.display.update()

pygame.quit()

if input(f'Score:{score}\nSave score? (y/n): ') == 'y':
    name = input('Name: ').lower()
    new = []

    read_scores = open('scores.csv', 'r')
    csv_data = list(csv.reader(read_scores))
    read_scores.close()

    new.append(['name', 'score'])

    name_found = False
    for line_count, line in enumerate(csv_data[1:]):
        if line:
            if line[0] == name:
                if int(line[1]) < score:
                    new.append([name, str(score)])
                else:
                    new.append(line)
                name_found = True
            else:
                new.append(line)
    if not name_found:
        new.append([name, str(score)])

    write_scores = open('scores.csv', 'w')
    csv_writer = csv.writer(write_scores)
    csv_writer.writerows(new)
    write_scores.close()
