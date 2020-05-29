import pygame, os, json, random, neat, pickle
from typing import List
from itertools import chain

pygame.init()


def remove_row(row_num, stopped_blocks, GRID_WIDTH, SPACING):
    stopped_blocks.remove(stopped_blocks[row_num])
    stopped_blocks.append([4 for _ in range(GRID_WIDTH//SPACING-1)])
    return stopped_blocks


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


def get_all_block_pos(block, current_pos):
    """
    :param block: shape of block as 2D list
    :param current_pos: current position of top left block
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


def return_pixel_coordinates(block_coords, SPACING, Y_PAD, X_TOP_PAD):
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

class Agent:
    def __init__(self, GRID_WIDTH, GRID_HEIGHT, SPACING):
        # 2D list, where each list is one row. Each list is full of 4s, which get replaced for each block that lands.
        self.stopped_blocks = [[4 for _ in range(GRID_WIDTH // SPACING - 1)] for _ in range(GRID_HEIGHT // SPACING - 1)]

        # create the first block
        self.current = random.choice(shapes)
        self.current_pos = (6, 1)
        self.current_block_pos = get_all_block_pos(self.current, self.current_pos)
        self.current_im = random.randint(0, 2)

        self.score = 0

        self.current_time = 0

        self.lowest_block = 0

        # time leveller allows program to get time since last block dropped
        self.time_leveller = 0

        self.block_drop_speed = 1

        self.moved = False

    def eval_placement(self):
        score = 0
        highest_block = self.highest_cube()
        lowest_block = self.lowest_cube()

        if highest_block:
            for block in self.current_block_pos:
                if 16-block[1] == highest_block:
                    score -= 15
                    break

        for block in self.current_block_pos:
            if 16 - block[1] == lowest_block:
                score += 15
                break

        return score

        """
        lowest_of_col = []
        col_nums = max([x[0] for x in self.current_block_pos])

        for n in range(col_nums):
            n += 1
            lowest = (0, 0)
            for b in self.current_block_pos:
                if b[0] == n and b[1] > lowest[1]:
                    lowest = b
            lowest_of_col.append(lowest)

        for block in self.current_block_pos:
            row = 16-block[1] + 1
            if row != 16:
                if block in lowest_of_col and self.stopped_blocks[row][block[0]-1] == 4:
                    score -= 20
                    break
        """

    def process(self, ticks, GRID_WIDTH, GRID_HEIGHT, SPACING) -> int:
        score_increased = 0

        # get time
        self.current_time = ticks - self.time_leveller

        # get all block positions
        self.current_block_pos = get_all_block_pos(self.current, self.current_pos)

        # get lowest block
        for row_count, row in enumerate(self.current[::-1]):
            if any(row):
                lowest_pos = 2 - row_count
                self.lowest_block = self.current_pos[1] + lowest_pos + 2
                break

        # check for complete line
        for row_count, row in enumerate(self.stopped_blocks):
            for block in row:
                if block == 4:
                    break
            else:
                self.stopped_blocks = remove_row(row_count, self.stopped_blocks, GRID_WIDTH, SPACING)
                self.score += 1
                score_increased += 150
                print('row!')

        # drops block 1 row
        if self.current_time > self.block_drop_speed * self.current_pos[1]:
            self.current_pos = (self.current_pos[0], self.current_pos[1] + 1)

        # creates new block if current touches bottom
        if self.lowest_block > GRID_HEIGHT / SPACING or check_block_hit(self.current_block_pos, self.stopped_blocks):
            score_increased += self.eval_placement()

            for block in self.current_block_pos:
                """
                Replaces the 4 in the stopped_blocks 2D list with the current image number.
                16-block[1] is the row to change
                block[0]-1 is the position of the 4 to change
                """
                self.stopped_blocks[16 - block[1]][block[0] - 1] = self.current_im

            self.moved = False
            self.current = random.choice(shapes)
            self.current_pos = (6, 1)
            self.current_im = random.randint(0, 2)
            self.time_leveller += ticks - self.time_leveller
            self.current_block_pos = get_all_block_pos(self.current, self.current_pos)

        return score_increased

    def activate_nn(self, network: neat.nn.FeedForwardNetwork):
        player_block_input = list(chain.from_iterable(self.current))

        highest_cube = self.highest_cube_pos()
        lowest_cube = self.lowest_cube_pos()

        output = network.activate((highest_cube, lowest_cube))

        cubes_per_col = 13
        window = 2/13

        for x in range(cubes_per_col):
            if window*x < output[0] < window*(x+1):
                self.move(x+1)
                self.moved = True

        """
        if output[1] > 0 and self.current != cube_no_rotate:

            self.current = rotate_matrix(self.current)
            self.current_block_pos = get_all_block_pos(self.current, self.current_pos)
            for block_pos in self.current_block_pos:
                if block_pos[0] <= 1:
                    self.current_pos = (self.current_pos[0] + 1, self.current_pos[1])
                    break
                elif block_pos[0] >= 12:
                    self.current_pos = (self.current_pos[0] - 1, self.current_pos[1])
                    break
                elif block_pos[1] > 16:
                    self.current_pos = (self.current_pos[0], self.current_pos[1] - 1)
        """

    def move(self, col):
        self.current_pos = (col, self.current_pos[1])

    def highest_cube(self):
        highest_cols = []
        for cols in zip(*self.stopped_blocks):
            for count, col in enumerate(cols[::-1]):
                if col != 4:
                    highest_cols.append(len(cols) - count)
                    break
            else:
                highest_cols.append(0)
        return max(highest_cols)

    def highest_cube_pos(self):
        highest_cols = []
        for col_count, cols in enumerate(zip(*self.stopped_blocks)):
            for count, col in enumerate(cols[::-1]):
                if col != 4:
                    highest_cols.append(len(cols) - count)
                    break
            else:
                highest_cols.append(0)
        return highest_cols.index(max(highest_cols))

    def lowest_cube(self):
        lowest_cols = []
        for cols in zip(*self.stopped_blocks):
            for count, col in enumerate(cols[::-1]):
                if col != 4:
                    lowest_cols.append(len(cols) - count)
                    break
            else:
                lowest_cols.append(0)
        return min(lowest_cols)

    def lowest_cube_pos(self):
        lowest_cols = []
        for cols in zip(*self.stopped_blocks):
            for count, col in enumerate(cols[::-1]):
                if col != 4:
                    lowest_cols.append(len(cols) - count)
                    break
            else:
                lowest_cols.append(0)
        return lowest_cols.index(min(lowest_cols))


# centre video
os.environ['SDL_VIDEO_CENTERED'] = '1'

# create window
root = pygame.display.set_mode((600, 900))

# open shapes json file
with open('../shapes.json', 'r') as f:
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

font = pygame.font.SysFont(pygame.font.get_default_font(), 30, True)


def main_game(genomes: List[neat.DefaultGenome], config: neat.Config):
    # values to create board
    SPACING = 40
    GRID_HEIGHT = 680
    GRID_WIDTH = 560
    X_TOP_PAD = 10
    Y_PAD = 40
    x_axis_grid = [num + X_TOP_PAD for num in range(GRID_HEIGHT) if num % SPACING == 0]
    y_axis_grid = [num + Y_PAD for num in range(GRID_WIDTH) if num % SPACING == 0]

    nets = []
    agent = []
    gens = []

    for _, g in genomes:
        g.fitness = 0
        net = neat.nn.feed_forward.FeedForwardNetwork.create(g, config)
        nets.append(net)
        agent.append(Agent(GRID_WIDTH, GRID_HEIGHT, SPACING))
        gens.append(g)

    # max time for number of iterations per second
    fps = 20
    clock = pygame.time.Clock()

    best_agent = random.choice(agent)
    prev_best_switch = 0
    max_best_switch = 4000


    running = True

    while running and len(agent) > 0:

        clock.tick(fps)

        best = (-1000, None)
        for count, a in enumerate(agent):
            score_increased = a.process(pygame.time.get_ticks(), GRID_WIDTH, GRID_HEIGHT, SPACING)
            if score_increased:
                gens[count].fitness += score_increased
            if gens[count].fitness > best[0]:
                best = (gens[count].fitness, a)

        if best[1] and pygame.time.get_ticks() - prev_best_switch > max_best_switch:
            best = best[1]
            best_agent = best
            prev_best_switch = pygame.time.get_ticks()
        else:
            best = best_agent

        # fill background black
        root.fill((0, 0, 0))

        for event in pygame.event.get():
            # quit when exit button pressed
            if event.type == pygame.QUIT:
                with open('best_genome.pickle', 'wb') as file:
                    pickle.dump(best, file)
                running = False

        # Draw gridlines
        for line in x_axis_grid:
            pygame.draw.line(root, (100, 100, 127), (y_axis_grid[0], line), (y_axis_grid[-1], line))
        for line in y_axis_grid:
            pygame.draw.line(root, (100, 100, 127), (line, x_axis_grid[0]), (line, x_axis_grid[-1]))

        # draw all blocks at correct positions
        for row_count, row in enumerate(best.current):
            for col_count, space in enumerate(row):
                if space:
                    block_coords = (best.current_pos[0]+col_count, best.current_pos[1]+row_count)
                    stopped_blocks_x, stopped_blocks_y = return_pixel_coordinates(block_coords, SPACING, Y_PAD, X_TOP_PAD)
                    root.blit(blocks[best.current_im], (stopped_blocks_x+1, stopped_blocks_y))

        # draw landed blocks
        for im, stopped_block_pos in find_stopped_blocks_pos(best.stopped_blocks):
            root.blit(blocks[im], return_pixel_coordinates(stopped_block_pos, SPACING, Y_PAD, X_TOP_PAD))

        try:
            score_text = f'Score: {best.score}          Fitness: {gens[agent.index(best)].fitness}'
        except ValueError:
            score_text = f'Score: {best.score}'

        score_surf = font.render(score_text, False, (0, 255, 0))
        root.blit(score_surf, (200, 700))

        # get input layer data for neural network
        for x, a in enumerate(agent):
            if not a.moved:
                a.activate_nn(nets[x])

        nets_remove = []
        agent_remove = []
        gens_remove = []

        for a in agent:
            for col in a.stopped_blocks[-1]:
                if col != 4:
                    gens[agent.index(a)].fitness -= 1
                    nets_remove.append(nets[agent.index(a)])
                    gens_remove.append(gens[agent.index(a)])
                    agent_remove.append(agent[agent.index(a)])
                    break

        for x in range(len(agent_remove)):
            agent.remove(agent_remove[x])
            nets.remove(nets_remove[x])
            gens.remove(gens_remove[x])

        pygame.display.update()


def neat_run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(main_game, 30000)

    with open('best_genome.pickle', 'wb') as file:
        pickle.dump(winner, file)


if __name__ == '__main__':
    neat_run()
