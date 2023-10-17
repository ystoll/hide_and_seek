import random

import argparse
import cv2
import numpy as np

import utils.colors as colors
from hidegame.entity import Entity
from hidegame.maps import DEFAULT_MAP, EMPTY, MAPS, WALL
from utils.vector2 import Vector2


class Game:
    """
    Game class. Contains the grid, the player and the agent.

    """

    def __init__(self, mode=None, map_name=DEFAULT_MAP, level=-1) -> None:
        """
        Initialize the game

        Parameters
        ----------
        mode : str, optional
            "human" to play it and render it, else None (for AI training),
            by default None
        map_name : str, optional
            name of the map to load, by default "statement", the map that is in the pdf
            statement. See maps.py for the list of available maps.
            If "random", a random map is generated. See generate_random_map() for more.
        """

        self.map_name = map_name
        self.grid = self._load_map(map_name, level=level)  # contains the map

        self.GRID_W = len(self.grid[0])
        self.GRID_H = len(self.grid)

        self.SPEED = 12
        self.CELL_SIZE = 32
        self.WIDTH = self.GRID_W * self.CELL_SIZE
        self.HEIGHT = self.GRID_H * self.CELL_SIZE

        self.player = Entity(Vector2(0, 0), colors.RED)
        self.agent = Entity(Vector2(1, 1), colors.BLUE)

        self.nb_walls = 0
        self.wall_positions = []
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self._is_wall(Vector2(x, y)):
                    self.wall_positions.append(Vector2(x, y))
                    self.nb_walls += 1

        # "human" to play it and render it, else None (for AI training)
        self.mode = mode

        if self.mode == "human":
            self.init_game_start()

    def _load_map(self, map_name: str, level: int = -1) -> list:
        """
        Load the map from maps.py.
        If map_name == "random", generate a random map instead.

        Parameters
        ----------
        map_name : str
            name of the map to load, or "random" to generate a random map
        level : int, default -1
            level of the map loaded if map_name == "random".

        Returns
        -------
        list
            the map as a list of lists
        """
        if map_name != "random":
            if map_name in MAPS:
                return MAPS[map_name]
            else:
                raise ValueError(
                    f"Map '{map_name}' does not exist. Please choose one"
                    + f" of {MAPS.keys()}"
                )
        else:
            return self.generate_random_map(width=12, height=12, level=level)

    def generate_random_map(self, width=12, height=12, level=-1) -> list:
        """
        Generates a map with a specified number of wall packs.

        This function creates a grid map with `level` wall packs symmetrically distributed.
        The grid size is defined by `width` and `height`. Wall packs are non-overlapping and
        are separated by at least one empty cell from other packs and the grid boundaries.
        The centers of these wall packs are pre-defined based on the number of packs specified,
        ensuring symmetric distribution.

        Parameters
        ----------
        width : int, optional
            The width of the grid map. Default is 12.
        height : int, optional
            The height of the grid map. Default is 12.
        level : int
            Number of wall packs to be placed on the grid. Must be between 1 and 5 inclusive.

        Returns
        -------
        list of str
            The generated grid map. Each string in the list represents one row of the grid.
            Empty cells are represented by '.' and wall cells by '#'.

        Raises
        ------
        ValueError
            If `level` is not between 1 and 5 inclusive.

        Examples
        --------
        >>> for row in generate_random_map_n_packs(level=3):
        >>>     print(row)
        This will print a 12x12 grid with 3 wall packs symmetrically distributed.
        """
        if level < 1 or level > 5:
            raise ValueError("level should be between 1 and 5 inclusive")

        grid = [[EMPTY for _ in range(width)] for _ in range(height)]

        centers = {
            1: [(6, 6)],
            2: [(3, 6), (9, 6)],
            3: [(3, 3), (3, 9), (9, 6)],
            4: [(3, 3), (3, 9), (9, 3), (9, 9)],
            5: [(6, 6), (3, 3), (3, 9), (9, 3), (9, 9)],
        }

        walls_per_pack_values = {1: 15, 2: 12, 3: 10, 4: 8, 5: 5}

        boundaries = {
            1: [((4, 4), (8, 8))],
            2: [((1, 4), (4, 8)), ((7, 4), (9, 8))],
            3: [((1, 1), (4, 4)), ((1, 7), (4, 10)), ((7, 4), (10, 8))],
            4: [
                ((1, 1), (4, 4)),
                ((1, 7), (4, 10)),
                ((7, 2), (10, 4)),
                ((7, 7), (10, 10)),
            ],
            5: [
                ((5, 5), (7, 7)),
                ((2, 2), (4, 4)),
                ((2, 7), (4, 10)),
                ((7, 2), (10, 4)),
                ((7, 7), (10, 10)),
            ],
        }

        walls_per_pack = walls_per_pack_values[level]

        for center, boundary in zip(centers[level], boundaries[level]):
            cx, cy = center
            (start_x, start_y), (end_x, end_y) = boundary

            placed_walls = 1
            grid[cy][cx] = WALL
            potential_placements = {
                (cx + dx, cy + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            } - {(cx, cy)}

            while placed_walls < walls_per_pack:
                if not potential_placements:
                    break

                new_x, new_y = random.choice(list(potential_placements))
                potential_placements.remove((new_x, new_y))

                if (
                    start_x <= new_x <= end_x
                    and start_y <= new_y <= end_y
                    and grid[new_y][new_x] == EMPTY
                ):
                    grid[new_y][new_x] = WALL
                    placed_walls += 1

                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        adj_x, adj_y = new_x + dx, new_y + dy
                        if (adj_x, adj_y) not in potential_placements and grid[adj_y][
                            adj_x
                        ] == EMPTY:
                            potential_placements.add((adj_x, adj_y))

        return grid

    def init_game_start(self) -> None:
        """
        Initialize the game state (player and agent positions)
        Place player and agent at random but the player must see the agent
        """

        while True:
            self._place_entity_at_random(self.player)
            self._place_entity_at_random(self.agent)

            # No collision between player and agent
            if self.player.pos == self.agent.pos:
                continue

            if self.player.can_see(self.agent, self.grid):
                break

        self.agent.is_seen = True

    def _cell_to_pixel(self, cell: int, center=False) -> int:
        """
        Given a cell index, return the top-left pixel position of the cell.
        If center is True, return the center of the cell instead of the top-left corner.

        Parameters
        ----------
        cell : int
            cell index
        center : bool, optional
            if True, return the center of the cell instead of the top-left corner,
            by default False

        Returns
        -------
        int
            top-left pixel position of the cell
        """

        offset = self.CELL_SIZE // 2 if center else 0
        return cell * self.CELL_SIZE + offset

    def _draw_grid(self, board, color=colors.BLACK, thickness: int = 1) -> None:
        """
        Draw a grid on the display board.

        Parameters
        ----------
        board : np.ndarray
            board to draw on
        color : tuple, optional, by default colors.BLACK, from colors.py
            color of the grid
        thickness : int, optional, by default 1
            thickness of the grid lines
        """
        width, height, _ = board.shape

        # draw vertical lines
        for cell_x in range(self.GRID_W):
            pixel_x = self._cell_to_pixel(cell_x)
            cv2.line(
                board, (pixel_x, 0), (pixel_x, height), color=color, thickness=thickness
            )

        # draw horizontal lines
        for cell_y in range(self.GRID_H):
            pixel_y = self._cell_to_pixel(cell_y)
            cv2.line(
                board, (0, pixel_y), (width, pixel_y), color=color, thickness=thickness
            )

    def _fill_cell(self, board, cell_x, cell_y, color) -> None:
        """
        Fill a cell with a color on the display board.

        Parameters
        ----------
        board : np.ndarray
            board to draw on
        cell_x : int
            x index of the cell
        cell_y : int
            y index of the cell
        color : tuple, from colors.py
            color to fill the cell with
        """

        board[
            (cell_y * self.CELL_SIZE): ((cell_y + 1) * self.CELL_SIZE),
            (cell_x * self.CELL_SIZE): ((cell_x + 1) * self.CELL_SIZE),
        ] = color

    def _place_entity_at_random(self, entity: Entity) -> None:
        """
        Place an entity at random on the grid, avoiding walls.

        Parameters
        ----------
        entity : Entity
            entity to place on the grid
        """

        while True:
            x = random.randint(0, self.GRID_W - 1)
            y = random.randint(0, self.GRID_H - 1)
            if self.grid[y][x] == EMPTY:
                entity.pos = Vector2(x, y)
                break

    def _draw_line(
        self,
        board,
        start_cell_x,
        start_cell_y,
        target_cell_x,
        target_cell_y,
        color,
        thickness: int = 1,
    ) -> None:
        """
        Draw a line between two cells on the display board.

        Parameters
        ----------
        board : np.ndarray
            board to draw on
        start_cell_x : int
            x index of the start cell
        start_cell_y : int
            y index of the start cell
        target_cell_x : int
            x index of the target cell
        target_cell_y : int
            y index of the target cell
        color : tuple, from colors.py
            color of the line
        thickness : int, optional, by default 1
            thickness of the line
        """

        start_pixel_x = self._cell_to_pixel(start_cell_x, center=True)
        start_pixel_y = self._cell_to_pixel(start_cell_y, center=True)
        target_pixel_x = self._cell_to_pixel(target_cell_x, center=True)
        target_pixel_y = self._cell_to_pixel(target_cell_y, center=True)
        cv2.line(
            board,
            (start_pixel_x, start_pixel_y),
            (target_pixel_x, target_pixel_y),
            color=color,
            thickness=thickness,
        )

    def render(self) -> np.ndarray:
        """
        Render the game state on a display board, and returns it.

        Returns
        -------
        np.ndarray
            display board
        """
        board = np.zeros((self.WIDTH, self.HEIGHT, 3)) + 255  # white background

        # render walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self._is_wall(Vector2(x, y)):
                    self._fill_cell(board, x, y, colors.BLACK)

        # render player and agent
        self.player.draw(board)
        self.agent.draw(board)

        self._draw_grid(board)

        # Render line between player and agent
        # If the agent is seen, the line is green, otherwise it is red
        see_color = colors.GREEN if self.agent.is_seen else colors.RED

        self._draw_line(
            board,
            self.player.x,
            self.player.y,
            self.agent.x,
            self.agent.y,
            see_color,
            thickness=3,
        )
        # Display only if in human mode
        if self.mode == "human":
            cv2.imshow("Hide and Seek", board)

        return board

    def handle_inputs(self) -> bool:
        """
        Handle keyboard inputs in human mode.
        ZQSD to move the player, IJKL to move the agent.
        x, delete or escape to exit the game.
        """
        key = cv2.waitKey(int(1000 / self.SPEED))

        # delete and escape keys to exit the game
        if key in [8, 27, ord("x")]:
            return True

        new_pos = Vector2(self.player.pos.x, self.player.pos.y)
        if key == ord("q"):  # player left
            new_pos.x -= 1
        elif key == ord("d"):  # player right
            new_pos.x += 1
        elif key == ord("z"):  # player up
            new_pos.y -= 1
        elif key == ord("s"):  # player down
            new_pos.y += 1
        if key == ord("i"):  # agent up
            self.handle_action(2)
        elif key == ord("j"):  # agent left
            self.handle_action(0)
        elif key == ord("k"):  # agent down
            self.handle_action(3)
        elif key == ord("l"):  # agent right
            self.handle_action(1)

        # Make the move only if the new position is valid
        if (
            self._is_valid_coordinates(new_pos)
            and not self._is_wall(new_pos)
            and new_pos != self.agent.pos
        ):
            self.player.pos = new_pos

        if self.player.can_see(self.agent, self.grid):
            self.agent.is_seen = True
        else:
            self.agent.is_seen = False

        return False

    def _is_valid_coordinates(self, coord: Vector2) -> bool:
        """
        Check if the coordinates are valid, i.e. inside the grid.

        Parameters
        ----------
        coord : Vector2
            coordinates to check

        Returns
        -------
        bool
            True if the coordinates are valid, False otherwise
        """
        return (
            coord.x >= 0
            and coord.x < self.GRID_W
            and coord.y >= 0
            and coord.y < self.GRID_H
        )

    def _is_wall(self, coord: Vector2) -> bool:
        """
        Check if the coordinates are a wall.

        Parameters
        ----------
        coord : Vector2
            coordinates to check

        Returns
        -------
        bool
            True if the coordinates are a wall, False otherwise
        """
        return self.grid[coord.y][coord.x] == WALL

    def handle_action(self, action) -> None:
        """
        Handle the action of the agent.
        The agent cannot move through walls and cannot move outside the grid.
        The action are:
            0: move left
            1: move right
            2: move up
            3: move down
        """
        new_pos = Vector2(self.agent.pos.x, self.agent.pos.y)
        if action == 0:
            new_pos.x -= 1
        elif action == 1:
            new_pos.x += 1
        elif action == 2:
            new_pos.y -= 1
        elif action == 3:
            new_pos.y += 1

        if (
            self._is_valid_coordinates(new_pos)
            and not self._is_wall(new_pos)
            and new_pos != self.player.pos
        ):
            self.agent.pos = new_pos

        if self.player.can_see(self.agent, self.grid):
            self.agent.is_seen = True
        else:
            self.agent.is_seen = False

    def run(self) -> None:
        """
        Run the game.
        """
        print("Run")
        while True:
            self.render()

            quit = self.handle_inputs()
            if quit:
                break

        cv2.destroyAllWindows()

    def display_map(self, map_name: str = "random", level: int = 4) -> None:
        """
        Display a map of level 'level' on screen.
        Parameters
        ----------
        map_name : str
            name of the map to display
        level : int, default -1
            level of the map displayed if map_name == "random".

        """
        self.grid = self._load_map(map_name, level)
        board = np.zeros((self.WIDTH, self.HEIGHT, 3)) + 255  # white background

        # render walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self._is_wall(Vector2(x, y)):
                    self._fill_cell(board, x, y, colors.BLACK)
        self._draw_grid(board)
        cv2.imshow("Hide and Seek", board)
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "Play the game in human mode."
        ),
    )
    parser.add_argument(
        "--map",
        type=str,
        default="random",
        help=(
            "Map on which the grid will be trained. Default: random"
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help=(
            "Level of the displayed map, if map is random. Default: None"
        ),
    )

    args = parser.parse_args()
    if (args.map != "random" and args.level):
        raise ValueError("A level must be set only if map='random'")

    if args.map and not args.play:
        game = Game(mode="human")
        game.display_map(args.map, args.level)
    else:
        game = Game(mode="human", map_name=args.map, level=args.level)
        game.run()
