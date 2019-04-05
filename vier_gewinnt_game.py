import numpy as np

class game():
    def __init__(self, rows=6, cols=7):
        self.board = np.zeros((rows, cols))
        self.player_turn = 1
        self.turn_count = 0
        self.height = np.zeros(7, dtype=np.int)
        self.rows = rows
        self.cols = cols
    
    def make_move(self, col):
        if not (0 <= col and col <= self.cols):
            err_msg = 'The board has no column {}.'.format(col)
            #print(err_msg)
            return(False)
            raise ValueError(err_msg)
        
        if self.height[col] >= self.rows:
            err_msg = 'Column {} is full.\nHeight array:{}\nNumber of rows: {}'.format(col,self.height, self.rows)
            #print(err_msg)
            return(False)
            raise ValueError(err_msg)
        
        if not any([h < self.rows for h in self.height]):
            return(True)
        
        self.board[self.rows-self.height[col]-1][col] = self.player_turn
        
        self.turn_count += 1
        
        if self.check_win_condition(self.rows-self.height[col]-1, col):
            #print("Congratulation Player {}, you won the game.".format(self.player_turn))
            return(True)
        
        self.height[col] += 1
        
        self.change_player()
        
        return(False)
    
    def check_win_condition(self, row, col):
        if self.check_length_at_pos(row, col) >= 4:
            return(True)
        else:
            return(False)
    
    def point_on_board(self, r, c):
        return((r in range(self.rows)) and (c in range(self.cols)))
    
    def change_player(self):
        if self.player_turn == 1:
            self.player_turn = 2
        else:
            self.player_turn = 1
    
    def print_board(self):
        print(self.board)
    
    def make_move_and_print(self, col):
        won = self.make_move(col)
        if won:
            print("Player {} wins the game!".format(self.player_turn))
        self.print_board()
    
    def reset(self):
        self.__init__()
        
    def copy(self):
        ret = game(rows=self.rows, cols=self.cols)
        ret.board = self.board.copy()
        ret.player_turn = self.player_turn
        ret.turn_count = self.turn_count
        ret.height = self.height.copy()
        
        return(ret)
    
    def get_inverted_player_game(self):
        ret = self.copy()
        for i in range(ret.rows):
            for j in range(ret.cols):
                if ret.board[i][j] == 1:
                    ret.board[i][j] = 2
                elif ret.board[i][j] == 2:
                    ret.board[i][j] = 1
        return(ret)
    
    def get_inverse_player_board(self):
        return(self.get_inverted_player_game().board)
    
    def get_board(self):
        ret = self.board.copy()
        return(ret)
    
    def get_turn_num(self):
        return(self.turn_count)
    
    def valid_move(self, col):
        if not col in range(self.cols):
            return(False)
        
        if not self.height[col] < self.rows:
            return(False)
        
        return(True)
    
    def check_length_at_pos(self, row, col):
        ll_tr = [[[-1, -1], [-2, -2], [-3, -3]], [[1, 1], [2, 2], [3, 3]]]
        lr_tl = [[[1, -1], [2, -2], [3, -3]], [[-1, 1], [-2, 2], [-3, 3]]]
        b_t = [[[0, 1], [0, 2], [0, 3]], [[0, -1], [0, -2], [0, -3]]]
        l_r = [[[1, 0], [2, 0], [3, 0]], [[-1, 0], [-2, 0], [-3, 0]]]
        
        run = [ll_tr, lr_tl, b_t, l_r]
        
        lengths = []
        
        if not self.point_on_board(row, col):
            return(0)
        
        if not (self.board[row][col] == 1 or self.board[row][col] == 2):
            return(0)
        
        for l in run:
            count = 1
            for p in l:
                for pt in p:
                    nr = row + pt[0]
                    nc = col + pt[1]
                    valid = self.point_on_board(nr, nc)
                    valid = valid and self.board[nr][nc] == self.board[row][col]
                    if valid:
                        count += 1
                    else:
                        break
            
            lengths.append(count)
            
        return(max(lengths))
    
    def col_to_point(self, col):
        if not self.valid_move(col):
            return(None)
        
        return((self.rows-self.height[col]-1, col))
        
        
