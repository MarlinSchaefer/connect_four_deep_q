import numpy as np

class normal_ai():
    def __init__(self, ai_type='rational', rows=6, cols=7, name='N/A'):
        self.possibilities = ['rational', 'random']
        if ai_type in self.possibilities:
            self.ai_type = ai_type
        else:
            self.ai_type = 'rational'
        self.rows = rows
        self.cols = cols
        self.name = name
    
    def get_move(self, game):
        if self.ai_type == 'rational':
            return(self.get_rational_move(game))
        elif self.ai_type == 'random':
            return(self.get_random_move(game))
    
    def get_rational_move(self, game):
        max_own_length = []
        for i in range(self.cols):
            g = game.copy()
            if not g.col_to_point(i) == None:
                r, c = g.col_to_point(i)
                g.make_move(i)
                max_own_length.append(g.check_length_at_pos(r, c))
            else:
                max_own_length.append(0)
        
        max_opponent_length = []
        for i in range(self.cols):
            g = game.copy()
            g.change_player()
            if not g.col_to_point(i) == None:
                r, c = g.col_to_point(i)
                g.make_move(i)
                max_opponent_length.append(g.check_length_at_pos(r, c))
            else:
                max_opponent_length.append(0)
        
        if max(max_own_length) > 3:
            return(np.argmax(max_own_length))
        
        if max(max_opponent_length) > 3:
            return(np.argmax(max_opponent_length))
        
        if max(max_own_length) == 3:
            return(np.argmax(max_own_length))
        
        if max(max_opponent_length) == 3:
            return(np.argmax(max_opponent_length))
        
        if max(max_own_length) == 2:
            return(np.argmax(max_own_length))
        
        if max(max_opponent_length) == 2:
            return(np.argmax(max_opponent_length))
        
        return(self.get_random_move(game))
    
    def get_random_move(self, game):
        c = list(range(self.cols))
        
        done = False
        
        while not done and len(c) > 0:
            a = np.random.choice(c)
            
            if game.valid_move(a):
                done = True
            else:
                c.remove(a)
        
        return(a)
    
    def change_nature(self, new_nature):
        if new_nature in self.possibilities:
            self.ai_type = new_nature
        else:
            raise ValueError('{} is not a supported AI Type.'.format(new_nature))
