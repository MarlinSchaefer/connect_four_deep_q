import numpy as np
from vier_gewinnt_game import game
from vier_gewinnt_ai import vr_ai
from normal_ai import normal_ai
from progress_bar import progress_tracker

class trainer():
    def __init__(self, rows=6, cols=7, learning_rate=0.1, gamma=0.99, epsilon=0.5, epsilon_passive=0.0):
        self.game = game(rows=rows, cols=cols)
        self.p1 = vr_ai(rows=rows, cols=cols, name='Player 1')
        self.p2 = vr_ai(rows=rows, cols=cols, name='Player 2')
        self.active = 1
        self.passive = 2
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_passive = epsilon_passive
        self.rational_bot = normal_ai(ai_type='rational', rows=rows, cols=cols, name='Rational Bot')
        self.random_bot = normal_ai(ai_type='random', rows=rows, cols=cols, name='Random Bot')
    
    def get_active_bot(self):
        if self.active == 1:
            #print("Returning active bot {}".format(self.p1.name))
            return(self.p1)
        else:
            #print("Returning active bot {}".format(self.p2.name))
            return(self.p2)
        
    def get_passive_bot(self):
        if self.passive == 1:
            return(self.p1)
        elif self.passive == 2:
            return(self.p2)
        elif self.passive == 3:
            return(self.rational_bot)
        elif self.passive == 4:
            return(self.random_bot)
        
    def swap_active(self):
        if self.active == 1:
            self.active = 2
            self.passive = 1
        else:
            self.active = 1
            self.passive = 2
            
    def reward_move(self, col):
        """Returns rewards for a move in column col. Also makes the move in
        that column and uses the passive bot to make a bot move.
        
        Arguments
        ---------
        col : int
            Column to make a move in.
        
        Return
        ------
        reward : int
        done : bool
        next_state : np.array
        """
        
        reward_valid_move = 1
        
        reward = reward_valid_move
        
        if not self.game.valid_move(col):
            next_state, reward, done = self.game.board, -10, True
            return(next_state, reward, done)
        
        #Make move and check if that wins the game
        done = self.game.make_move(col)
        
        if done:
            next_state, reward = self.game.board, 20
        
        #Check if the opponent can win with his next move
        opp_win = self.check_next_move()
        
        if opp_win:
            reward = -50
        
        if not done:
            if np.random.random() < self.epsilon_passive:
                done = self.game.make_move(np.random.randint(0, self.game.cols))
            else:
                done = self.game.make_move(self.get_passive_bot_move(self.game))
        
        next_state = self.game.board
        
        if reward == reward_valid_move and done:
            reward = -30
        
        return(next_state, reward, done)
    
    def check_next_move(self):
        for i in range(self.game.cols):
            g = self.game.copy()
            if g.make_move(i):
                return(True)
        return(False)
    
    def get_passive_bot_move(self, g):
        bot = self.get_passive_bot()
        
        if self.passive == 2 or self.passive == 1:
            state = g.get_inverse_player_board()
        else:
            state = g
        
        return(bot.get_move(state))
    
    def train_bots(self, training_cycles=100, testing_cycles=100, change_act_pas=100, train_only_active=False, random_ratio=0.1):
        self.swap_active()
        for i in range(training_cycles * (1 if train_only_active else 2)):
            self.game.reset()
            if i % change_act_pas == 0:
                self.swap_active()
                active = self.get_active_bot()
                passive = self.get_passive_bot()
            if i % 20 == 0:
                print("Steps taken: {} | Currently active: {} | Currently passive: {}".format(i, active.name, passive.name))
            
            done = False
            
            while not done:
                old_state = self.game.get_board()
                
                #print("Got old")
            
                curr_q = active.get_q_values(self.game.board)
                
                #print("Current q: {}".format(curr_q))
                
                #print("Got curr_q")
                
                a = np.argmax(curr_q)
                q_val = np.max(curr_q)
                
                #print("Set necessray values")
                
                if np.random.random() < self.epsilon:
                    #print("Went into if")
                    a = np.random.randint(0, self.game.cols)
                
                #print("After if")
                
                next_state, r, done = self.reward_move(a)
                
                #print("Made move if possible")
                
                target_vec = active.get_q_values(self.game.board)
                
                #print("Calculating next q")
                
                target = (1 - self.learning_rate) * curr_q[a] + self.learning_rate * (r + self.gamma * np.max(target_vec))
                
                #print("Setting vector")
                
                curr_q[a] = target
                
                #print("Making training step")
                
                active.train_step(old_state, curr_q)
                
                #print("Done status: {}\ntarget Q-Val".format(done, target))
            
        
        final = self.compare_active_passive(testing_cycles)
        
        if final[0] > final[1]:
            self.p1.set_weights(self.p2.get_weights())
            #self.p2.randomize_weights(ratio=random_ratio)
        else:
            self.p2.set_weights(self.p1.get_weights())
            #self.p1.randomize_weights(ratio=random_ratio)
        
        print(final)
        
        print("Bot 1 against random bot: {}".format(self.compare_against_random()))
        
        print("Bot 1 against rational bot: {}".format(self.compare_against_rational()))
    
    def set_active_p1(self):
        self.active = 1
    
    def set_active_p2(self):
        self.active = 2
    
    def get_active_num(self):
        return(self.active)
                
    def play_game(self):
        self.game.reset()
        
        start_player = np.random.randint(1,3)
        
        if start_player == 1:
            self.set_active_p2()
        else:
            self.set_active_p1()
        
        done = False
        
        while not done:
            self.swap_active()
            curr_bot = self.get_active_bot()
            
            if self.game.get_turn_num() % 2 == 0:
                state = self.game.get_inverse_player_board()
            else:
                state = self.game.get_board()
                
            bot_move = curr_bot.get_move(state)
            
            if self.game.valid_move(bot_move):
                done = self.game.make_move(bot_move)
            else:
                self.swap_active()
                done = True
        
        return(self.get_active_num())
    
    def compare_active_passive(self, testing_games):
        l = [self.play_game() for i in range(testing_games)]
        
        return([l.count(1), l.count(2)])
    
    def compare_against_random(self, num_of_games=100):
        player_one_won = 0
        bar = progress_tracker(num_of_games, name='Playing against random')
        
        for i in range(num_of_games):
            #print("From compare: {}".format(i))
            done = False
            self.game.reset()
            player = i
            while not done:
                if player % 2 == 0:
                    if i % 2 == 0:
                        a = self.p1.get_move(self.game.get_board())
                        #a = self.rational_bot.get_move(self.game)
                    else:
                        a = self.p1.get_move(self.game.get_inverse_player_board())
                        #a = self.rational_bot.get_move(self.game)
                    #print("Rational")
                else:
                    a = self.random_bot.get_move(self.game)
                    #print("Random")
                
                done = self.game.make_move(a)
                #self.game.print_board()
                
                player += 1
            
            #print("i: {} | player: {}".format(i, player))
            if player % 2 == 1:
                player_one_won += 1
            
            bar.iterate()
        
        print("From compare: {}".format(float(player_one_won) / float(num_of_games)))
    
        return(float(player_one_won) / float(num_of_games))
    
    def train_bots_mult_epochs(self, epochs, training_cycles=100, testing_cycles=100, change_act_pas=100, train_only_active=False, random_ratio=0.1):
        for i in range(epochs):
            print("Epoch {}/{}".format(i, epochs))
            self.train_bots(training_cycles=training_cycles, testing_cycles=testing_cycles, change_act_pas=change_act_pas, train_only_active=train_only_active, random_ratio=random_ratio)
    
    def human_vs_active(self):
        bot = self.get_active_bot()
        self.game = game()
        
        done = False
        
        player = 0
        
        print("Game columns: {}".format(self.game.cols))
        print("Game rows: {}".format(self.game.rows))
        
        while not done:
            if player % 2 == 0:
                self.game.print_board()
                a = raw_input('Please input the column number [from 0 to {}]'.format(self.game.cols-1))
                print("Your input: {}".format(a))
                done = self.game.make_move(int(a))
                self.game.print_board()
            else:
                a = bot.get_move(self.game.get_inverse_player_board())
                print("Bot chose column {}".format(a))
                if not self.game.valid_move(a):
                    done = True
                    print("Bot made an invalid move so you win.")
                    player += 1
                else:
                    done = self.game.make_move(a)
            
            player += 1
                    
        
        if player % 2 == 1:
            print("The human has won against the machine!")
        else:
            print("The machine has won against the human!")
    
    def human_vs_rational(self):
        bot = self.rational_bot
        self.game.reset()
        
        done = False
        
        player = 0
        
        while not done:
            if player % 2 == 0:
                self.game.print_board()
                a = raw_input('Please input the column number [from 0 to {}]'.format(self.game.cols-1))
                print("Your input: {}".format(a))
                done = self.game.make_move(int(a))
                self.game.print_board()
            else:
                a = bot.get_move(self.game)
                print("Bot chose column {}".format(a))
                if not self.game.valid_move(a):
                    done = True
                    print("Bot made an invalid move so you win.")
                    player += 1
                else:
                    done = self.game.make_move(a)
            
            player += 1
        
        if player % 2 == 1:
            print("The human has won against the machine!")
        else:
            self.game.print_board()
            print("The machine has won against the human!")
    
    def train_vs_rational(self, training_cycles=100):
        old_passive = self.passive
        self.passive = 3
        active = self.get_active_bot()
        for i in range(training_cycles):
            self.game.reset()
            
            if i % 20 == 0:
                print("Steps taken: {}".format(i))
            
            done = False
            
            while not done:
                old_state = self.game.get_board()
                
                #print("Got old")
            
                curr_q = active.get_q_values(self.game.board)
                
                #print("Current q: {}".format(curr_q))
                
                #print("Got curr_q")
                
                a = np.argmax(curr_q)
                q_val = np.max(curr_q)
                
                #print("Set necessray values")
                
                if np.random.random() < self.epsilon:
                    #print("Went into if")
                    a = np.random.randint(0, self.game.cols)
                
                #print("After if")
                
                next_state, r, done = self.reward_move(a)
                
                #print("Made move if possible")
                
                target_vec = active.get_q_values(self.game.board)
                
                #print("Calculating next q")
                
                target = (1 - self.learning_rate) * curr_q[a] + self.learning_rate * (r + self.gamma * np.max(target_vec))
                
                #print("Setting vector")
                
                curr_q[a] = target
                
                #print("Making training step")
                
                active.train_step(old_state, curr_q)
                
                #print("Done status: {}\ntarget Q-Val".format(done, target))
        
        print("Bot 1 against random bot: {}".format(self.compare_against_random()))
        
        print("Bot 1 against rational bot: {}".format(self.compare_against_rational()))
        
        self.passive = old_passive
    
    def compare_against_rational(self, num_of_games=100):
        bot = self.p1
        rational = self.rational_bot
        
        bot_win = 0
        bar = progress_tracker(num_of_games, name='Playing against rational bot')
        
        for i in range(num_of_games):
            self.game.reset()
            done = False
            player = i
            
            while not done:
                if player % 2 == 0:
                    if i % 2 == 0:
                        a = bot.get_move(self.game.get_board())
                    else:
                        a = bot.get_move(self.game.get_inverse_player_board())
                else:
                    a = rational.get_move(self.game)
                
                done = self.game.make_move(a)
                
                player += 1
            
            if player % 2 == 1:
                bot_win += 1
            
            bar.iterate()
            
        print("From compare: {}".format(float(bot_win) / float(num_of_games)))
    
        return(float(bot_win) / float(num_of_games))
