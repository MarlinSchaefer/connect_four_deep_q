from interface import trainer

def main():
    t = trainer(gamma=0.999, epsilon=0.2, epsilon_passive=0.1, learning_rate=0.2)
    
    for i in range(50):
        print("Epoch {}".format(i+1))
        t.train_vs_rational(training_cycles=5000)
    
    t.train_bots_mult_epochs(50, training_cycles=500, testing_cycles=100, random_ratio=0.5, change_act_pas=5)
    
    wanna_play = 'y'
    while wanna_play == 'y':
        t.human_vs_active()
        wanna_play = raw_input('Wanna play again?')
    
main()
