from datetime import datetime, timedelta
import sys

class progress_tracker():
    def __init__(self, num_of_steps, name='Progress', steps_taken=0):
        self.t_start = datetime.now()
        self.num_of_steps = num_of_steps
        self.steps_taken = steps_taken
        self.name = name
        self._printed_header = False
        self.last_string_length = 0
        
    
    @property
    def eta(self):
        now = datetime.now()
        return(int(round(float((now - self.t_start).seconds) / float(self.steps_taken) * float(self.num_of_steps - self.steps_taken))))
    
    @property
    def percentage(self):
        return(int(100 * float(self.steps_taken) / float(self.num_of_steps)))
    
    def get_print_string(self):
        curr_perc = self.percentage
        real_perc = self.percentage
        #Length of the progress bar is 25. Hence one step equates to 4%.
        bar_len = 25
        
        if not curr_perc % 4 == 0:
            curr_perc -= curr_perc % 4
        
        if int(curr_perc / 4) > 0:
            s = '[' + '=' * (int(curr_perc / 4) - 1) + '>' + '.' * (bar_len - int(curr_perc / 4)) + ']'
        else:
            s = '[' + '.' * bar_len + ']'
        
        tot_str = str(self.num_of_steps)
        curr_str = str(self.steps_taken)
        curr_str = ' ' * (len(tot_str) - len(curr_str)) + curr_str
        eta = str(timedelta(seconds=self.eta)) + 's'
        perc_str = ' ' * (len('100') - len(str(real_perc))) + str(real_perc)
        
        out_str = curr_str + '/' + tot_str + ': ' + s + ' ' + perc_str + '%' + ' ETA: ' + eta
        
        if self.last_string_length > len(out_str):
            back = '\b \b' * (self.last_string_length - len(out_str))
        else:
            back = ''
        
        self.last_string_length = len(out_str)
        
        return('\r' + back + out_str)
    
    def print_progress_bar(self, update=True):
        if not self._printed_header:
            print(self.name + ':')
            self._printed_header = True
        
        if update:
            sys.stdout.write('\r' + self.get_print_string())
            sys.stdout.flush()
            if self.steps_taken == self.num_of_steps:
                self.print_final(update=update)
        else:
            print(self.get_print_string())
            if self.steps_taken == self.num_of_steps:
                self.print_final(update=update)
    
    def iterate(self, iterate_by=1, print_prog_bar=True, update=True):
        self.steps_taken += iterate_by
        if print_prog_bar:
            self.print_progress_bar(update=update)
    
    def print_final(self, update=True):
        final_str = str(self.steps_taken) + '/' + str(self.num_of_steps) + ': [' + 25 * '=' + '] 100% - Time elapsed: ' + str(timedelta(seconds=(datetime.now() - self.t_start).seconds)) + 's'
        if update:
            clear_str = '\b \b' * self.last_string_length
            
            sys.stdout.write(clear_str + final_str + '\n')
            sys.stdout.flush()
        else:
            print(final_str)
