import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "\tTrial time: %f seconds." %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)



def TicTocExperiment():
    # Generator that returns time differences
    ti = time.time()          # initial time
    tf = time.time() # final time
    while True:
        tf = time.time()
        yield tf-ti # returns the time difference

TicTocExp = TicTocExperiment() # create an instance of the TicTocGen generator

def toc_experiment(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicTocExp)
    if tempBool:
        print( "\tExperiment time: %f seconds." %tempTimeInterval )

def tic_experiment():
    # Records a time in TicToc, marks the beginning of a time interval
    toc_experiment(False)