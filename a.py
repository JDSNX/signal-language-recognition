import os
import sys


def notepad():
    os.startfile('notepad')

def spotify():
    os.startfile('spotify')

if __name__ == "__main__":
    args = sys.argv

    if args[1] == 'notepad':
        notepad()
    elif args[1] == 'spotify':
        spotify()