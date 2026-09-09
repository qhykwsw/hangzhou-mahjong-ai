"""Offline smoke run; uses genuine Mahjong and both trainable policy families."""
from .cli import main

if __name__ == '__main__':
    main(['iterate', '--run-id', 'demo', '--generator', 'search', '--rounds', '4'])
