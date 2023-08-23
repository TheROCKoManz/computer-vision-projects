
import sys
from utils.setup_files import pre_setup

def run(source, filepath):
    # basic startup setup
    pre_setup()

    from CountWalks import main
    # main(source=source, filepath=filepath)


if __name__ == '__main__':
    source = sys.argv[1]
    filepath = sys.argv[2] if source == 'file' else ''
    run(source=source,filepath=filepath)