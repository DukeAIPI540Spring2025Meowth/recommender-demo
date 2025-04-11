from scripts.etl.etl import extract
from scripts.etl.etl import transform

def main():
    '''
    Entry point for the setup script.
    '''
    extract()
    transform()

if __name__ == "__main__":
    main()
