from General_A03 import *
import A03

###############################################################################
# MAIN
###############################################################################

def main():    
    train(cell_finder=A03.CellFinder, cell_type=BCCD_TYPES.WBC)
       
if __name__ == "__main__": 
    main()
    