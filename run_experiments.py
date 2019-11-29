from run_dataset_exp import run_dataset_exp

def main():
    run_dataset_exp("compass", "compass.logs", 5)
    #run_dataset_exp("bank", "bank.logs", 5)
    #run_dataset_exp("adult", "adult_census.logs", 5)
    # run_dataset_exp("kdd","kdd_census.logs",1)

if __name__ == '__main__':
    main()
