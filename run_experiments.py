from run_dataset_exp import run_dataset_exp

def main():
    # run_dataset_exp("compass", "compass.logs", 5) # done! paper result were replicated! :)
    # run_dataset_exp("bank", "bank.logs", 5) # acc was replicated, pRule was 0.07 lower than in the paper!
    # run_dataset_exp("adult", "adult_census.logs", 5) # should work, needs testing, will probably take more than 25 hours
    run_dataset_exp("kdd","kdd_census.logs",1)
    # run_dataset_exp("dutch", "dutch.logs", 1)
    print('We did it! (╯°□°）╯︵ ┻━┻')

if __name__ == '__main__':
    main()