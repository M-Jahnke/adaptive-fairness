from run_dataset_exp import run_dataset_exp

def main():
    run_dataset_exp("compass", "compass.logs", 5)           # works, paper result were replicated!
    run_dataset_exp("bank", "bank.logs", 5)                 # works, pRule was 0.04 lower than in the paper!
    # run_dataset_exp("adult", "adult_census.logs", 5)      # needs testing (with less rows)
    # run_dataset_exp("kdd","kdd_census.logs",1)            # needs testing (with less rows)
    # run_dataset_exp("dutch", "dutch.logs", 1)             # needs testing (with less rows)
    # run_dataset_exp("synth_opp", "opp.logs", 5, True)     # works, replicated for avoid disparate treatment!
    # run_dataset_exp("synth_same", "same.logs", 5, True)   # works, replicated for avoid disparate treatment!
    run_dataset_exp("synth_opp", "opp_dt.logs", 5, False)   # works, acc is 0.05 lower than in the paper!
    run_dataset_exp("synth_same", "same_dt.logs", 5, False) # works, acc was replicated!
    print('We did it! (╯°□°）╯︵ ┻━┻')

if __name__ == '__main__':
    main()