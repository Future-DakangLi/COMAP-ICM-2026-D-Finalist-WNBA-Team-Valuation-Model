"""Run the full no-answer-leak workflow from raw inputs to figures/results."""
from . import step00_collect, step01_clean, step02_explore, step03_elo, step04_attendance_pricing, step05_finance_valuation, step06_optimization, step07_scenarios, step08_monte_carlo, step09_sensitivity, step10_business_plots

def main():
    step00_collect.main()
    step01_clean.main()
    step02_explore.main()
    step03_elo.main()
    step04_attendance_pricing.main()
    step05_finance_valuation.main()
    step06_optimization.main()
    step07_scenarios.main()
    step08_monte_carlo.main()
    step09_sensitivity.main()
    step10_business_plots.main()
    print('\nDone. This workflow does not read any final-output file; all outputs are recomputed from staged inputs and assumptions.');

if __name__ == '__main__':
    main()
