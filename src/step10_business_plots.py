"""Step 10: final business plots."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from .paths import RESULTS, FIGURES, ensure_dirs

def main():
    ensure_dirs()
    base = pd.read_csv(RESULTS/'02_business_outputs'/'baseline_finance_summary.csv').iloc[0]
    comps = pd.read_csv(RESULTS/'02_business_outputs'/'decision_comparison_table.csv')

    # Finance waterfall-like bar chart.
    labels = ['Ticket rev','Other rev','Salary','Operating cost','Interest','Net profit']
    values = [base['ticket_revenue_m'], base['other_revenue_m'], -base['salary_m'], -base['operating_cost_m'], -base['interest_m'], base['net_profit_m']]
    plt.figure(figsize=(8,4.8))
    plt.bar(labels, values)
    plt.axhline(0, linewidth=0.8)
    plt.title('Baseline financial bridge')
    plt.ylabel('Million USD')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES/'03_business'/'baseline_financial_waterfall.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7,4.5))
    plt.bar(comps['action'], comps['net_profit_m'])
    plt.title('Decision comparison: net profit')
    plt.ylabel('Million USD')
    plt.tight_layout()
    plt.savefig(FIGURES/'03_business'/'decision_profit_comparison.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7,4.5))
    plt.bar(comps['action'], comps['valuation_m'])
    plt.title('Decision comparison: valuation')
    plt.ylabel('Million USD')
    plt.tight_layout()
    plt.savefig(FIGURES/'03_business'/'decision_valuation_comparison.png', dpi=180)
    plt.close()
    print('[business-plots] wrote final plots')

if __name__ == '__main__':
    main()
