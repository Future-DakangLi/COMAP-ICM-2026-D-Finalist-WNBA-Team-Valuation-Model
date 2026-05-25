# Data Collection Notes

`src/step00_collect.py` contains public-source URLs and can try to download them when `ONLINE_COLLECTION=1`.

Some sources are easy to download as CSV, such as GitHub-hosted schedule files and FRED interest-rate data. Other sources may require extra work:

- SeatGeek ticket data may require API credentials.
- Salary websites may block automated scraping or change their HTML table structure.
- Basketball Reference tables may require polite rate limits.

For this reason, the repository includes `data/00_seed/` as an offline raw/source-like snapshot so that the pipeline can still run on any machine.
