# InsightForge — Data Analytics Studio

A cloud-deployable Streamlit application for exploring CSV and Excel datasets, generating data-driven KPIs and charts, profiling data quality, applying interactive filters, saving filter bookmarks, joining related tables, and exporting filtered data and reports.

## What is included

- Upload multiple `.csv`, `.xlsx`, and legacy `.xls` files. Each Excel sheet is loaded as a separate table.
- Preview and infer date/numeric fields, flag suspicious headers, and inspect column-level quality statistics.
- Keep imported tables intact while applying optional cleaning to an analysis copy.
- Automatically create KPIs, trends, category breakdowns, distributions, and evidence-based descriptive insights.
- Use categorical, numeric-range, and date-range slicers; selectable overview charts can cross-filter other visuals where supported.
- Save, restore, and delete named dashboard views in the active browser session.
- Inspect likely relationships across uploaded tables and create a separate derived joined table.
- Explore detailed records and build custom bar, stacked bar, line, area, scatter, histogram, box, pie, and correlation charts.
- Export filtered data as CSV/Excel, export charts as interactive HTML, and generate a PDF report.

## Run locally

Recommended: Python 3.11.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, the `.gitignore` file, and the `.streamlit/config.toml` folder.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect the repository.
4. Choose the branch and set **Main file path** to `app.py`.
5. In advanced settings, select Python 3.11 if that option is available, then deploy.
6. Open the app URL and test a small CSV and a multi-sheet Excel workbook before using larger files.

Cloud services have their own quotas, resource limits, and terms. Large files and joins can use substantial memory; start with small datasets. The configured per-file upload limit is 200 MB, but practical limits may be lower on your host.

## Privacy and calculation notes

- The app does **not** call an external AI model or transmit dataset contents to an AI provider. Insights are rule-based and derived from the current data view.
- Uploaded data is held in the Streamlit session's memory; it is not intentionally persisted to a database or written to disk by the application. Restarting the app/session can clear it. Hosting-provider infrastructure and policies still apply.
- Optional cleaning actions affect an analysis copy only. The imported table is retained separately in the current session.
- Relationship suggestions are heuristic. Verify key meaning and uniqueness before joining; joins can multiply rows when keys are duplicated.
- Date parsing can be ambiguous across regional formats. Review inferred types and date ranges before drawing conclusions.
- The tool describes associations and changes; it does not infer causation or guarantee that the source data is correct.

## Project structure

```text
insightforge_analytics/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Troubleshooting

- **Legacy `.xls` error:** Ensure `xlrd` installed successfully (`pip install xlrd>=2.0`).
- **Excel `.xlsx` error:** Check that the file is not password protected or corrupt.
- **Wrong headers:** Open the Data Quality tab and switch between “Treat first row as data” and “Restore first row as header”.
- **Slow chart or join:** Try a smaller filtered dataset or remove unused columns before uploading. The detailed table preview is paginated.
- **PDF export error:** Confirm `reportlab` and `matplotlib` installed; other app functionality can still work without the PDF dependencies.

## License

Use and modify this starter project for your own learning and analytics workflows.
