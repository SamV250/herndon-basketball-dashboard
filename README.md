# 🏀 Herndon Basketball Dashboard

An interactive **Streamlit dashboard** for the Herndon Basketball program — built to visualize, analyze, and explore team and player data.  
This app automatically adapts to **any CSV or Excel file**: it can handle rosters, scouting reports, and full statistical datasets without hardcoded column names.

---

## 📊 Features

- **Automatic data detection**  
  Upload any CSV/XLSX — the app figures out which columns are numeric, categorical, or text.

- **Roster / Scouting View**  
  Displays each player or row as an expandable “card” with strengths, weaknesses, and notes.

- **Statistical Analysis**  
  Generates descriptive stats, correlation heatmaps, and quick K-Means clustering if numeric data is present.

- **Shot Chart Visualization**  
  If your file contains `x`, `y`, and `result` columns, the app automatically plots makes, misses, and shot density.

- **Four-Factor Summary (if applicable)**  
  Recognizes basketball-style data (`EFG_O`, `TOR`, `ORB`, `FTR`, etc.) and renders an instant radar chart.

- **Quick JSON Report**  
  Exports an auto-generated summary of your uploaded dataset.

---

## 🧠 Example Use Cases

- Upload a **player scouting sheet** to visualize strengths/weaknesses  
- Upload a **team stats file** to run analytics & see four-factor metrics  
- Upload **shot-tracking data** to instantly see shot charts and heatmaps  

---

## ⚙️ How to Run Locally

1. **Clone this repo**
   ```bash
   git clone https://github.com/SamV250/herndon-basketball-dashboard.git
   cd herndon-basketball-dashboard
