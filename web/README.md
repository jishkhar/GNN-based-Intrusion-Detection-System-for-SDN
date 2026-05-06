# GNN-Based IDS Dashboard

A comprehensive web-based dashboard for visualizing and analyzing the performance of a Graph Neural Network (GNN) trained for intrusion detection in Software-Defined Networks (SDN).

## 📋 Features

### 📊 **Six Main Sections**

1. **Dashboard Overview** — Key metrics at a glance
   - Best F1 Score, Test Accuracy, Precision, Recall
   - Model configuration (best epoch, threshold, batch size)
   - Training class distribution and class weights

2. **GNN Performance** — Detailed performance analysis
   - Test metrics comparison (bar charts)
   - Threshold-tuned (0.73) vs Argmax (0.5) comparison
   - Per-class performance (Normal vs Attack)
   - Detailed metrics table

3. **Training History** — Learning progression across 33 epochs
   - Training & Validation loss curves
   - Test accuracy progression
   - Test F1 score over epochs
   - Precision & Recall evolution
   - Best epoch statistics

4. **GNN vs Baselines** — Model comparison
   - Accuracy, F1 Score, ROC-AUC comparison charts
   - Comprehensive comparison table
   - Random Forest vs XGBoost vs GNN performance

5. **Detailed Analysis** — Per-class breakdown
   - Class distribution in test set
   - Prediction distribution
   - Per-class performance radar chart
   - Macro vs Weighted averages
   - Detailed per-class metrics table

6. **Project Information** — Architecture and dataset overview
   - Model details (Graph Attention Network)
   - Dataset information (CICIDS2017)
   - Feature descriptions

### 🎨 **User Interface Features**

- ✅ Responsive design (works on desktop, tablet, mobile)
- ✅ Professional gradient color scheme with Bootstrap 5
- ✅ Interactive charts with Chart.js (15+ visualizations)
- ✅ Smooth animations and transitions
- ✅ Inline explanations for technical terms
- ✅ Easy navigation with sticky top navbar
- ✅ Dark table headers for better readability
- ✅ Hover effects and visual feedback

### 🔧 **Technology Stack**

**Backend:**
- FastAPI 0.104.1 — Modern async Python web framework
- Uvicorn 0.24.0 — ASGI server
- Pydantic 2.5.0 — Data validation

**Frontend:**
- Bootstrap 5.3 — Responsive UI components
- Chart.js 4.4.0 — Interactive charts
- Vanilla JavaScript — No build step required

**Data Source:**
- JSON files loaded from `/results/` directory at runtime
- No database required

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with virtual environment activated
- GNN training metrics already generated (in `results/` directory)

### Installation & Execution

1. **Navigate to the web directory:**
   ```bash
   cd web
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   python -m uvicorn app:app --port 3000 --reload
   ```

   Or use the provided startup script:
   ```bash
   bash run.sh
   ```

4. **Open your browser:**
   ```
   http://localhost:3000
   ```

## 📁 File Structure

```
web/
├── app.py                      # FastAPI application with endpoints
├── config.py                   # Configuration and paths
├── requirements.txt            # Python dependencies
├── run.sh                      # Startup script
├── README.md                   # This file
└── static/
    ├── index.html             # Main HTML dashboard
    ├── css/
    │   └── style.css          # Custom styling
    └── js/
        └── app.js             # JavaScript logic and charts
```

## 🔗 API Endpoints

### Data Endpoints

- `GET /` — Serve the dashboard (index.html)
- `GET /api/metrics/gnn` — GNN training metrics and history
- `GET /api/metrics/baseline` — Baseline model metrics (Random Forest, XGBoost)
- `GET /api/classification-report` — Per-class classification metrics
- `GET /api/summary` — Quick summary of all metrics
- `GET /api/health` — Health check endpoint

### Interactive Features

- Browse sections using navbar navigation
- Click on metric cards for details
- Hover over charts for tooltips
- Responsive navigation on mobile devices
- Smooth scroll to sections

## 📊 Data Loading

The dashboard automatically loads data from these JSON files:

1. **`results/gnn_metrics.json`** — GNN training metrics
   - Best validation F1 and loss
   - Test set performance metrics
   - Training history per epoch
   - Class-wise results
   - Predictions breakdown

2. **`results/baseline_metrics.json`** — Baseline model results
   - Random Forest metrics
   - XGBoost metrics
   - Feature list

3. **`results/gnn_classification_report.json`** — Detailed classification metrics
   - Per-class: Precision, Recall, F1, Support
   - Macro and Weighted averages

## 🎯 Key Metrics Explained

### Core Metrics
- **Accuracy** — Overall correctness (% of correct predictions)
- **Precision** — Of detected attacks, how many were real attacks? (minimizes false alarms)
- **Recall** — Of all actual attacks, how many did we catch? (minimizes missed attacks)
- **F1 Score** — Harmonic mean of precision and recall (best for imbalanced data)
- **ROC-AUC** — Area under the Receiver Operating Characteristic curve

### Decision Threshold
- **Threshold-Tuned (0.73)** — Custom optimized threshold for IDS use case
- **Argmax (0.5)** — Standard probability threshold

### Class Distribution
- **Class 0** — Normal network traffic
- **Class 1** — Attack/Malicious traffic

## 🔬 Model Performance Highlights

- **Test Accuracy:** 99.82%
- **Test F1 Score:** 99.45%
- **Best Epoch:** 33 (convergence point)
- **Decision Threshold:** 0.73 (optimized for IDS)
- **ROC-AUC:** ~99.99%

## 🛠️ Development

### Adding Custom Charts

1. Add canvas element in `index.html`:
   ```html
   <canvas id="myChart"></canvas>
   ```

2. Create initialization function in `app.js`:
   ```javascript
   function initMyChart() {
       const ctx = document.getElementById('myChart').getContext('2d');
       charts.myChart = new Chart(ctx, {
           type: 'line',
           data: { /* ... */ },
           options: { /* ... */ }
       });
   }
   ```

3. Call in `initializeCharts()`:
   ```javascript
   initMyChart();
   ```

### Modifying Styling

- Global theme colors in `style.css` `:root` variables
- Component-specific styles in respective sections
- Responsive breakpoints at bottom of CSS file

## 📈 Performance Notes

- Charts render in ~100ms with 33 epochs of history
- All data loads in a single request
- Responsive to window resizing
- Optimized for modern browsers (Chrome, Firefox, Safari, Edge)

## 🐛 Troubleshooting

### Port Already in Use
```bash
lsof -i :3000  # Find process using port
kill -9 <PID>  # Kill the process
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Data Files Not Found
Ensure these files exist in `../results/`:
- `gnn_metrics.json`
- `baseline_metrics.json`
- `gnn_classification_report.json`

### Charts Not Rendering
1. Check browser console for errors (F12 → Console)
2. Verify API endpoints return data (F12 → Network tab)
3. Clear browser cache and reload

## 📝 Notes

- The dashboard is read-only (no database writes)
- All data is loaded at page load
- No external API calls (fully self-contained)
- Works offline once loaded
- Mobile-friendly design
- Fully responsive

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Chart.js Documentation](https://www.chartjs.org/)
- [Bootstrap 5 Documentation](https://getbootstrap.com/)
- [GNN Architecture Docs](../Docs/GNN_IDS_Architecture.md)

## 📄 License

Same as parent project (GNN-based Intrusion Detection System for SDN)

---

**Created:** May 2026  
**Last Updated:** May 6, 2026  
**Status:** ✅ Production Ready
