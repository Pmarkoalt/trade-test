# Interactive Examples - Jupyter Notebooks

This directory contains Jupyter notebooks with interactive examples for the trading system.

## Notebooks

### 1. Getting Started (`01_Getting_Started.ipynb`)
Introduction to the trading system, key concepts, and configuration files.

**Topics:**
- System overview
- Configuration structure
- Key components

### 2. Data Loading and Exploration (`02_Data_Loading_and_Exploration.ipynb`)
Learn how to load and explore market data.

**Topics:**
- Loading data from CSV files
- Data validation
- Visualizing price and volume data
- Working with multiple symbols

### 3. Basic Backtest (`03_Basic_Backtest.ipynb`)
Run your first backtest and view results.

**Topics:**
- Running a backtest
- Viewing performance metrics
- Analyzing equity curves
- Comparing different periods

### 4. Strategy Configuration (`04_Strategy_Configuration.ipynb`)
Create and customize strategy configurations.

**Topics:**
- Loading strategy configs
- Modifying parameters
- Understanding entry/exit rules
- Risk management settings

### 5. Portfolio Analysis (`05_Portfolio_Analysis.ipynb`)
Deep dive into backtest results analysis.

**Topics:**
- Equity curve analysis
- Trade-by-trade breakdown
- Risk metrics
- Performance attribution

### 6. Validation Suite (`06_Validation_Suite.ipynb`)
Run statistical validation tests on your strategy.

**Topics:**
- Bootstrap testing
- Permutation testing
- Stress tests
- Sensitivity analysis

## Setup

### Prerequisites

1. Install Jupyter:
```bash
pip install jupyter notebook
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Install optional visualization dependencies:
```bash
pip install matplotlib seaborn plotly
```

### Running Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `examples/notebooks/` directory

3. Open notebooks in order (01 → 06) for a complete tutorial

Or use JupyterLab:
```bash
jupyter lab
```

## Usage Tips

- **Run cells in order**: Notebooks are designed to be run sequentially
- **Modify parameters**: Experiment with different configurations
- **Check data paths**: Ensure data files exist in the expected locations
- **Review outputs**: Pay attention to warnings and error messages

## Data Requirements

The notebooks use test data from `tests/fixtures/`. For production use:
- Update data paths in configurations
- Ensure data is in OHLCV format
- Verify date ranges match your data

## Troubleshooting

### Import Errors
- Ensure project root is in Python path
- Check that all dependencies are installed
- Verify you're running from the correct directory

### Data Not Found
- Check file paths in configurations
- Verify data files exist
- Use absolute paths if relative paths fail

### Memory Issues
- Reduce date ranges in configurations
- Process fewer symbols at once
- Close unused notebooks

## Next Steps

After completing these notebooks:
1. Read the main README.md for system overview
2. Review EXAMPLE_CONFIGS/ for configuration examples
3. Explore the test suite in tests/ for more examples
4. Check agent-files/ for detailed architecture documentation

## Contributing

To add new example notebooks:
1. Follow the naming convention: `NN_Topic_Name.ipynb`
2. Include markdown cells for explanations
3. Add setup cells at the beginning
4. Test notebooks with sample data
5. Update this README with notebook description

