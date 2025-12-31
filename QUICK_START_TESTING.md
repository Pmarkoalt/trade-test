# Quick Start: Testing Your Trading System

## 🚀 Fast Track (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Test

```bash
./quick_test.sh
```

This verifies:
- ✅ Python version
- ✅ Dependencies installed
- ✅ Test data available
- ✅ Module imports work
- ✅ Basic unit test passes

### 3. Run All Tests

```bash
# Unit tests
pytest tests/ -v

# Integration test
pytest tests/integration/ -v
```

### 4. Run a Backtest

```bash
python -m trading_system backtest \
    --config tests/fixtures/configs/run_test_config.yaml \
    --period train
```

Check results in: `tests/results/*/train/`

## 📋 What Gets Tested

### Unit Tests
- Data loading and validation
- Indicator calculations (MA, ATR, momentum)
- Strategy signal generation
- Portfolio management
- Execution and fills
- Risk management

### Integration Tests
- End-to-end workflow
- No lookahead bias
- Portfolio operations
- Trade execution

### CLI Commands
- Full backtest runs
- Validation suite
- Holdout evaluation

## 🎯 Success Criteria

✅ All unit tests pass  
✅ Integration test passes  
✅ Backtest completes without errors  
✅ Output files generated  
✅ Portfolio equity updates correctly  

## 📚 Full Documentation

See `TESTING_GUIDE.md` for comprehensive testing instructions.

## 🐛 Troubleshooting

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**No trades generated?**
- Normal with 3-month test data
- System needs 20+ days for signals
- Use longer date range for more trades

