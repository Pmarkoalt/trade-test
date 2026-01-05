#!/bin/bash
# Interactive script for debugging integration tests in Docker
# Usage: ./scripts/debug_integration_tests.sh [test_name]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed or not in PATH${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}✗ docker-compose is not installed or not in PATH${NC}"
    exit 1
fi

# Function to show menu
show_menu() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Docker Integration Test Debugging Menu${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "1. Build Docker image"
    echo "2. Run all integration tests"
    echo "3. Run all integration tests (verbose)"
    echo "4. Run all integration tests (with output capture)"
    echo "5. Run specific test (interactive)"
    echo "6. Run specific test (with pdb debugger)"
    echo "7. Open interactive shell in container"
    echo "8. Re-run only failed tests"
    echo "9. List available integration tests"
    echo "0. Exit"
    echo ""
    read -p "Select option [0-9]: " choice
}

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker-compose build
    echo -e "${GREEN}✓ Docker image built${NC}"
}

# Function to run all integration tests
run_all_tests() {
    echo -e "${YELLOW}Running all integration tests...${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -v \
        --tb=short
}

# Function to run all integration tests (verbose)
run_all_tests_verbose() {
    echo -e "${YELLOW}Running all integration tests (verbose)...${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -vvv \
        --tb=long \
        --showlocals \
        --log-cli-level=DEBUG
}

# Function to run all integration tests with output capture
run_all_tests_capture() {
    echo -e "${YELLOW}Running all integration tests with output capture...${NC}"
    mkdir -p results
    OUTPUT_FILE="results/integration_test_output_$(date +%Y%m%d_%H%M%S).txt"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -vvv \
        --tb=long \
        --showlocals \
        2>&1 | tee "$OUTPUT_FILE"
    echo -e "${GREEN}✓ Output saved to $OUTPUT_FILE${NC}"
}

# Function to run specific test
run_specific_test() {
    read -p "Enter test name (e.g., test_integration_workflow): " test_name
    if [ -z "$test_name" ]; then
        echo -e "${RED}✗ Test name cannot be empty${NC}"
        return
    fi
    echo -e "${YELLOW}Running test: $test_name${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -vvv \
        --tb=long \
        --showlocals \
        -k "$test_name"
}

# Function to run specific test with pdb
run_specific_test_pdb() {
    read -p "Enter test name (e.g., test_integration_workflow): " test_name
    if [ -z "$test_name" ]; then
        echo -e "${RED}✗ Test name cannot be empty${NC}"
        return
    fi
    echo -e "${YELLOW}Running test: $test_name (with pdb debugger)${NC}"
    echo -e "${CYAN}Note: The test will pause at breakpoints or failures${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -vvv \
        --tb=long \
        --showlocals \
        --capture=no \
        --pdb \
        -k "$test_name"
}

# Function to open interactive shell
open_shell() {
    echo -e "${YELLOW}Opening interactive shell...${NC}"
    echo -e "${CYAN}You can run pytest commands manually from here${NC}"
    echo -e "${CYAN}Example: pytest tests/integration/ -v -k test_name${NC}"
    docker-compose run --rm trading-system /bin/bash
}

# Function to re-run failed tests
rerun_failed() {
    echo -e "${YELLOW}Re-running only failed tests...${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -v \
        --tb=short \
        --lf \
        --failed-first
}

# Function to list available tests
list_tests() {
    echo -e "${YELLOW}Listing available integration tests...${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        --collect-only \
        -q
}

# Main execution
if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        show_menu
        case $choice in
            1)
                build_image
                ;;
            2)
                run_all_tests
                ;;
            3)
                run_all_tests_verbose
                ;;
            4)
                run_all_tests_capture
                ;;
            5)
                run_specific_test
                ;;
            6)
                run_specific_test_pdb
                ;;
            7)
                open_shell
                ;;
            8)
                rerun_failed
                ;;
            9)
                list_tests
                ;;
            0)
                echo -e "${GREEN}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please try again.${NC}"
                ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Command-line mode: run specific test
    TEST_NAME="$1"
    echo -e "${YELLOW}Running test: $TEST_NAME${NC}"
    docker-compose run --rm --entrypoint pytest trading-system tests/integration/ \
        -vvv \
        --tb=long \
        --showlocals \
        -k "$TEST_NAME"
fi
