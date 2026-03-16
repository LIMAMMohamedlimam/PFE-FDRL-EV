import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# ================= CONFIGURATION =================
# Set the directory containing the input CSV files
INPUT_DIR = "./data/iso_new_england_hourly_wholesale_load_cost/" 
# Set the name of the final output file
OUTPUT_FILE = "./iso_ne_prices.csv"
# =================================================

REQUIRED_COLUMNS = ["Local Date", "Local Hour", "Total Cost"]

def find_header_row(reader):
    """
    Scans the CSV rows to find the Header row (starts with 'H') 
    that contains the actual column names (e.g., 'Total Cost'), 
    ignoring the row that contains data types (e.g., 'Number').
    """
    for row in reader:
        if not row:
            continue
        # Check if row starts with 'H' (after stripping whitespace/quotes handled by csv)
        if row[0].strip() == "H":
            # Check if this row contains one of our required column names
            # This distinguishes the Name row from the Type row
            if any(col in row for col in REQUIRED_COLUMNS):
                return row
    return None

def validate_file(filepath):
    """
    Checks if a file has the required structure and columns.
    Returns True if valid, raises ValueError if invalid.
    """
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header_row = find_header_row(reader)
            
            if header_row is None:
                raise ValueError("Could not find valid Header row (starting with 'H' containing column names).")
            
            # Clean headers (remove whitespace)
            clean_headers = [h.strip() for h in header_row]
            
            # Check for required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in clean_headers]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
        return True
    except Exception as e:
        raise ValueError(f"Validation failed for {filepath.name}: {str(e)}")

def process_file(filepath):
    """
    Extracts timestamp and price from a single file.
    Returns a list of tuples: (timestamp_str, price_str)
    """
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header_row = find_header_row(reader)
        
        # Map column names to indices
        clean_headers = [h.strip() for h in header_row]
        idx_date = clean_headers.index("Local Date")
        idx_hour = clean_headers.index("Local Hour")
        idx_cost = clean_headers.index("Total Cost")
        
        for row in reader:
            if not row or row[0].strip() != "D":
                continue
            
            try:
                raw_date = row[idx_date].strip()
                raw_hour = row[idx_hour].strip()
                price = row[idx_cost].strip()
                
                # Parse Date: Input "09/01/2021" -> DateTime object
                dt = datetime.strptime(raw_date, "%m/%d/%Y")
                
                # Parse Hour: Input "1" -> Output "00", Input "2" -> Output "01"
                # ISO-NE Hour 1 is typically 00:00 - 01:00
                hour_int = int(raw_hour)
                output_hour = hour_int - 1
                
                # Format Timestamp: "2021-09-01 00:00"
                timestamp_str = dt.strftime(f"%Y-%m-%d {output_hour:02d}:00")
                
                data.append((timestamp_str, price))
            except Exception as e:
                print(f"Warning: Skipping malformed row in {filepath.name}: {row} - Error: {e}")
                continue
                
    return data

def main():
    input_path = Path(INPUT_DIR)
    
    # 1. Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        sys.exit(1)
    
    # 2. Get all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No .csv files found in '{INPUT_DIR}'.")
        sys.exit(1)
        
    print(f"Found {len(csv_files)} CSV files. Starting validation...")
    
    # 3. Validate ALL files first (as requested)
    valid_files = []
    for file in csv_files:
        try:
            validate_file(file)
            valid_files.append(file)
            print(f"  [OK] {file.name}")
        except ValueError as e:
            print(f"  [FAIL] {file.name} - {e}")
            print("\nAborting process. Please fix the files above and try again.")
            sys.exit(1)
            
    print("\nValidation successful. Processing files...")
    
    # 4. Process and Concatenate
    all_data = []
    for file in valid_files:
        try:
            file_data = process_file(file)
            all_data.extend(file_data)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            sys.exit(1)
            
    if not all_data:
        print("Warning: No data rows ('D' rows) extracted from any file.")
        # Still create empty file with header
    else:
        # 5. Sort by Timestamp
        # Sorting strings works fine for ISO format YYYY-MM-DD HH:MM
        all_data.sort(key=lambda x: x[0])
        print(f"Sorted {len(all_data)} records by timestamp.")
    
    # 6. Write Output
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price"])
            writer.writerows(all_data)
            
        print(f"\nSuccess! Output written to '{OUTPUT_FILE}'")
        print(f"Total records: {len(all_data)}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()