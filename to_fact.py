import os
from pathlib import Path

class TPCHFactConverter:
    """Convert TPC-H .tbl files to natural language facts"""
    
    def __init__(self, input_dir, output_file="tpch_facts.txt"):
        self.input_dir = Path(input_dir)
        self.output_file = output_file
        
        # Define schemas for each TPC-H table
        self.schemas = {
            'nation': {
                'columns': ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'],
                'template': 'With the nation key set to {n_nationkey}, the country named {n_name}, set to the region with key {n_regionkey}, has the comment "{n_comment}".'
            },
            'region': {
                'columns': ['r_regionkey', 'r_name', 'r_comment'],
                'template': 'The region with key {r_regionkey} is named {r_name} and has the comment "{r_comment}".'
            },
            'supplier': {
                'columns': ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'],
                'template': 'Supplier {s_suppkey} named {s_name}, located at {s_address} in nation {s_nationkey}, with phone {s_phone}, account balance {s_acctbal}, has the comment "{s_comment}".'
            },
            'customer': {
                'columns': ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'],
                'template': 'Customer {c_custkey} named {c_name}, located at {c_address} in nation {c_nationkey}, with phone {c_phone}, account balance {c_acctbal}, market segment {c_mktsegment}, has the comment "{c_comment}".'
            },
            'part': {
                'columns': ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'],
                'template': 'Part {p_partkey} named {p_name}, manufactured by {p_mfgr}, brand {p_brand}, type {p_type}, size {p_size}, container {p_container}, retail price {p_retailprice}, has the comment "{p_comment}".'
            },
            'partsupp': {
                'columns': ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'],
                'template': 'Part {ps_partkey} supplied by supplier {ps_suppkey} has available quantity {ps_availqty}, supply cost {ps_supplycost}, and comment "{ps_comment}".'
            },
            'orders': {
                'columns': ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'],
                'template': 'Order {o_orderkey} placed by customer {o_custkey} has status {o_orderstatus}, total price {o_totalprice}, order date {o_orderdate}, priority {o_orderpriority}, clerk {o_clerk}, ship priority {o_shippriority}, and comment "{o_comment}".'
            },
            'lineitem': {
                'columns': ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'],
                'template': 'Line item {l_linenumber} of order {l_orderkey} for part {l_partkey} from supplier {l_suppkey}: quantity {l_quantity}, extended price {l_extendedprice}, discount {l_discount}, tax {l_tax}, return flag {l_returnflag}, line status {l_linestatus}, ship date {l_shipdate}, commit date {l_commitdate}, receipt date {l_receiptdate}, ship instruction "{l_shipinstruct}", ship mode {l_shipmode}, comment "{l_comment}".'
            }
        }
    
    def parse_tbl_line(self, line, delimiter='|'):
        """Parse a line from a .tbl file"""
        # TPC-H .tbl files end with a delimiter, so we remove the last empty element
        parts = line.strip().split(delimiter)
        if parts[-1] == '':
            parts = parts[:-1]
        return parts
    
    def create_fact(self, table_name, values):
        """Create a natural language fact from table values"""
        schema = self.schemas.get(table_name)
        if not schema:
            return None
        
        # Create a dictionary mapping column names to values
        data = {}
        for i, col in enumerate(schema['columns']):
            if i < len(values):
                data[col] = values[i].strip()
            else:
                data[col] = ''
        
        # Format the template with the data
        try:
            fact = schema['template'].format(**data)
            return fact
        except KeyError as e:
            print(f"Warning: Missing key {e} for table {table_name}")
            return None
    
    def convert_file(self, tbl_file, table_name):
        """Convert a single .tbl file to facts"""
        facts = []
        try:
            with open(tbl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        values = self.parse_tbl_line(line)
                        fact = self.create_fact(table_name, values)
                        if fact:
                            facts.append(fact)
        except Exception as e:
            print(f"Error processing {tbl_file}: {e}")
        
        return facts
    
    def convert_all(self, max_rows_per_table=None):
        """Convert all TPC-H .tbl files to facts (and optionally make truncated .tbt copies)."""
        all_facts = []

        for table_name in self.schemas.keys():
            tbl_file = self.input_dir / f"{table_name}.tbl"

            if not tbl_file.exists():
                print(f"Warning: {tbl_file} not found, skipping...")
                continue

            print(f"Processing {table_name}.tbl...")
            facts = []
            raw_lines = []

            try:
                with open(tbl_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            values = self.parse_tbl_line(line)
                            fact = self.create_fact(table_name, values)
                            if fact:
                                facts.append(fact)
                                raw_lines.append(line.strip())

                        # Stop early if we reach the row limit
                        if max_rows_per_table and len(facts) >= max_rows_per_table:
                            break
            except Exception as e:
                print(f"Error processing {tbl_file}: {e}")
                continue

            # Save truncated .tbt file if limit is active
            if max_rows_per_table:
                tbt_file = self.input_dir / f"{table_name}.tbt"
                try:
                    with open(tbt_file, 'w', encoding='utf-8') as out_tbt:
                        out_tbt.write('\n'.join(raw_lines) + '\n')
                    print(f"  Wrote truncated file: {tbt_file} ({len(raw_lines)} rows)")
                except Exception as e:
                    print(f"  Error writing {tbt_file}: {e}")

            all_facts.append(f"\n{'='*80}")
            all_facts.append(f"TABLE: {table_name.upper()}")
            all_facts.append(f"{'='*80}\n")
            all_facts.extend(facts)

            print(f"  Converted {len(facts)} rows")

        # Write all facts to output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_facts))

        print(f"\nAll facts written to {self.output_file}")
        print(f"Total facts: {len([f for f in all_facts if not f.startswith('=') and f.strip()])}")
        
# Example usage
if __name__ == "__main__":
    # Configure your paths
    INPUT_DIR = "./out"  # Directory containing .tbl files
    OUTPUT_FILE = "tpch_facts_small.txt"
    
    # Optional: limit rows per table for testing (None = all rows)
    MAX_ROWS = 10000  # Set to e.g., 100 for testing
    
    converter = TPCHFactConverter(INPUT_DIR, OUTPUT_FILE)
    converter.convert_all(max_rows_per_table=MAX_ROWS)