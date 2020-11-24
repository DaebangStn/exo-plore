FOUND_OUTPUT = """
"""

import csv
import os

PARAM_FILE_NAMES = [
    # 'selected_k5_d1',
    # 'selected_k5_d2',
    # 'selected_k5_d3',
    # 'selected_k5_d4',
    'selected_k2_d1',
    'selected_k2_d2',
    'selected_k2_d3',
    'selected_k2_d4',
]

def parse_and_save_csv_blocks(multiline_text, param_file_names):
    lines = multiline_text.strip().split('\n')
    csv_lines = []
    param_idx = 0
    for line in lines:
        if len(line) == 0:
            continue
        csv_file_content = line.startswith('param_idx') or line[:1].isdigit()
        if not csv_file_content and len(csv_lines) > 0:
            save_txt_block(csv_lines, param_file_names[param_idx])
            param_idx += 1
            csv_lines = []
        elif csv_file_content:
            csv_lines.append(line)
    
    if len(csv_lines) > 0:
        save_txt_block(csv_lines, param_file_names[param_idx])

def save_txt_block(csv_lines, param_file_name):
    output_txt = os.path.join(os.path.dirname(__file__), '../../experiment_data/params/{}.csv'.format(param_file_name))
    with open(output_txt, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"Saved {output_txt}")

def main():
    parse_and_save_csv_blocks(FOUND_OUTPUT, PARAM_FILE_NAMES)

if __name__ == "__main__":
    main()


