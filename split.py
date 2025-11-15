import os
from pathlib import Path

def split_prompts_file(input_file, output_dir="prompts_split"):
    """
    Parse a prompts file and generate individual files for each prompt.
    
    Args:
        input_file: Path to the input prompts file
        output_dir: Directory to store the split prompt files
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    
    # Split by the prompt delimiter
    prompts = content.split('==== PROMPT END ====')
    
    # Filter out empty prompts and strip whitespace
    prompts = [p.strip() for p in prompts if p.strip()]
    
    # Generate individual files
    for idx, prompt in enumerate(prompts, 1):
        output_file = os.path.join(output_dir, f"{idx}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt + '\n==== PROMPT END ====')
        print(f"Created: {output_file}")
    
    print(f"\nTotal prompts generated: {len(prompts)}")

if __name__ == "__main__":
    split_prompts_file("prompts.txt")