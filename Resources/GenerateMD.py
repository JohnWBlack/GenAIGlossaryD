"""
generate_markdown.py

A Python script to generate markdown files for a Generative AI Glossary from a structured CSV file.

Author: JohnWBlack
License: MIT License (See LICENSE file for full text)
Version: 1.0
"""

import os
import pandas as pd

# Function to load the CSV file
def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        print("CSV file loaded successfully!")
        print(data.head())  # Debugging: Print the first few rows
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

# Function to generate markdown files
def generate_markdown(data, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group by Level 1 (main categories)
    categories = data['Level 1'].dropna().unique()
    if not categories.any():
        print("No categories found in 'Level 1' column.")
        return

    for category in categories:
        print(f"Processing category: {category}")  # Debugging
        # Filter data for the current category
        category_data = data[data['Level 1'] == category]

        # Start building the markdown content
        markdown_content = f"# {category}\n\n"
        markdown_content += f"This section provides definitions and explanations for key terms related to **{category}**.\n\n"

        # Group by Level 2 (subcategories) within the current category
        subcategories = category_data['Level 2'].dropna().unique()
        for subcategory in subcategories:
            print(f"Processing subcategory: {subcategory}")  # Debugging
            markdown_content += f"## {subcategory}\n\n"

            # Filter terms within the subcategory
            terms_data = category_data[category_data['Level 2'] == subcategory]
            for _, row in terms_data.iterrows():
                term = row['Level 3'] if not pd.isna(row['Level 3']) else "General"
                definition = row['Notes'] if not pd.isna(row['Notes']) else "No definition provided."
                print(f"Term: {term}, Definition: {definition}")  # Debugging
                markdown_content += f"- **{term}:** {definition}\n"
            markdown_content += "\n"

        # Save the markdown file with UTF-8 encoding
        file_name = f"{category.replace(' ', '_')}.md"
        with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as file:
            file.write(markdown_content)

    print(f"Markdown files have been generated in the directory: {output_dir}")

# Main function to execute the script
if __name__ == "__main__":
    # Input CSV file path
    csv_file_path = input("Enter the path to the CSV file: ").strip()

    # Output directory for markdown files
    output_directory = input("Enter the output directory for markdown files: ").strip()

    # Load the CSV data
    glossary_data = load_csv(csv_file_path)

    if glossary_data is not None:
        # Generate markdown files
        generate_markdown(glossary_data, output_directory)
