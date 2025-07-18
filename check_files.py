import os
from pypdf import PdfReader
from pathlib import Path
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

def count_tokens(text):
  # Simple whitespace tokenization
  return len(text.split())

def get_pdf_info(pdf_path):
  try:
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
      text += page.extract_text() or ""
    tokens = count_tokens(text)
    return num_pages, tokens
  except Exception as e:
    print(f"Error reading {pdf_path}: {e}")
    return 0, 0

def main():
  pdf_dir = Path("files")
  pdf_files = list(pdf_dir.glob("*.pdf"))
  print(f"Found {len(pdf_files)} PDF files.")

  page_counts = []
  token_counts = []

  for pdf in pdf_files:
    pages, tokens = get_pdf_info(pdf)
    page_counts.append(pages)
    token_counts.append(tokens)

  if not pdf_files:
    print("No PDF files found.")
    return

  avg_pages = sum(page_counts) / len(page_counts)
  avg_tokens = sum(token_counts) / len(token_counts)
  print(f"Average number of pages: {avg_pages:.2f}")
  print(f"Average number of tokens: {avg_tokens:.2f}")

  # Plot average lengths
  # plt.figure(figsize=(6,4))
  # plt.bar(['Pages', 'Tokens'], [avg_pages, avg_tokens])
  # plt.title('Average PDF Lengths')
  # plt.ylabel('Average Count')
  # plt.savefig('average_lengths.png')
  # plt.show()

  # Plot histogram of token counts
  plt.figure(figsize=(8,5))
  plt.hist(token_counts, bins=20, color='skyblue', edgecolor='black')
  plt.title('Histogram of PDF Token Counts')
  plt.xlabel('Token Count')
  plt.ylabel('Number of PDFs')
  plt.savefig('token_histogram.png')
  plt.show()

  # Group files by base name (before first underscore or dot)

  group_tokens = defaultdict(list)
  for pdf, tokens in zip(pdf_files, token_counts):
    # Extract group name: everything before first underscore or dot
    base = pdf.stem.split('_')[0].split('.')[0]
    group_tokens[base].append(tokens)

  group_avg_tokens = {k: sum(v)/len(v) for k, v in group_tokens.items()}

  # Plot average token count per group
  plt.figure(figsize=(10, 5))
  plt.bar(group_avg_tokens.keys(), group_avg_tokens.values(), color='orange', edgecolor='black')
  plt.title('Average Token Count per File Group')
  plt.xlabel('File Group')
  plt.ylabel('Average Token Count')
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig('group_avg_token_counts.png')
  plt.show()

  # Histogram of token counts grouped by file group (all groups in one plot, different colors)
  plt.figure(figsize=(10, 6))
  colors = plt.cm.tab20.colors  # Up to 20 distinct colors
  for idx, (group, tokens) in enumerate(group_tokens.items()):
    plt.hist(tokens, bins=10, alpha=0.6, label=group, color=colors[idx % len(colors)], edgecolor='black')
  plt.title('Histogram of Token Counts by File Group')
  plt.xlabel('Token Count')
  plt.ylabel('Number of PDFs')
  plt.legend(title='File Group')
  plt.tight_layout()
  plt.savefig('grouped_token_histogram.png')
  plt.show()

if __name__ == "__main__":
  main()