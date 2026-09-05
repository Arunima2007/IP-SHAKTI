"""Script to discover, hash, and inventory all PDF documents."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.inventory import scan_and_inventory_documents

if __name__ == "__main__":
    print("Running document discovery and inventory...")
    inventory = scan_and_inventory_documents()
    print(f"\nInventory complete. Discovered {len(inventory)} documents.")
    for idx, item in enumerate(inventory, 1):
        ocr_flag = f" [OCR: {len(item.ocr_pages)} pages]" if item.requires_ocr else ""
        print(f"{idx:02d}. [{item.category.upper()}] {item.filename} ({item.page_count} pages, Jurisdiction: {item.jurisdiction}){ocr_flag}")
