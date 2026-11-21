import os
from typing import List
from unstructured.partition.pdf import partition_pdf
from embedchain import App
from embedchain.loaders.base_loader import BaseLoader
from embedchain.models.data_type import DataType
from embedchain.utils.misc import clean_string

load_dotenv = True
# Set your API key for LLM synthesis (or configure a local HuggingFace/Ollama LLM in config)
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
class MultimodalInsurancePDFLoader(BaseLoader):
    """
    Custom Embedchain loader using Unstructured 'hi_res' strategy to preserve
    benefit tables and OCR text from embedded infographics/flowcharts.
    """
    def load_data(self, file_path: str):
        # Partition PDF using layout-aware models + Tesseract OCR for images
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",                 # Uses layout detection for tables & images
            infer_table_structure=True,        # Preserves rows/columns as HTML
            extract_images_in_pdf=False,       # Set True if saving raw image crops
            languages=["eng"]
        )

        data = []
        for idx, el in enumerate(elements):
            # Preserve exact table structure via HTML metadata if available
            if el.category == "Table" and getattr(el.metadata, "text_as_html", None):
                content = f"### Extracted Insurance Table:\n{el.metadata.text_as_html}"
            elif el.category == "Image":
                content = f"### Extracted Infographic/Chart Text:\n{el.text}"
            else:
                content = el.text

            cleaned_content = clean_string(content)
            if not cleaned_content:
                continue

            data.append({
                "content": cleaned_content,
                "meta_data": {
                    "url": file_path,
                    "page_number": getattr(el.metadata, "page_number", 1),
                    "element_type": el.category,
                    "chunk_index": idx
                }
            })

        return {
            "doc_id": f"insurance_pdf_{os.path.basename(file_path)}",
            "data": data
        }


# Configure Embedchain with HuggingFace Embeddings + Qdrant Scalar Quantization (int8)
config = {
    "app": {
        "config": {
            "id": "indian-insurance-rag"
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            # Strong dense embedding model for financial/insurance text & tables
            "model": "BAAI/bge-base-en-v1.5"
        }
    },
    "vectordb": {
        "provider": "qdrant",
        "config": {
            "collection_name": "insurance_quantized_vectors",
            "path": "./qdrant_insurance_db",  # Local persistent storage
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100
            },
            # Quantize float32 embeddings to int8 (4x memory reduction, ~99% recall retention)
            "quantization_config": {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True
                }
            }
        }
    },
    "chunker": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "length_function": "len"
    }
}


def process_insurance_pdfs(pdf_files: List[str]):
    app = App.from_config(config=config)
    custom_loader = MultimodalInsurancePDFLoader()

    for pdf_path in pdf_files:
        print(f"Ingesting {pdf_path} (extracting tables & OCRing images)...")
        app.add(
            pdf_path,
            data_type=DataType.CUSTOM,
            loader=custom_loader
        )
    return app


if __name__ == "__main__":
    # List of local Indian insurance PDFs (brochures, wordings, IRDAI reports)
    insurance_pdfs = [
        "StarHealthAssureInsurancePolicy-Brochure.pdf",
        "IRDAI_Annual_Report_2024-25.pdf"
    ]

    rag_app = process_insurance_pdfs(insurance_pdfs)

    # Query tabular & visual data
    query = (
        "Compare the room rent limits, cumulative bonus, and waiting periods "
        "across different Sum Insured tiers from the policy tables."
    )
    response = rag_app.query(query)
    print("\n--- Extracted Insights ---")
    print(response)