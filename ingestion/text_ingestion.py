# text_ingestion.py
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from docling.datamodel.document import TextItem, TableItem, PictureItem
from langchain.schema import Document
from .vectordb import vector_db
from .images_ingestion import ingest_images
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os

# Configure Docling
pipeline_options = PdfPipelineOptions(
    generate_picture_images=True,
    generate_page_images=True,
    images_scale=1.5
)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)


def save_picture_item(element, conv_doc, out_dir: Path, base_name: str, counter: int):
    """
    Save a PictureItem to out_dir. Returns the saved filepath (str) or None.
    """
    try:
        pil_img = element.get_image(conv_doc)
        if pil_img is None:
            return None
        if pil_img.mode in ("RGBA", "P", "LA"):
            pil_img = pil_img.convert("RGB")
        filename = f"{base_name}_picture_{counter}.png"
        filepath = out_dir / filename
        # ensure out_dir exists
        out_dir.mkdir(parents=True, exist_ok=True)
        pil_img.save(filepath, format="PNG")
        return str(filepath)
    except Exception as e:
        print(f"⚠️ Failed to save picture: {e}")
        return None


def export_table_item(table_item, base_name: str, table_ix: int, out_dir: Path):
    """
    Export a TableItem to CSV in out_dir and return (csv_path_str or None, markdown_str)
    """
    try:
        df = table_item.export_to_dataframe()
    except Exception:
        try:
            rows = []
            for row in table_item.data.table_cells:
                cells = [c.text for c in row]
                rows.append(cells)
            df = pd.DataFrame(rows)
        except Exception as e:
            print(f"⚠️ Failed to extract table: {e}")
            df = pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{base_name}-table-{table_ix + 1}.csv"
    table_md = ""
    if not df.empty:
        try:
            df.to_csv(csv_path, index=False)
            table_md = df.to_markdown(index=False)
        except Exception as e:
            print(f"⚠️ Failed to convert table: {e}")
    return str(csv_path) if csv_path.exists() else None, table_md


def load_documents(file_paths, out_dir: Path):
    """
    Convert documents with Docling and save extracted images & table CSVs to out_dir.
    Returns (docs, extracted_images)
    """
    docs = []
    extracted_images = []
    for path in tqdm(file_paths, desc="📂 Converting documents with Docling"):
        path = Path(path)
        print(f"📄 Ingesting {path.name} ...")
        base_name = path.stem
        doc_images = []
        try:
            conv_res = converter.convert(str(path))
            conv_doc = conv_res.document
            table_counter = 0
            picture_counter = 0
            for element, level in conv_doc.iterate_items():
                if isinstance(element, TextItem):
                    text = (element.text or "").strip()
                    if text:
                        meta = {
                            "source": str(path),
                            "element_type": "text",
                            "page": getattr(element, "page_no", 1),
                        }
                        docs.append(Document(page_content=text, metadata=meta))
                elif isinstance(element, TableItem):
                    table_counter += 1
                    csv_path, table_md = export_table_item(
                        element, base_name, table_counter - 1, out_dir
                    )
                    content = table_md or "[table content]"
                    meta = {
                        "source": str(path),
                        "element_type": "table",
                        "table_index": table_counter,
                    }
                    if csv_path:
                        meta["csv_path"] = csv_path
                        meta["local_path"] = csv_path
                    docs.append(Document(page_content=content, metadata=meta))
                elif isinstance(element, PictureItem):
                    picture_counter += 1
                    pic_path = save_picture_item(
                        element, conv_doc, out_dir, base_name, picture_counter
                    )
                    if pic_path:
                        doc_images.append(pic_path)
                        meta = {
                            "source": str(path),
                            "element_type": "picture",
                            "picture_index": picture_counter,
                            "local_path": pic_path,
                        }
                        docs.append(Document(
                            page_content=f"[IMAGE: {os.path.basename(pic_path)}]",
                            metadata=meta
                        ))
            try:
                full_md = conv_doc.export_to_markdown()
                if full_md.strip():
                    docs.append(Document(
                        page_content=full_md,
                        metadata={"source": str(path), "element_type": "full_document"}
                    ))
            except Exception:
                # ignore export_to_markdown failures
                pass
            extracted_images.extend(doc_images)
            print(f"📸 Extracted {len(doc_images)} images from {path.name}")
        except Exception as e:
            print(f"❌ Docling failed for {path}: {e}")
    print(f"📊 Total documents loaded: {len(docs)} chunks")
    print(f"📊 Total extracted images: {len(extracted_images)}")
    return docs, extracted_images


def ingest_texts(user_id, chatbot_id, file_paths=None, base_storage: Path = Path("uploads")):
    """
    Ingest texts and store images & tables under:
      {base_storage}/users/{user_id}/{chatbot_id}/documents

    - file_paths: list of files to ingest
    - base_storage: base folder (defaults to 'uploads')
    """
    if not file_paths:
        print("⚠️ No documents to ingest!")
        return

    # Make sure base_storage is a Path
    base_storage = Path(base_storage)

    # Build and ensure user/chatbot-specific documents directory
    out_dir = base_storage / "users" / str(user_id) / str(chatbot_id) / "documents"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs, extracted_images = load_documents(file_paths, out_dir)
    if not docs:
        print("⚠️ No documents were successfully loaded.")
        return

    # Create embeddings & splitter
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")

    chunks = splitter.split_documents(docs)

    # Store with user_id and chatbot_id metadata
    vector_db.store_documents(
        documents=chunks,
        user_id=user_id,
        chatbot_id=chatbot_id,
        content_type="text"
    )

    print(f"✅ {len(chunks)} text chunks added for user {user_id}, chatbot {chatbot_id}.")

    if extracted_images:
        # extracted_images already contain absolute/relative paths inside out_dir
        print(f"🖼️ Ingesting {len(extracted_images)} images…")
        ingest_images(
            user_id=user_id,
            chatbot_id=chatbot_id,
            file_paths=extracted_images
        )
