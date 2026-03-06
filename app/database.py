import uuid
from datetime import datetime, timezone

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://plansmartai:plansmartai@localhost:5432/plansmartai",
)

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)          # blueprint or spec
    file_path = Column(String, nullable=False)
    processing_status = Column(String, default="pending")  # pending/processing/done/failed
    progress_detail = Column(String, nullable=True)     # human-readable step description
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class BlueprintPage(Base):
    __tablename__ = "blueprint_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String, nullable=False)
    numarkdown_text = Column(Text, nullable=True)
    qdrant_point_id = Column(String, nullable=True)


class TextChunk(Base):
    __tablename__ = "text_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    section_header = Column(String, nullable=True, default="")
    qdrant_point_id = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)
