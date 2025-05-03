from fastapi import APIRouter, Query
from server.app.model.toc import ToC
from typing import List, Optional

from server.app.service.toc_service import ToCService

router = APIRouter(prefix="/pdf", tags=["Pdf"])

@router.get("/toc")
def get_toc(pdf_path: str = Query(None, description="path to the pdf")):
    return ToCService.get_toc(pdf_path=pdf_path)