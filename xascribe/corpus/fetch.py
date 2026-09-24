"""Legally open full text only: open-access PDFs (metadata OA links, arXiv, Unpaywall) or
Europe PMC open-access XML.  Records without open text can be kept abstract-only."""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

from .sources import contact_email, http_get, normalize_doi


def _is_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def pdf_to_text(data: bytes) -> str:
    import fitz  # PyMuPDF
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n\n".join(page.get_text() for page in doc)


def epmc_fulltext(pmcid: str) -> str:
    r = http_get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    if r is None or not r.content.strip().startswith(b"<"):
        return ""
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    return "\n\n".join(" ".join(p.itertext()) for p in (body if body is not None else root).iter("p"))


def unpaywall_pdf(doi: str) -> str:
    if not doi:
        return ""
    r = http_get(f"https://api.unpaywall.org/v2/{doi}", params={"email": contact_email()})
    if r is None:
        return ""
    loc = (r.json() or {}).get("best_oa_location") or {}
    return loc.get("url_for_pdf") or ""


def candidate_urls(rec: dict) -> list[str]:
    urls = [u["url"] for u in rec.get("oa_urls", []) if u.get("kind") == "pdf"]
    if rec.get("arxiv_id"):
        urls.append(f"https://arxiv.org/pdf/{rec['arxiv_id']}")
    up = unpaywall_pdf(normalize_doi(rec.get("doi", "")))
    if up:
        urls.append(up)
    return list(dict.fromkeys(urls))


def fetch_text(rec: dict, cache_dir: Path) -> tuple[str, str]:
    """Return (text, provenance); ('', '') when no open full text is available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{rec['doc_id']}.txt"
    if cached.exists() and cached.stat().st_size > 2000:
        return cached.read_text(encoding="utf-8"), "cache"
    if rec.get("pmcid") and rec.get("epmc_oa"):
        t = epmc_fulltext(rec["pmcid"])
        if len(t) > 3000:
            cached.write_text(t, encoding="utf-8")
            return t, f"europepmc:{rec['pmcid']}"
    for url in candidate_urls(rec):
        r = http_get(url, timeout=60, retries=1)
        if r is None or not _is_pdf(r.content):
            continue
        try:
            t = pdf_to_text(r.content)
        except Exception:
            continue
        if len(t) > 3000:
            cached.write_text(t, encoding="utf-8")
            return t, url
    return "", ""


def clean_text(text: str) -> str:
    """Normalise whitespace, drop the reference list when it sits in the last part of the text."""
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    m = list(re.finditer(r"\n\s*(references|references and notes|bibliography)\s*\n", text, re.I))
    if m and m[-1].start() > 0.55 * len(text):
        text = text[:m[-1].start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Paragraph-aware 1000-character chunks with 200-character overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p).strip()
            continue
        if buf:
            chunks.append(buf)
        while len(p) > size:
            chunks.append(p[:size]); p = p[size - overlap:]
        buf = p
    if buf:
        chunks.append(buf)
    return [((chunks[i - 1][-overlap:] + " ") if i else "") + c for i, c in enumerate(chunks)]
