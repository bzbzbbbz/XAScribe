"""Bibliographic search back ends and record normalisation.

Every back end is a public, key-free API (OpenAlex, Europe PMC, arXiv; Crossref is
available but not used by default because its PDF links are usually subscriber
TDM links, not open access).  All of them return the same record shape:

    {
      "doc_id":   stable, filesystem-safe identifier (from DOI / arXiv id / title)
      "doi":      normalised DOI ('' if none)
      "arxiv_id": arXiv identifier ('' if none)
      "pmcid":    PubMed Central id ('' if none)
      "title", "abstract", "journal", "volume", "issue", "pages": str
      "year":     int | None
      "authors":  [{"given": str, "family": str}, ...]
      "oa_urls":  [{"url", "kind" ('pdf'|'landing'), "provenance", "license", "version"}]
      "is_oa":    bool   (the metadata source asserts a legal open-access copy exists)
      "epmc_oa":  bool   (Europe PMC open-access subset: full-text XML is available)
      "sources":  ["openalex", ...]
      "queries":  [search strings that surfaced the record]
    }

Also home of the citation formatter (RSC style) and the DOI / HTML-entity helpers,
because they operate purely on record metadata.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import sys
import time
import urllib.parse
from typing import Callable, Iterable
from xml.etree import ElementTree as ET

import requests

TIMEOUT = 40

# Minimum seconds between requests to the same host (arXiv asks for >= 3 s).
_MIN_INTERVAL = {"export.arxiv.org": 3.2, "api.openalex.org": 0.12,
                 "www.ebi.ac.uk": 0.2, "api.unpaywall.org": 0.1, "api.crossref.org": 0.2}
_LAST_CALL: dict[str, float] = {}


def contact_email() -> str:
    """Contact address sent to OpenAlex/Unpaywall/Crossref 'polite pools'."""
    return os.environ.get("XASCRIBE_CONTACT_EMAIL", "xascribe-user@example.org")


def user_agent() -> dict:
    return {"User-Agent": f"XAScribe-corpus/3.0 (+https://github.com/xascribe; mailto:{contact_email()})"}


def safe_print(msg: str) -> None:
    """print() that never dies on a narrow console code page (e.g. cp936)."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def _throttle(url: str) -> None:
    host = urllib.parse.urlparse(url).netloc
    wait = _MIN_INTERVAL.get(host, 0.3) - (time.time() - _LAST_CALL.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[host] = time.time()


def http_get(url: str, params: dict | None = None, timeout: int = TIMEOUT, retries: int = 3,
             stream: bool = False, errors: list | None = None, **kw) -> requests.Response | None:
    """GET with per-host throttling and retry on 429/5xx/timeouts.  None on failure."""
    last = ""
    for attempt in range(retries):
        _throttle(url)
        try:
            r = requests.get(url, params=params, headers=user_agent(), timeout=timeout,
                             stream=stream, **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After", "")
                time.sleep(min(30.0, float(ra)) if ra.isdigit() else 3.0 * (attempt + 1))
                continue
            break
        except requests.RequestException as exc:
            last = type(exc).__name__
            time.sleep(2.0 * (attempt + 1))
    if errors is not None:
        errors.append(f"GET {url[:120]} failed: {last}")
    return None


# ------------------------------------------------------------------ text helpers
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*)?/?>")


def clean_markup(text) -> str:
    """Unescape (possibly double-escaped) HTML entities, then strip tags.

    'LiNi&lt;sub&gt;0.8&lt;/sub&gt;O&lt;sub&gt;2&lt;/sub&gt;' -> 'LiNi0.8O2'.  A bare '<'
    followed by a space or digit (e.g. 'x < 0.5') is not treated as a tag.
    """
    if not text:
        return ""
    text = str(text)
    for _ in range(3):
        un = html.unescape(text)
        if un == text:
            break
        text = un
    text = _TAG.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


_DOI_PREFIX = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/|^doi:\s*", re.I)


def normalize_doi(doi) -> str:
    """Lower-case DOI without resolver prefix: 'https://doi.org/10.1021/X' -> '10.1021/x'."""
    if not doi:
        return ""
    d = urllib.parse.unquote(str(doi).strip())
    d = _DOI_PREFIX.sub("", d).strip()
    d = _DOI_PREFIX.sub("", d).strip()          # tolerate 'doi: https://doi.org/...'
    return d.rstrip(".,;) ").lower()


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_markup(title).lower())[:140]


def slugify(s: str, n: int = 90) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:n].strip("_.") or "doc"


def make_doc_id(rec: dict) -> str:
    if rec.get("doi"):
        return slugify(rec["doi"])
    if rec.get("arxiv_id"):
        return slugify("arXiv_" + rec["arxiv_id"])
    h = hashlib.sha1(normalize_title(rec.get("title", "")).encode()).hexdigest()[:12]
    return f"t_{h}"


# ----------------------------------------------------------------- author names
_PARTICLES = {"van", "von", "der", "den", "de", "del", "della", "di", "da", "du", "dos",
              "das", "le", "la", "ter", "ten", "zu", "st.", "bin", "al", "el"}
_HYPHENS = re.compile(r"[-‐‑‒–]")


def parse_name(name: str) -> dict:
    """Split a free-form author string into {'given', 'family'}.

    Handles 'Family, Given', Europe PMC 'Family GI', and 'Given [particles] Family'.
    """
    name = clean_markup(name).strip().strip(",")
    if not name:
        return {"given": "", "family": ""}
    if "," in name:
        fam, giv = name.split(",", 1)
        return {"given": giv.strip(), "family": fam.strip()}
    m = re.match(r"^(.+?)\s+([A-Z]{1,3})\.?$", name)       # 'Dai H', 'Mukerjee SK'
    if m and not re.match(r"^[A-Z]\.", name):
        return {"given": m.group(2), "family": m.group(1)}
    toks = name.split()
    if len(toks) == 1:
        return {"given": "", "family": toks[0]}
    i = len(toks) - 1
    while i > 1 and toks[i - 1].lower() in _PARTICLES:
        i -= 1
    return {"given": " ".join(toks[:i]), "family": " ".join(toks[i:])}


def _initials(given: str) -> str:
    out = []
    for tok in given.replace(".", ". ").split():
        tok = tok.strip()
        if not tok:
            continue
        if re.fullmatch(r"[A-Z]{2,3}", tok):                  # Europe PMC 'JF'
            out.extend(f"{c}." for c in tok)
            continue
        parts = [p for p in _HYPHENS.split(tok) if p.strip(". ")]
        if not parts:
            continue
        out.append("-".join(p.strip(". ")[0].upper() + "." for p in parts))
    return " ".join(out)


def format_author(a) -> str:
    """{'given': 'Bing-Joe', 'family': 'Hwang'} or 'Bing-Joe Hwang' -> 'B.-J. Hwang'."""
    if isinstance(a, str):
        a = parse_name(a)
    fam = (a.get("family") or "").strip()
    ini = _initials(a.get("given") or "")
    return f"{ini} {fam}".strip() if fam else ini


def format_authors(authors: list, max_authors: int = 6) -> str:
    names = [n for n in (format_author(a) for a in (authors or [])) if n]
    if not names:
        return ""
    if len(names) > max_authors:
        return ", ".join(names[:max_authors]) + " et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def format_citation(rec: dict, include_doi: bool = True, max_authors: int = 6) -> str:
    """RSC-style reference string.

    'A. B. Author, C. Author and D. Author, Title, Journal, Year, Volume, Pages.'
    (first six authors, then 'et al.'); with include_doi the DOI is appended as
    ', DOI: 10.xxxx/yyyy.'  Missing fields are skipped.
    """
    title = clean_markup(rec.get("title", "")).rstrip(". ")
    parts = [format_authors(rec.get("authors") or [], max_authors), title,
             clean_markup(rec.get("journal", "")), str(rec.get("year") or ""),
             str(rec.get("volume") or "").strip(), str(rec.get("pages") or "").strip()]
    doi = normalize_doi(rec.get("doi", ""))
    if include_doi and doi:
        parts.append(f"DOI: {doi}")
    return ", ".join(p for p in parts if p) + "."


def format_pages(first, last) -> str:
    first, last = (str(first or "").strip(), str(last or "").strip())
    if first and last and last != first:
        return f"{first}–{last}"
    return first or last


# ----------------------------------------------------------------- record shape
def new_record(**kw) -> dict:
    rec = dict(doc_id="", doi="", arxiv_id="", pmcid="", title="", abstract="", journal="",
               volume="", issue="", pages="", year=None, authors=[], oa_urls=[],
               is_oa=False, epmc_oa=False, license="", sources=[], queries=[])
    rec.update(kw)
    rec["doi"] = normalize_doi(rec.get("doi"))
    rec["title"] = clean_markup(rec.get("title"))
    rec["abstract"] = clean_markup(rec.get("abstract"))
    rec["journal"] = clean_markup(rec.get("journal"))
    return rec


def _oa(url, kind, provenance, license="", version="") -> dict:
    return {"url": url, "kind": kind, "provenance": provenance,
            "license": license or "", "version": version or ""}


# ---------------------------------------------------------------------- OpenAlex
def _invert_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


_OA_SELECT = ("id,doi,title,display_name,publication_year,primary_location,biblio,authorships,"
              "abstract_inverted_index,best_oa_location,open_access,locations,type,ids")


def openalex(query: str, n: int = 25, from_year: int | None = None,
             errors: list | None = None) -> list[dict]:
    filters = ["type:article|review|preprint"]
    if from_year:
        filters.append(f"from_publication_date:{int(from_year)}-01-01")
    params = {"search": query, "per-page": min(200, max(1, n)), "filter": ",".join(filters),
              "select": _OA_SELECT, "mailto": contact_email()}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    r = http_get("https://api.openalex.org/works", params=params, errors=errors)
    if r is None:
        return []
    out = []
    for w in r.json().get("results", [])[:n]:
        prim = w.get("primary_location") or {}
        src = prim.get("source") or {}
        bib = w.get("biblio") or {}
        oa_urls, seen = [], set()
        locs = [w.get("best_oa_location") or {}] + list(w.get("locations") or [])
        for loc in locs:
            if not loc or not loc.get("is_oa"):
                continue
            lic, ver = loc.get("license") or "", loc.get("version") or ""
            prov = "openalex:" + ((loc.get("source") or {}).get("display_name") or "oa_location")
            if loc.get("pdf_url") and loc["pdf_url"] not in seen:
                seen.add(loc["pdf_url"])
                oa_urls.append(_oa(loc["pdf_url"], "pdf", prov, lic, ver))
            lp = loc.get("landing_page_url")
            if lp and lp not in seen and not lp.startswith("https://doi.org/"):
                seen.add(lp)
                oa_urls.append(_oa(lp, "landing", prov, lic, ver))
        ids = w.get("ids") or {}
        pmcid = str(ids.get("pmcid") or "").rsplit("/", 1)[-1]
        out.append(new_record(
            doi=w.get("doi") or "",
            pmcid=pmcid if pmcid.upper().startswith("PMC") else "",
            title=w.get("title") or w.get("display_name") or "",
            abstract=_invert_abstract(w.get("abstract_inverted_index")),
            year=w.get("publication_year"),
            journal=src.get("display_name") or "",
            volume=bib.get("volume") or "", issue=bib.get("issue") or "",
            pages=format_pages(bib.get("first_page"), bib.get("last_page")),
            authors=[parse_name((a.get("author") or {}).get("display_name") or "")
                     for a in (w.get("authorships") or [])],
            oa_urls=oa_urls,
            is_oa=bool((w.get("open_access") or {}).get("is_oa")),
            license=(w.get("best_oa_location") or {}).get("license") or "",
            sources=["openalex"],
        ))
    return out


# -------------------------------------------------------------------- Europe PMC
def europepmc(query: str, n: int = 25, from_year: int | None = None,
              errors: list | None = None) -> list[dict]:
    q = f"({query})" + (f" AND PUB_YEAR:[{int(from_year)} TO 2100]" if from_year else "")
    params = {"query": q, "format": "json", "pageSize": min(100, max(1, n)), "resultType": "core"}
    r = http_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params,
                 errors=errors)
    if r is None:
        return []
    out = []
    for w in (r.json().get("resultList") or {}).get("result", [])[:n]:
        ji = w.get("journalInfo") or {}
        authors = []
        for a in (w.get("authorList") or {}).get("author", []) or []:
            if a.get("lastName"):
                authors.append({"given": a.get("firstName") or a.get("initials") or "",
                                "family": a["lastName"]})
            elif a.get("collectiveName"):
                authors.append({"given": "", "family": a["collectiveName"]})
        if not authors and w.get("authorString"):
            authors = [parse_name(s) for s in w["authorString"].rstrip(".").split(",") if s.strip()]
        epmc_oa = w.get("isOpenAccess") == "Y" and w.get("inEPMC") == "Y" and bool(w.get("pmcid"))
        oa_urls = []
        for ft in (w.get("fullTextUrlList") or {}).get("fullTextUrl", []) or []:
            # Only 'Open access' copies; 'Free' (free-to-read) and subscription links are skipped,
            # and Europe PMC's own rendered PDFs are skipped because the XML route is used instead.
            if ft.get("availabilityCode") == "OA" and ft.get("documentStyle") == "pdf" \
                    and ft.get("site") != "Europe_PMC":
                oa_urls.append(_oa(ft.get("url"), "pdf", "europepmc:" + (ft.get("site") or ""),
                                   w.get("license") or ""))
        year = w.get("pubYear")
        out.append(new_record(
            doi=w.get("doi") or "",
            pmcid=w.get("pmcid") or "",
            title=w.get("title") or "",
            abstract=w.get("abstractText") or "",
            year=int(year) if str(year or "").isdigit() else None,
            journal=(ji.get("journal") or {}).get("title") or w.get("bookOrReportDetails", {}).get("publisher", "")
            if isinstance(w.get("bookOrReportDetails"), dict) else (ji.get("journal") or {}).get("title") or "",
            volume=ji.get("volume") or "", issue=ji.get("issue") or "",
            pages=(w.get("pageInfo") or "").replace("-", "–"),
            authors=authors, oa_urls=oa_urls,
            is_oa=w.get("isOpenAccess") == "Y", epmc_oa=epmc_oa,
            license=w.get("license") or "",
            sources=["europepmc"],
        ))
    return out


# ------------------------------------------------------------------------- arXiv
_STOP = {"the", "of", "and", "in", "for", "on", "with", "a", "an", "to", "by", "from", "at",
         "during", "study", "studies", "using", "via", "review", "analysis", "based"}
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
         "os": "http://a9.com/-/spec/opensearch/1.1/"}


def _arxiv_query_variants(query: str) -> list[str]:
    """arXiv treats juxtaposed terms as OR, so AND the leading significant terms,
    relaxing from four terms to two if nothing matches."""
    toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", query) if t.lower() not in _STOP]
    toks = list(dict.fromkeys(toks))
    variants = []
    for k in (4, 3, 2):
        if len(toks) >= k or (k == 2 and toks):
            sel = toks[:k]
            variants.append(" AND ".join(f'all:"{t}"' if "-" in t else f"all:{t}" for t in sel))
    return list(dict.fromkeys(variants))


def arxiv(query: str, n: int = 15, from_year: int | None = None,
          errors: list | None = None) -> list[dict]:
    for sq in _arxiv_query_variants(query):
        r = http_get("https://export.arxiv.org/api/query",
                     params={"search_query": sq, "start": 0, "max_results": n}, timeout=60,
                     errors=errors)
        if r is None:
            return []
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            return []
        entries = root.findall("a:entry", _ATOM)
        if not entries:
            continue
        out = []
        for e in entries:
            aid = (e.findtext("a:id", "", _ATOM) or "").rsplit("/abs/", 1)[-1]
            aid_nov = re.sub(r"v\d+$", "", aid)
            year = (e.findtext("a:published", "", _ATOM) or "")[:4]
            if from_year and year.isdigit() and int(year) < int(from_year):
                continue
            jref = clean_markup(e.findtext("arxiv:journal_ref", "", _ATOM))
            out.append(new_record(
                doi=e.findtext("arxiv:doi", "", _ATOM) or "",
                arxiv_id=aid_nov,
                title=e.findtext("a:title", "", _ATOM),
                abstract=e.findtext("a:summary", "", _ATOM),
                year=int(year) if year.isdigit() else None,
                journal=jref or "arXiv preprint",
                pages="" if jref else f"arXiv:{aid_nov}",
                authors=[parse_name(a.findtext("a:name", "", _ATOM))
                         for a in e.findall("a:author", _ATOM)],
                oa_urls=[_oa(f"https://arxiv.org/pdf/{aid_nov}", "pdf", "arxiv",
                             "arXiv non-exclusive distribution licence")],
                is_oa=True, sources=["arxiv"],
            ))
        return out
    return []


# ---------------------------------------------------------------------- Crossref
def crossref(query: str, n: int = 25, from_year: int | None = None,
             errors: list | None = None) -> list[dict]:
    """Metadata only: Crossref 'link' entries are usually subscriber TDM links, so no
    OA URL is taken from here (Unpaywall decides open access later)."""
    params = {"query.bibliographic": query, "rows": min(100, n), "mailto": contact_email(),
              "select": "DOI,title,abstract,issued,container-title,author,volume,issue,page"}
    if from_year:
        params["filter"] = f"from-pub-date:{int(from_year)}-01-01"
    r = http_get("https://api.crossref.org/works", params=params, errors=errors)
    if r is None:
        return []
    out = []
    for w in (r.json().get("message") or {}).get("items", [])[:n]:
        issued = ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        out.append(new_record(
            doi=w.get("DOI", ""), title=(w.get("title") or [""])[0],
            abstract=w.get("abstract") or "", year=issued,
            journal=(w.get("container-title") or [""])[0],
            volume=w.get("volume") or "", issue=w.get("issue") or "",
            pages=(w.get("page") or "").replace("-", "–"),
            authors=[{"given": a.get("given", ""), "family": a.get("family", a.get("name", ""))}
                     for a in (w.get("author") or [])],
            sources=["crossref"],
        ))
    return out


BACKENDS: dict[str, Callable] = {"openalex": openalex, "europepmc": europepmc,
                                 "arxiv": arxiv, "crossref": crossref}
DEFAULT_SOURCES = ("openalex", "europepmc", "arxiv")


# ----------------------------------------------------------------- de-duplication
def _keys(rec: dict) -> list[str]:
    ks = []
    if rec.get("doi"):
        ks.append("doi:" + rec["doi"])
    if rec.get("arxiv_id"):
        ks.append("arxiv:" + rec["arxiv_id"].lower())
    t = normalize_title(rec.get("title", ""))
    if len(t) >= 25:
        ks.append("title:" + t)
    return ks


def merge_into(base: dict, other: dict) -> dict:
    """Fill gaps in `base` from `other` (same work found by another back end)."""
    for f in ("doi", "arxiv_id", "pmcid", "abstract", "journal", "volume", "issue", "pages",
              "year", "license"):
        if not base.get(f) and other.get(f):
            base[f] = other[f]
    if len(other.get("abstract") or "") > len(base.get("abstract") or "") * 1.3:
        base["abstract"] = other["abstract"]
    if base.get("journal") in ("arXiv preprint", "") and other.get("journal") not in ("", "arXiv preprint"):
        base["journal"], base["volume"], base["pages"] = other["journal"], other["volume"], other["pages"]
    if len(other.get("authors") or []) > len(base.get("authors") or []):
        base["authors"] = other["authors"]
    urls = {u["url"] for u in base["oa_urls"]}
    base["oa_urls"] += [u for u in other.get("oa_urls", []) if u["url"] not in urls]
    base["is_oa"] = base["is_oa"] or other.get("is_oa", False)
    base["epmc_oa"] = base["epmc_oa"] or other.get("epmc_oa", False)
    base["sources"] = list(dict.fromkeys(base["sources"] + other.get("sources", [])))
    base["queries"] = list(dict.fromkeys(base["queries"] + other.get("queries", [])))
    return base


def dedupe(records: Iterable[dict]) -> list[dict]:
    out, index = [], {}
    for rec in records:
        if not rec.get("title"):
            continue
        hit = next((index[k] for k in _keys(rec) if k in index), None)
        if hit is None:
            out.append(rec)
            hit = rec
        else:
            merge_into(hit, rec)
        for k in _keys(hit):
            index.setdefault(k, hit)
    for rec in out:
        rec["doc_id"] = make_doc_id(rec)
    return out


def search_all(queries: Iterable[str], backends=DEFAULT_SOURCES, per_query: int = 25,
               from_year: int | None = None, progress: Callable | None = None,
               errors: list | None = None) -> tuple[list[dict], dict]:
    """Run every query on every back end; return (unique records, raw hit counts)."""
    raw, counts = [], {b: 0 for b in backends}
    for q in queries:
        for b in backends:
            try:
                recs = BACKENDS[b](q, per_query if b != "arxiv" else min(per_query, 15),
                                   from_year, errors=errors)
            except Exception as exc:                       # never let one source kill a run
                if errors is not None:
                    errors.append(f"{b} '{q[:50]}': {type(exc).__name__}: {exc}")
                recs = []
            for r in recs:
                r["queries"] = [q]
            counts[b] += len(recs)
            raw.extend(recs)
            if progress:
                progress(f"   {b:<9} {len(recs):3d} hits  '{q[:60]}'")
    uniq = dedupe(raw)
    return uniq, {"raw_hits": counts, "raw_total": len(raw), "unique": len(uniq)}
