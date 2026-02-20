
# app.py — Single-file MLB Player Value Table Service (FastAPI)
# Endpoints:
#   POST /salary
#   POST /war
#   POST /player_value_table
#
# Dependencies: fastapi uvicorn requests beautifulsoup4 pybaseball pandas numpy

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# pybaseball (WAR)
from pybaseball import playerid_lookup, pitching_stats, batting_stats, pitching_stats_bref, batting_stats_bref
from pybaseball import cache as pyb_cache

pyb_cache.enable()

app = FastAPI(title="MLB Player Value Table Service", version="0.1")

# -----------------------
# Simple in-memory cache (v1)
# -----------------------
_MEMCACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days

def cache_get(key: str) -> Optional[Any]:
    item = _MEMCACHE.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > CACHE_TTL_SECONDS:
        return None
    return item["val"]

def cache_set(key: str, val: Any) -> None:
    _MEMCACHE[key] = {"ts": time.time(), "val": val}

def sha_key(prefix: str, s: str) -> str:
    return prefix + ":" + hashlib.sha256(s.encode("utf-8")).hexdigest()

# -----------------------
# MLB StatsAPI client (minimal)
# -----------------------
MLB_STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"

@dataclass
class FileCache:
    cache_dir: str = ".mlb_api_cache"
    ttl_seconds: int = 60 * 60 * 24 * 14

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path_for_key(self, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for_key(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fetched_at = payload.get("_fetched_at", 0)
            if (time.time() - fetched_at) > self.ttl_seconds:
                return None
            return payload.get("data")
        except Exception:
            return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        path = self._path_for_key(key)
        payload = {"_fetched_at": time.time(), "data": data}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

class MLBStatsAPI:
    def __init__(self, cache: Optional[FileCache] = None, timeout: int = 30):
        self.cache = cache or FileCache()
        self.timeout = timeout

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        key = f"GET:{url}?{json.dumps(params, sort_keys=True)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        self.cache.set(key, data)
        return data

    def search_people(self, name_query: str) -> pd.DataFrame:
        data = self._get(f"{MLB_STATSAPI_BASE}/people/search", params={"names": name_query})
        rows = []
        for p in data.get("people", []):
            primary_pos = (p.get("primaryPosition") or {}).get("abbreviation")
            current_team = (p.get("currentTeam") or {}).get("name")
            rows.append({
                "name": p.get("fullName"),
                "mlbam_id": p.get("id"),
                "active": p.get("active"),
                "primary_position": primary_pos,
                "current_team": current_team,
                "birth_date": p.get("birthDate"),
            })
        return pd.DataFrame(rows)

    def get_person(self, mlbam_id: int) -> Dict[str, Any]:
        data = self._get(
            f"{MLB_STATSAPI_BASE}/people/{mlbam_id}",
            params={"hydrate": "currentTeam,team"}
        )
        people = data.get("people", [])
        if not people:
            raise ValueError(f"No person returned for mlbam_id={mlbam_id}")
        return people[0]

    def resolve_player(
        self,
        name_query: str,
        prefer_team: Optional[str] = None,
        prefer_position: Optional[str] = None,
        prefer_active: bool = True
    ) -> Dict[str, Any]:
        candidates = self.search_people(name_query)
        if candidates.empty:
            raise ValueError(f"No StatsAPI people results for '{name_query}'.")

        def score(row) -> int:
            s = 0
            if prefer_active and bool(row.get("active")):
                s += 10
            if prefer_team and isinstance(row.get("current_team"), str) and prefer_team.lower() in row["current_team"].lower():
                s += 5
            if prefer_position and isinstance(row.get("primary_position"), str) and row["primary_position"].upper() == prefer_position.upper():
                s += 5
            if isinstance(row.get("name"), str) and row["name"].lower() == name_query.lower():
                s += 3
            return s

        candidates = candidates.copy()
        candidates["score"] = candidates.apply(score, axis=1)
        candidates = candidates.sort_values(["score", "active"], ascending=[False, False]).reset_index(drop=True)

        top = candidates.iloc[0].to_dict()
        mlbam_id = int(top["mlbam_id"])
        profile = self.get_person(mlbam_id)

        return {
            "mlbam_id": mlbam_id,
            "name": profile.get("fullName") or top.get("name"),
            "primary_position": (profile.get("primaryPosition") or {}).get("abbreviation"),
            "current_team": (profile.get("currentTeam") or {}).get("name"),
            "active": profile.get("active"),
            "birth_date": profile.get("birthDate"),
            "raw_profile": profile,
            "resolution_candidates": candidates.drop(columns=["score"]).head(10).to_dict(orient="records"),
        }

    def season_team_splits(self, mlbam_id: int, year: int, group: str):
        data = self._get(
            f"{MLB_STATSAPI_BASE}/people/{mlbam_id}/stats",
            params={"stats": "season", "group": group, "season": year}
        )
        stats = data.get("stats", [])
        if not stats:
            return []
        return stats[0].get("splits", []) or []

    def team_string_for_season(self, mlbam_id: int, year: int, is_pitcher: bool) -> str:
        group = "pitching" if is_pitcher else "hitting"
        splits = self.season_team_splits(mlbam_id, year, group)
        if not splits:
            return ""
        teams = []
        for s in splits:
            t = (s.get("team") or {}).get("name") or ""
            if t and t.lower() != "major league baseball":
                teams.append(t)
        teams = list(dict.fromkeys(teams))
        if len(teams) == 0:
            return ""
        if len(teams) == 1:
            return teams[0]
        return f"{teams[0]} \u2192 {teams[-1]}"

    def transactions_for_player(self, mlbam_id: int, start_date: str, end_date: str):
        data = self._get(
            f"{MLB_STATSAPI_BASE}/transactions",
            params={"playerId": mlbam_id, "startDate": start_date, "endDate": end_date}
        )
        return data.get("transactions", []) or []

    @staticmethod
    def summarize_transactions(transactions):
        out = []
        for t in transactions or []:
            out.append({
                "date": t.get("date"),
                "typeCode": t.get("typeCode"),
                "typeDesc": t.get("typeDesc"),
                "description": t.get("description"),
                "fromTeam": (t.get("fromTeam") or {}).get("name"),
                "toTeam": (t.get("toTeam") or {}).get("name"),
            })
        out.sort(key=lambda x: (x.get("date") or ""))
        return out

def build_player_season_skeleton(api: MLBStatsAPI, player_name: str, start_year: int, end_year: int,
                                prefer_team: Optional[str] = None, prefer_position: Optional[str] = None):
    resolved = api.resolve_player(player_name, prefer_team=prefer_team, prefer_position=prefer_position, prefer_active=True)
    mlbam_id = resolved["mlbam_id"]
    pos = (resolved.get("primary_position") or "").upper()
    is_pitcher = (pos == "P") if pos else (prefer_position == "P")

    rows = []
    for year in range(start_year, end_year + 1):
        team_str = api.team_string_for_season(mlbam_id, year, is_pitcher=is_pitcher)
        rows.append({"Year": year, "Team": team_str})
    return resolved, pd.DataFrame(rows), is_pitcher

# -----------------------
# Salary (Spotrac scrape)
# -----------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StablewoodValueBot/0.1)"}

def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()

def spotrac_search_player(player_name: str) -> Optional[str]:
    q = player_name.strip()
    search_url = f"https://www.spotrac.com/search/?query={requests.utils.quote(q)}"
    r = requests.get(search_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.select("a"):
        href = a.get("href") or ""
        if href.startswith("/mlb/") and href.count("/") >= 4:
            return "https://www.spotrac.com" + href
    return None

def parse_money_to_millions(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip().replace(",", "")
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*M", s, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    val = float(m.group(1))
    return val / 1_000_000 if val > 1000 else val

def parse_contract_terms_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = " ".join(text.split()).lower()
    m = re.search(r"(\d+)\s*(?:yr|year)[s]?\s*[/,-]?\s*\$([\d,]+(?:\.\d+)?)", t)
    if m:
        years = int(m.group(1))
        total_raw = m.group(2).replace(",", "")
        total = float(total_raw)
        total_m = total / 1_000_000 if total > 1000 else total
        return {"years": years, "total_value_m": round(total_m, 3)}
    m = re.search(r"(\d+)\s*-\s*year[s]?\s*,?\s*\$([\d,]+(?:\.\d+)?)\s*(m|million)?", t)
    if m:
        years = int(m.group(1))
        val = float(m.group(2).replace(",", ""))
        if m.group(3) in ("m", "million"):
            return {"years": years, "total_value_m": round(val, 3)}
        return {"years": years, "total_value_m": round(val / 1_000_000 if val > 1000 else val, 3)}
    return None

def infer_contract_start_year_from_salary_years(salary_by_year: Dict[int, float], contract_years: int) -> Optional[int]:
    if not salary_by_year or not contract_years:
        return None
    years_sorted = sorted(salary_by_year.keys())
    high_years = [y for y in years_sorted if salary_by_year.get(y, 0) is not None and salary_by_year.get(y, 0) > 2.0]
    if not high_years:
        return None
    for start in high_years:
        ok = True
        for k in range(contract_years):
            y = start + k
            if y not in salary_by_year or salary_by_year.get(y) is None or salary_by_year.get(y) <= 2.0:
                ok = False
                break
        if ok:
            return start
    return None

def spotrac_scrape_salary_and_contract(player_url: str) -> Dict[str, Any]:
    r = requests.get(player_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Spotrac fetch failed: {r.status_code}")
    soup = BeautifulSoup(r.text, "html.parser")

    salary_by_year: Dict[int, float] = {}
    warnings: List[str] = []

    for tbl in soup.select("table"):
        text = tbl.get_text(" ", strip=True).lower()
        if ("salary" in text or "cap hit" in text or "cash" in text) and ("202" in text or "201" in text or "200" in text):
            for tr in tbl.select("tr"):
                cols = [c.get_text(" ", strip=True) for c in tr.select("td,th")]
                if len(cols) < 2:
                    continue
                year_match = None
                for c in cols:
                    if re.fullmatch(r"(19|20)\d{2}", c.strip()):
                        year_match = int(c.strip()); break
                if not year_match:
                    continue
                cash_val = None
                for c in cols:
                    if "$" in c or re.search(r"\d+(\.\d+)?\s*M", c, re.I):
                        cash_val = parse_money_to_millions(c)
                        if cash_val is not None:
                            break
                if cash_val is not None:
                    salary_by_year[year_match] = float(round(cash_val, 3))
            if len(salary_by_year) >= 2:
                break

    if not salary_by_year:
        warnings.append("Spotrac salary table not found or could not be parsed.")

    page_text = soup.get_text(" ", strip=True)
    contract_terms = parse_contract_terms_from_text(page_text)
    contract_record = None
    if contract_terms:
        start_year = infer_contract_start_year_from_salary_years(salary_by_year, contract_terms["years"])
        if start_year is not None:
            contract_record = {
                "start_year": int(start_year),
                "years": int(contract_terms["years"]),
                "total_value_m": float(contract_terms["total_value_m"]),
                "source": "Spotrac",
                "source_url": player_url
            }
        else:
            warnings.append("Contract terms found but start_year could not be inferred from salary years.")
    else:
        warnings.append("Contract terms not found on Spotrac page (or could not be parsed).")

    return {"salary_by_year": salary_by_year, "contract_record": contract_record, "warnings": warnings}

# -----------------------
# WAR (pybaseball): fWAR primary, bWAR fallback
# -----------------------
def split_first_last(player_name: str):
    parts = [p for p in player_name.strip().split(" ") if p]
    if len(parts) < 2:
        return player_name.strip(), ""
    return parts[0], parts[-1]

def resolve_war_ids(player_name: str, mlbam_id: Optional[int]) -> Dict[str, Any]:
    first, last = split_first_last(player_name)
    df = playerid_lookup(last, first)
    if df.empty:
        return {"idfg": None, "bbref_id": None, "warnings": ["playerid_lookup returned no matches."]}
    for c in ["key_mlbam", "key_fangraphs", "key_bbref"]:
        if c not in df.columns:
            df[c] = None
    if mlbam_id is not None:
        m = df[df["key_mlbam"] == int(mlbam_id)]
        if not m.empty:
            row = m.iloc[0]
            return {
                "idfg": int(row["key_fangraphs"]) if pd.notna(row["key_fangraphs"]) else None,
                "bbref_id": str(row["key_bbref"]) if pd.notna(row["key_bbref"]) else None,
                "warnings": []
            }
    warnings = ["Matched by name (mlbam_id not found in lookup results)."]
    if "mlb_played_last" in df.columns:
        df2 = df.copy()
        df2["mlb_played_last"] = pd.to_numeric(df2["mlb_played_last"], errors="coerce")
        df2 = df2.sort_values("mlb_played_last", ascending=False)
        row = df2.iloc[0]
    else:
        row = df.iloc[0]
    return {
        "idfg": int(row["key_fangraphs"]) if pd.notna(row["key_fangraphs"]) else None,
        "bbref_id": str(row["key_bbref"]) if pd.notna(row["key_bbref"]) else None,
        "warnings": warnings
    }

def fetch_fwar_by_year(idfg: int, is_pitcher: bool, start_year: int, end_year: int) -> Dict[int, float]:
    df = pitching_stats(start_year, end_year) if is_pitcher else batting_stats(start_year, end_year)
    if df.empty or "IDfg" not in df.columns or "Season" not in df.columns or "WAR" not in df.columns:
        return {}
    m = df[df["IDfg"] == int(idfg)]
    if m.empty:
        return {}
    out = {}
    for _, r in m.iterrows():
        y = int(r["Season"])
        war = pd.to_numeric(r["WAR"], errors="coerce")
        if pd.notna(war):
            out[y] = float(war)
    return out

def fetch_bwar_for_year(bbref_id: str, is_pitcher: bool, year: int) -> Optional[float]:
    df = pitching_stats_bref(year) if is_pitcher else batting_stats_bref(year)
    if df is None or df.empty or "WAR" not in df.columns:
        return None
    id_cols = [c for c in df.columns if c.lower() in ("id","player_id","mlb_id","key_bbref","bbref_id")]
    if not id_cols:
        return None
    for c in id_cols:
        m = df[df[c].astype(str) == str(bbref_id)]
        if not m.empty:
            war = pd.to_numeric(m.iloc[0]["WAR"], errors="coerce")
            if pd.notna(war):
                return float(war)
    return None

# -----------------------
# Service time (estimated)
# -----------------------
SERVICE_DAYS_PER_YEAR = 172

def format_service_time(total_days: int) -> str:
    years = total_days // SERVICE_DAYS_PER_YEAR
    days = total_days % SERVICE_DAYS_PER_YEAR
    return f"{years}.{days:03d}"

def estimate_entering_service_time(api: MLBStatsAPI, mlbam_id: int, year: int, is_pitcher: bool) -> (Optional[str], str):
    profile = api.get_person(mlbam_id)
    debut = profile.get("mlbDebutDate")
    if not debut or not re.match(r"^\d{4}-\d{2}-\d{2}$", debut):
        return None, "Missing mlbDebutDate in StatsAPI; cannot estimate."
    debut_year = int(debut[:4])

    if year <= debut_year:
        return "0.000", "Estimated (pre-debut entering season)."

    def regular_season_end_date(y: int) -> str:
        return f"{y}-10-01"

    def days_between(start_yyyy_mm_dd: str, end_yyyy_mm_dd: str) -> int:
        from datetime import date
        sy, sm, sd = map(int, start_yyyy_mm_dd.split("-"))
        ey, em, ed = map(int, end_yyyy_mm_dd.split("-"))
        return max(0, (date(ey, em, ed) - date(sy, sm, sd)).days)

    def played_in_mlb_year(y: int) -> bool:
        group = "pitching" if is_pitcher else "hitting"
        splits = api.season_team_splits(mlbam_id, y, group)
        return len(splits) > 0

    total_days = 0
    debut_days = min(SERVICE_DAYS_PER_YEAR, days_between(debut, regular_season_end_date(debut_year)))
    total_days += debut_days

    for y in range(debut_year + 1, year):
        if played_in_mlb_year(y):
            total_days += SERVICE_DAYS_PER_YEAR

    return format_service_time(total_days), "Estimated from debut date + assumed full MLB seasons; may differ from official."

# -----------------------
# Narratives (contract + transaction + service time)
# -----------------------
def parse_service_time_to_days(st: Optional[str]) -> Optional[int]:
    if not st or not isinstance(st, str):
        return None
    m = re.fullmatch(r"(\d+)\.(\d{3})", st.strip())
    if not m:
        return None
    y = int(m.group(1))
    d = int(m.group(2))
    d = max(0, min(d, SERVICE_DAYS_PER_YEAR - 1))
    return y * SERVICE_DAYS_PER_YEAR + d

def classify_contract_bucket(service_time: Optional[str]) -> Optional[str]:
    days = parse_service_time_to_days(service_time)
    if days is None:
        return None
    y3, y4, y5, y6 = 3*SERVICE_DAYS_PER_YEAR, 4*SERVICE_DAYS_PER_YEAR, 5*SERVICE_DAYS_PER_YEAR, 6*SERVICE_DAYS_PER_YEAR
    if days < y3: return "Pre-Arb"
    if days < y4: return "Arb Year 1"
    if days < y5: return "Arb Year 2"
    if days < y6: return "Arb Year 3"
    return "Free Agent"

def pick_season_transactions(transactions: List[Dict[str, Any]], season_year: int) -> List[Dict[str, Any]]:
    out = []
    for t in transactions or []:
        d = (t.get("date") or "")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        y = int(d[:4])
        if (y == season_year and d <= f"{season_year}-10-31") or (y == season_year - 1 and d >= f"{season_year-1}-11-01"):
            out.append(t)
    out.sort(key=lambda x: (x.get("date") or ""))
    return out

def transaction_type_from_events(events: List[Dict[str, Any]], team_str: str) -> str:
    if not events:
        return "-"
    descs = " | ".join([(e.get("typeDesc") or "") + " " + (e.get("description") or "") for e in events]).lower()
    if "debut" in descs:
        return "Debut"
    if "traded" in descs or "trade" in descs:
        # if team_str contains arrow, keep it
        if "→" in team_str:
            return f"Traded ({team_str.replace(' ', '')})"
        return "Traded"
    if "free agent" in descs and ("signed" in descs or "sign" in descs):
        return "Free Agent Signing"
    if "qualifying offer" in descs and ("accept" in descs or "agreed" in descs):
        return "Accepted Qualifying Offer"
    chain = []
    if "waiver" in descs: chain.append("Waived")
    if "claimed" in descs: chain.append("Claimed")
    if "non-tender" in descs or "nontender" in descs: chain.append("Non-tendered")
    if chain:
        return " → ".join(dict.fromkeys(chain))
    first = events[-1]
    return (first.get("typeDesc") or "-").strip() or "-"

def infer_debut_year_from_transactions(all_transactions: List[Dict[str, Any]]) -> Optional[int]:
    for t in all_transactions or []:
        d = t.get("date") or ""
        desc = ((t.get("typeDesc") or "") + " " + (t.get("description") or "")).lower()
        if "debut" in desc and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            return int(d[:4])
    return None

def contract_status_event(year: int, debut_year: Optional[int], service_time_entering: Optional[str],
                          salary_m: Optional[float], contract_record: Optional[Dict[str, Any]] = None) -> str:
    if debut_year is not None and year == debut_year:
        return "Pre-Arb (MLB Debut)"
    if contract_record:
        start = int(contract_record["start_year"])
        years = int(contract_record["years"])
        total = float(contract_record["total_value_m"])
        if start <= year < start + years:
            yr_num = (year - start) + 1
            return f"Yr {yr_num} of {years}/${total:.0f}M"
    bucket = classify_contract_bucket(service_time_entering)
    if bucket:
        return bucket
    if salary_m is not None and salary_m <= 1.2:
        return "Pre-Arb"
    return "Needs Service Time"

# -----------------------
# Market $/WAR curve
# -----------------------
def market_dollars_per_war(year: int, base_year: int = 2013, base_value_m: float = 7.00, annual_growth: float = 0.05) -> float:
    if year < base_year:
        raise ValueError("year < base_year")
    return round(base_value_m * ((1 + annual_growth) ** (year - base_year)), 2)

# -----------------------
# API Schemas
# -----------------------
class SalaryRequest(BaseModel):
    player_name: str

class SalaryYear(BaseModel):
    year: int
    cash_paid_m: Optional[float] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    notes: Optional[str] = None

class SalaryResponse(BaseModel):
    player_name: str
    salary_by_year: List[SalaryYear]
    contract_record: Optional[Dict[str, Any]] = None
    warnings: List[str] = []

class WarRequest(BaseModel):
    player_name: str
    mlbam_id: Optional[int] = None
    is_pitcher: Optional[bool] = None
    start_year: int
    end_year: int

class WarYear(BaseModel):
    year: int
    war: Optional[float] = None
    war_source: Optional[str] = None
    notes: Optional[str] = None

class WarResponse(BaseModel):
    player_name: str
    mlbam_id: Optional[int] = None
    idfg: Optional[int] = None
    bbref_id: Optional[str] = None
    war_by_year: List[WarYear]
    warnings: List[str] = []

class ValueTableRequest(BaseModel):
    player_name: str
    start_year: int
    end_year: int
    prefer_team: Optional[str] = None
    prefer_position: Optional[str] = None
    base_year: int = 2013
    base_value_m: float = 7.00
    annual_growth: float = 0.05

# -----------------------
# Endpoints
# -----------------------
@app.post("/salary", response_model=SalaryResponse)
def salary(req: SalaryRequest):
    pn = req.player_name.strip()
    if not pn:
        raise HTTPException(status_code=400, detail="player_name required")

    ck = sha_key("salary", norm_name(pn))
    cached = cache_get(ck)
    if cached is not None:
        return cached

    player_url = spotrac_search_player(pn)
    if not player_url:
        out = SalaryResponse(player_name=pn, salary_by_year=[], contract_record=None,
                            warnings=["Spotrac search failed; no player URL found."]).model_dump()
        cache_set(ck, out)
        return out

    scraped = spotrac_scrape_salary_and_contract(player_url)
    salary_by_year = [
        SalaryYear(year=y, cash_paid_m=v, source="Spotrac", source_url=player_url, notes=None).model_dump()
        for y, v in sorted(scraped.get("salary_by_year", {}).items())
    ]
    out = SalaryResponse(
        player_name=pn,
        salary_by_year=salary_by_year,
        contract_record=scraped.get("contract_record"),
        warnings=list(scraped.get("warnings") or [])
    ).model_dump()

    cache_set(ck, out)
    return out

@app.post("/war", response_model=WarResponse)
def war(req: WarRequest):
    if req.end_year < req.start_year:
        raise HTTPException(status_code=400, detail="end_year must be >= start_year")

    ck = sha_key("war", f"{req.player_name}|{req.mlbam_id}|{req.is_pitcher}|{req.start_year}|{req.end_year}")
    cached = cache_get(ck)
    if cached is not None:
        return cached

    warnings: List[str] = []
    ids = resolve_war_ids(req.player_name, req.mlbam_id)
    warnings.extend(ids.get("warnings", []))
    idfg = ids.get("idfg")
    bbref_id = ids.get("bbref_id")

    if req.is_pitcher is None:
        warnings.append("is_pitcher not provided; defaulting to False.")
        is_pitcher = False
    else:
        is_pitcher = bool(req.is_pitcher)

    fwar_map = fetch_fwar_by_year(idfg, is_pitcher, req.start_year, req.end_year) if idfg else {}
    if not idfg:
        warnings.append("Could not resolve FanGraphs ID (IDfg); fWAR may be missing.")

    war_by_year: List[Dict[str, Any]] = []
    for y in range(req.start_year, req.end_year + 1):
        if y in fwar_map:
            war_by_year.append(WarYear(year=y, war=round(fwar_map[y], 1), war_source="fWAR").model_dump())
        else:
            if not bbref_id:
                war_by_year.append(WarYear(year=y, war=None, war_source=None, notes="Missing ID for bWAR fallback.").model_dump())
                continue
            bw = fetch_bwar_for_year(bbref_id, is_pitcher, y)
            if bw is not None:
                war_by_year.append(WarYear(year=y, war=round(bw, 1), war_source="bWAR", notes="Used bWAR fallback.").model_dump())
            else:
                war_by_year.append(WarYear(year=y, war=None, war_source=None, notes="WAR not found in fWAR or bWAR sources.").model_dump())

    out = WarResponse(
        player_name=req.player_name,
        mlbam_id=req.mlbam_id,
        idfg=idfg,
        bbref_id=bbref_id,
        war_by_year=war_by_year,
        warnings=warnings
    ).model_dump()
    cache_set(ck, out)
    return out

@app.post("/player_value_table")
def player_value_table(req: ValueTableRequest):
    if req.end_year < req.start_year:
        raise HTTPException(status_code=400, detail="end_year must be >= start_year")

    api = MLBStatsAPI(cache=FileCache(cache_dir=".mlb_cache", ttl_seconds=60*60*24*14))
    resolved, df, is_pitcher = build_player_season_skeleton(
        api, req.player_name, req.start_year, req.end_year,
        prefer_team=req.prefer_team, prefer_position=req.prefer_position
    )

    warnings: List[str] = []
    # salary
    s = salary(SalaryRequest(player_name=resolved["name"]))
    warnings.extend(s.get("warnings") or [])
    salary_map = {int(x["year"]): x for x in s.get("salary_by_year", [])}

    sal_vals = []
    for y in df["Year"].tolist():
        rec = salary_map.get(int(y))
        sal_vals.append(rec.get("cash_paid_m") if rec else np.nan)
    df["Salary ($M)"] = pd.to_numeric(sal_vals, errors="coerce")

    contract_record = s.get("contract_record")

    # war
    w = war(WarRequest(player_name=resolved["name"], mlbam_id=resolved["mlbam_id"], is_pitcher=is_pitcher,
                       start_year=req.start_year, end_year=req.end_year))
    warnings.extend(w.get("warnings") or [])
    war_map = {int(x["year"]): x for x in w.get("war_by_year", [])}

    wars, war_src = [], []
    for y in df["Year"].tolist():
        rec = war_map.get(int(y), {})
        wars.append(rec.get("war", np.nan))
        war_src.append(rec.get("war_source"))
    df["fWAR"] = pd.to_numeric(wars, errors="coerce")
    df["WAR Source"] = war_src

    # service time entering season
    st_vals, st_notes = [], []
    for y in df["Year"].tolist():
        st, note = estimate_entering_service_time(api, resolved["mlbam_id"], int(y), is_pitcher=is_pitcher)
        st_vals.append(st)
        st_notes.append(note)
    df["Service Time (Entering Season)"] = st_vals

    # transactions + narratives
    tx = api.summarize_transactions(api.transactions_for_player(resolved["mlbam_id"], f"{req.start_year-1}-11-01", f"{req.end_year}-10-31"))
    debut_year = infer_debut_year_from_transactions(tx)

    contract_events, tx_types = [], []
    for _, row in df.iterrows():
        year = int(row["Year"])
        team_str = str(row.get("Team") or "")
        season_events = pick_season_transactions(tx, season_year=year)
        tx_type = transaction_type_from_events(season_events, team_str)
        salary_m = row.get("Salary ($M)")
        salary_m = float(salary_m) if pd.notna(salary_m) else None
        st = row.get("Service Time (Entering Season)")
        st = str(st) if st is not None and str(st) != "nan" else None
        cse = contract_status_event(year, debut_year, st, salary_m, contract_record)
        tx_types.append(tx_type)
        contract_events.append(cse)

    df["Transaction Type"] = tx_types
    df["Contract Status / Event"] = contract_events

    # market + value math
    df["Market $/WAR ($M)"] = df["Year"].apply(lambda y: market_dollars_per_war(int(y), req.base_year, req.base_value_m, req.annual_growth))
    df["Value"] = df["fWAR"] * df["Market $/WAR ($M)"]
    df["Actual $/WAR ($M)"] = np.where((df["fWAR"] > 0) & (df["Salary ($M)"].notna()), df["Salary ($M)"] / df["fWAR"], np.nan)
    df["Surplus Value"] = np.where((df["Value"].notna()) & (df["Salary ($M)"].notna()), df["Value"] - df["Salary ($M)"], np.nan)

    out = df[[
        "Year","Team","Salary ($M)","Contract Status / Event","Transaction Type",
        "Service Time (Entering Season)","fWAR","WAR Source","Actual $/WAR ($M)",
        "Market $/WAR ($M)","Value","Surplus Value"
    ]].copy()

    # rounding
    for c in ["Salary ($M)","Actual $/WAR ($M)","Market $/WAR ($M)","Value","Surplus Value"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out["fWAR"] = pd.to_numeric(out["fWAR"], errors="coerce").round(1)

    totals = {
        "Year": "Totals",
        "Team": "",
        "Salary ($M)": float(out["Salary ($M)"].sum(skipna=True)),
        "Contract Status / Event": "",
        "Transaction Type": "",
        "Service Time (Entering Season)": "",
        "fWAR": float(out["fWAR"].sum(skipna=True)),
        "WAR Source": "",
        "Actual $/WAR ($M)": "",
        "Market $/WAR ($M)": "",
        "Value": float(out["Value"].sum(skipna=True)),
        "Surplus Value": float(out["Surplus Value"].sum(skipna=True)),
    }
    out = pd.concat([out, pd.DataFrame([totals])], ignore_index=True)

    return {
        "player": {
            "name": resolved["name"],
            "mlbam_id": resolved["mlbam_id"],
            "primary_position": resolved.get("primary_position"),
            "current_team": resolved.get("current_team")
        },
        "params": req.model_dump(),
        "rows": out.to_dict(orient="records"),
        "totals": totals,
        "warnings": warnings
    }
