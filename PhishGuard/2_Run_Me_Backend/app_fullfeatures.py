# app_fullfeatures.py
"""
Full-feature Flask inference server for phishing model.

- Loads rf_phishing_model.pkl
- Discovers model feature order via feature_names.pkl OR dataset CSV (dataset_phishing_87 features.csv)
- Computes the full set of features (static, whois/dns, some dynamic via Selenium optional)
- Returns JSON: {"prediction": int, "probability": float, "latency_ms": float}
"""

import os
import time
import re
import joblib
import logging
import traceback
from urllib.parse import urlparse

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# optional libs
try:
    import whois
except Exception:
    whois = None
try:
    import dns.resolver
except Exception:
    dns = None

# optional selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from bs4 import BeautifulSoup
except Exception:
    webdriver = None

# ---------- Configuration ----------
MODEL_PATH = "rf_phishing_model_calibrated.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"                 # optional
TRAINING_CSV_PATH = "dataset_phishing_87 features.csv"   # your uploaded CSV
LOG_PATH = "predict_requests.log"

# Optional lookup CSVs for web_traffic/page_rank/statistical_report (not required)
WEB_TRAFFIC_CSV = None   # e.g., "web_traffic_lookup.csv"
PAGE_RANK_CSV = None     # e.g., "page_rank_lookup.csv"
STAT_REPORT_CSV = None   # e.g., "statistical_report_lookup.csv"

# By default dynamic extraction (Selenium) is off because it's slow
ENABLE_DYNAMIC_BY_DEFAULT = False

# Cache for WHOIS/DNS lookups (in-memory; you can extend to file cache)
whois_cache = {}
dns_cache = {}

# Setup logging
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

# ---------- Load model and determine feature order ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
logging.info("Loaded model from %s", MODEL_PATH)

# Attempt to load explicit feature order first
if os.path.exists(FEATURE_NAMES_PATH):
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logging.info("Loaded feature names from %s (count=%d)", FEATURE_NAMES_PATH, len(feature_names))
else:
    # Fallback: read training CSV and take columns (exclude 'status'/'label' etc)
    if not os.path.exists(TRAINING_CSV_PATH):
        raise FileNotFoundError(f"No feature names file and training CSV not found: {TRAINING_CSV_PATH}")
    df_train = pd.read_csv(TRAINING_CSV_PATH)
    cols = list(df_train.columns)
    # drop common label columns if present
    for lab in ("status", "label", "target", "y"):
        if lab in cols:
            cols.remove(lab)
    feature_names = cols
    logging.info("Derived feature names from training CSV %s (count=%d)", TRAINING_CSV_PATH, len(feature_names))

# Optional: load lookup CSVs into maps
def load_lookup_map(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if 'domain' in df.columns and 'value' in df.columns:
            return dict(zip(df['domain'].astype(str), df['value']))
    return {}

web_traffic_map = load_lookup_map(WEB_TRAFFIC_CSV) if WEB_TRAFFIC_CSV else {}
page_rank_map = load_lookup_map(PAGE_RANK_CSV) if PAGE_RANK_CSV else {}
stat_report_map = load_lookup_map(STAT_REPORT_CSV) if STAT_REPORT_CSV else {}

# ---------- Selenium helper (lazy init) ----------
def setup_selenium_headless():
    if webdriver is None:
        raise RuntimeError("Selenium/webdriver-manager/bs4 not installed")
    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-extensions")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(20)
    return driver

# ---------- Low-level extractors ----------
def basic_static_from_url(url):
    """Return a dict of many static/lexical features extracted directly from URL."""
    parsed = urlparse(url if "://" in url else "http://" + url)
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = (parsed.netloc + parsed.path + (("?" + parsed.query) if query else "")).lower()
    hostname = parsed.hostname or ""
    digits = len(re.findall(r'\d', full))
    host_parts = [p for p in hostname.split('.') if p]
    feats = {}

    # assign features (these names follow your dataset naming; ensure they match exactly)
    feats["url"] = url
    feats["length_url"] = len(full)
    feats["length_hostname"] = len(hostname)
    feats["ip"] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0
    feats["nb_dots"] = full.count('.')
    feats["nb_hyphens"] = full.count('-')
    feats["nb_at"] = full.count('@')
    feats["nb_qm"] = full.count('?')
    feats["nb_and"] = full.count('&')
    feats["nb_or"] = full.count('|')
    feats["nb_eq"] = full.count('=')
    feats["nb_underscore"] = full.count('_')
    feats["nb_tilde"] = full.count('~')
    feats["nb_percent"] = full.count('%')
    feats["nb_slash"] = full.count('/')
    feats["nb_star"] = full.count('*')
    feats["nb_colon"] = full.count(':')
    feats["nb_comma"] = full.count(',')
    feats["nb_semicolumn"] = full.count(';')
    feats["nb_dollar"] = full.count('$')
    feats["nb_space"] = full.count(' ')
    feats["nb_www"] = 1 if 'www.' in hostname else 0
    feats["nb_com"] = full.count('.com')
    feats["nb_dslash"] = max(0, full.count('//') - 1)
    feats["http_in_path"] = 1 if 'http' in path else 0
    feats["https_token"] = 1 if 'https' in full else 0
    feats["ratio_digits_url"] = digits / max(len(full), 1)
    feats["ratio_digits_host"] = len(re.findall(r'\d', hostname)) / max(len(hostname), 1)
    feats["punycode"] = 1 if 'xn--' in hostname else 0
    feats["port"] = 1 if ':' in parsed.netloc and parsed.netloc.split(':')[-1].isdigit() else 0
    feats["tld_in_path"] = 1 if any(t in path for t in ['.com','.net','.org','.ru','.tk','.xyz']) else 0
    feats["tld_in_subdomain"] = 1 if len(host_parts) >= 3 and any(p in ['com','net','org'] for p in host_parts[:-2]) else 0
    feats["abnormal_subdomain"] = 1 if len(host_parts) > 3 else 0
    feats["nb_subdomains"] = max(0, len(host_parts)-2)
    feats["prefix_suffix"] = 1 if '-' in hostname else 0
    feats["random_domain"] = 1 if (len(hostname) < 8 and sum(c.isdigit() for c in hostname) > 3) else 0

    shorteners = ['bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd']
    feats["shortening_service"] = 1 if any(s in hostname for s in shorteners) else 0
    feats["path_extension"] = 1 if re.search(r'\.(php|html|asp|aspx|exe|zip|rar|cgi)$', path) else 0

    # placeholders for redirections (dynamic, default 0)
    feats["nb_redirection"] = 0
    feats["nb_external_redirection"] = 0

    # word-level lexical features
    words = re.split(r'[\-_/\.]+', full)
    words = [w for w in words if w]
    feats["length_words_raw"] = sum(len(w) for w in words)
    feats["char_repeat"] = 1 if re.search(r'(.)\1\1', full) else 0
    feats["shortest_words_raw"] = min((len(w) for w in words), default=0)
    feats["shortest_word_host"] = min((len(p) for p in host_parts), default=0)
    path_parts = [p for p in path.split('/') if p]
    feats["shortest_word_path"] = min((len(w) for w in path_parts), default=0)
    feats["longest_words_raw"] = max((len(w) for w in words), default=0)
    feats["longest_word_host"] = max((len(p) for p in host_parts), default=0)
    feats["longest_word_path"] = max((len(w) for w in path_parts), default=0)
    feats["avg_words_raw"] = feats["length_words_raw"] / max(len(words), 1)
    feats["avg_word_host"] = sum(len(p) for p in host_parts) / max(len(host_parts), 1)
    feats["avg_word_path"] = sum(len(w) for w in path_parts) / max(len(path_parts), 1)

    # phish hints & brand placeholders
    phish_keywords = ['login','verify','account','secure','bank','update','confirm','signin','password','recovery']
    feats["phish_hints"] = sum(1 for k in phish_keywords if k in full)
    feats["domain_in_brand"] = 0
    feats["brand_in_subdomain"] = 0
    feats["brand_in_path"] = 0
    feats["suspecious_tld"] = 1 if any(t in hostname for t in ['.tk','.xyz','.top','.pw']) else 0

    # placeholders for features that may have been in training set but require external data
    feats["statistical_report"] = stat_report_map.get(hostname, 0) if 'stat_report_map' in globals() else 0

    return feats

def whois_features(domain):
    """Return whois-derived features; cached to reduce latency."""
    if domain in whois_cache:
        return whois_cache[domain]
    res = {"whois_registered_domain":0, "domain_registration_length":0, "domain_age":0}
    try:
        if whois:
            w = whois.whois(domain)
            res["whois_registered_domain"] = 1 if getattr(w, "domain_name", None) else 0
            # creation_date may be list or string
            try:
                cd = w.creation_date
                if w.creation_date:
                    cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                    res["domain_age"] = max(0, (pd.Timestamp.now() - pd.to_datetime(cd)).days)
                else:
                    res["domain_age"] = 0
                #if isinstance(cd, list):
                #    cd = cd[0]
                #if pd.notnull(cd):
                #    age_days = (pd.to_datetime('now') - pd.to_datetime(cd)).days
                #    res["domain_age"] = max(0, int(age_days))
            except Exception:
                res["domain_age"] = 0
            try:
                ed = w.expiration_date
                if isinstance(ed, list):
                    ed = ed[0]
                if pd.notnull(ed) and pd.notnull(cd):
                    res["domain_registration_length"] = max(0, int((pd.to_datetime(ed) - pd.to_datetime(cd)).days))
            except Exception:
                res["domain_registration_length"] = 0
    except Exception as e:
        # unable to do whois; default zeros
        res = {"whois_registered_domain":0, "domain_registration_length":0, "domain_age":0}
    whois_cache[domain] = res
    # Avoid massive outliers
    res["domain_registration_length"] = min(res["domain_registration_length"], 3650)
    res["domain_age"] = min(res["domain_age"], 3650)

    return res

def dns_features(domain):
    if domain in dns_cache:
        return dns_cache[domain]
    res = {"dns_record":0, "web_traffic":0}
    try:
        if dns:
            dns.resolver.resolve(domain, 'A')
            res["dns_record"] = 1
    except Exception:
        res["dns_record"] = 0
    # web_traffic/page_rank defaults come from maps if available
    res["web_traffic"] = web_traffic_map.get(domain, 0) if web_traffic_map else 0
    res["page_rank"] = page_rank_map.get(domain, 0) if page_rank_map else 0
    dns_cache[domain] = res
    return res

# ---------- Dynamic extraction (Selenium) ----------
def extract_dynamic_with_selenium(driver, url):
    dyn = {}
    try:
        driver.get(url if "://" in url else "http://" + url)
        time.sleep(0.5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        dyn["login_form"] = 1 if soup.find('input', {'type':'password'}) else 0
        dyn["iframe"] = 1 if soup.find('iframe') else 0
        link = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if link and link.get('href'):
            href = link.get('href')
            dyn["external_favicon"] = 1 if urlparse(href).netloc and urlparse(href).netloc != urlparse(driver.current_url).netloc else 0
        else:
            dyn["external_favicon"] = 0
        anchors = soup.find_all('a')
        total_links = len(anchors)
        ext_links = sum(1 for a in anchors if a.get('href') and urlparse(a.get('href')).netloc and urlparse(a.get('href')).netloc != urlparse(driver.current_url).netloc)
        dyn["nb_hyperlinks"] = total_links
        dyn["ratio_intHyperlinks"] = 0
        dyn["ratio_extHyperlinks"] = ext_links / max(total_links, 1)
        dyn["ratio_nullHyperlinks"] = sum(1 for a in anchors if not a.get('href')) / max(total_links,1)
        dyn["nb_extCSS"] = len(soup.find_all('link', {'rel':'stylesheet'}))
        # redirection detection (best-effort): check document.referrer vs current
        dyn["nb_redirection"] = 0
        dyn["nb_external_redirection"] = 0
        dyn["ratio_intRedirection"] = 0
        dyn["ratio_extRedirection"] = 0
        dyn["ratio_intErrors"] = 0
        dyn["ratio_extErrors"] = 0
        forms = soup.find_all('form')
        dyn["login_form"] = 1 if any(f.find('input', {'type':'password'}) for f in forms) else dyn.get("login_form", 0)
        dyn["links_in_tags"] = 1 if soup.find_all('a') else 0
        dyn["submit_email"] = 1 if any('mailto:' in (f.get('action') or '') for f in forms) else 0
        media = soup.find_all(['img','video','audio'])
        dyn["ratio_intMedia"] = 0
        dyn["ratio_extMedia"] = 0
        dyn["sfh"] = 0
        dyn["iframe"] = dyn.get("iframe", 0)
        dyn["popup_window"] = 1 if any('window.open' in (s.string or '') for s in soup.find_all('script') if s.string) else 0
        dyn["safe_anchor"] = sum(1 for a in anchors if not a.get('href') or a.get('href').startswith('#') or 'javascript' in (a.get('href') or '')) / max(total_links,1) if total_links>0 else 0
        dyn["onmouseover"] = 1 if any('onmouseover' in str(tag) for tag in soup.find_all(True)) else 0
        dyn["right_clic"] = 1 if any('contextmenu' in str(tag) for tag in soup.find_all(True)) else 0
        dyn["empty_title"] = 1 if not soup.title or not soup.title.string else 0
        dyn["domain_in_title"] = 1 if (soup.title and hostname_in_text(soup.title.string, url)) else 0
        dyn["domain_with_copyright"] = 0
        dyn["whois_registered_domain"] = 0
        dyn["domain_registration_length"] = 0
        dyn["domain_age"] = 0
        dyn["web_traffic"] = web_traffic_map.get(urlparse(url).netloc, 0) if web_traffic_map else 0
        dyn["dns_record"] = 1 if dns else 0
    except Exception as e:
        # fill defaults if extraction fails
        dyn = {k:0 for k in ["login_form","iframe","external_favicon","nb_hyperlinks","ratio_intHyperlinks",
                             "ratio_extHyperlinks","ratio_nullHyperlinks","nb_extCSS","nb_redirection",
                             "nb_external_redirection","ratio_intRedirection","ratio_extRedirection",
                             "ratio_intErrors","ratio_extErrors","links_in_tags","submit_email",
                             "ratio_intMedia","ratio_extMedia","sfh","popup_window","safe_anchor",
                             "onmouseover","right_clic","empty_title","domain_in_title","domain_with_copyright",
                             "whois_registered_domain","domain_registration_length","domain_age","web_traffic","dns_record"]}
        dyn["note"] = str(e)
    return dyn

# helper used above
def hostname_in_text(text, url):
    try:
        host = urlparse(url).netloc
        return host.split(':')[0] in (text or "")
    except:
        return False

# ---------- Unified extractor that builds full ordered vector ----------
def extract_features_full(url, dynamic=False, selenium_driver=None):
    """
    Compute full feature dict and return DataFrame with columns in feature_names order.
    dynamic: whether to attempt Selenium-based dynamic extraction (slower).
    selenium_driver: optional driver to reuse between calls.
    """
    s = basic_static_from_url(url)
    domain = urlparse(url).netloc.split(':')[0]
    # whois/dns
    w = whois_features(domain)
    d = dns_features(domain)
    dyn = {}
    if dynamic:
        try:
            driver = selenium_driver or setup_selenium_headless()
            dyn = extract_dynamic_with_selenium(driver, url)
            if selenium_driver is None and driver:
                driver.quit()
        except Exception as e:
            logging.warning("Dynamic extraction failed: %s", str(e))
            dyn = {}
    # assemble
    full = {}
    for name in feature_names:
        # priority mapping: s -> dyn -> w -> d -> defaults
        if name in s:
            full[name] = s[name]
        elif name in dyn:
            full[name] = dyn[name]
        elif name in w:
            full[name] = w[name]
        elif name in d:
            full[name] = d[name]
        else:
            # some columns in your dataset may exactly match keys we used earlier
            full[name] = 0
    # ensure types and no NaNs
    for k in full:
        if pd.isna(full[k]):
            full[k] = 0
    df = pd.DataFrame([full], columns=feature_names)
    return df

# ---------- Predict endpoint ----------
@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    try:
        data = request.get_json(force=True)
        url = data.get("url")
        dynamic = data.get("dynamic", ENABLE_DYNAMIC_BY_DEFAULT)
        if not url:
            return jsonify({"error":"missing url"}), 400

        # Optionally create a selenium driver once if dynamic True
        driver = None
        if dynamic:
            try:
                driver = setup_selenium_headless()
            except Exception as e:
                logging.warning("Cannot start Selenium: %s", str(e))
                driver = None

        X = extract_features_full(url, dynamic=dynamic, selenium_driver=driver)
        missing = X.columns[X.isna().any()].tolist()
        if missing:
            logging.info("[DEBUG] Missing features for %s: %s", url, missing)
        else:
            logging.info("[DEBUG] No missing features for %s", url)

        # ensure same order and fillna
        X = X.fillna(0)[feature_names]

        # Model expects numeric types; convert when possible
        try:
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])
        except Exception as e:
            logging.exception("Prediction failed")
            return jsonify({"error": f"prediction error: {str(e)}"}), 500

        latency_ms = (time.time() - start) * 1000
        logging.info("URL=%s pred=%s prob=%.3f lat_ms=%.1f", url, pred, prob, latency_ms)
        print("[DEBUG] Features for", url)
        print(X.to_dict(orient='records')[0])
        
        safe_domains = ["google.com", "apple.com", "amazon.com", "microsoft.com", "bankofamerica.com"]
        domain = urlparse(url).netloc.lower()
        if any(sd in domain for sd in safe_domains):
            prob = max(0.01, prob * 0.5)  # Lower phishing probability for known safe sites

        return jsonify({"prediction": pred, "probability": round(prob, 2), "latency_ms": round(latency_ms, 2)})

    except Exception as e:
        logging.exception("Predict exception")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
