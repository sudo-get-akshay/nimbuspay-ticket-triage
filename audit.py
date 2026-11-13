"""Training-only, evidence-based repairs. Never used during inference."""
import argparse
import collections
import copy
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

KEYS = ['category', 'priority', 'amount_paise', 'txn_date', 'txn_id', 'channel', 'language', 'needs_human']
NULL_CATEGORIES = {'account_access', 'kyc', 'other'}
CATEGORIES = NULL_CATEGORIES | {'payment_failed', 'refund', 'fraud', 'offers'}
LEGAL = re.compile(r'\b(?:RBI|ombudsman|consumer court|lawyer|legal (?:notice|action))\b', re.I)
REPEAT = re.compile(r'third time|second complaint|already written \d+ times|contacted support about this before|again and again|teesri baar|pehle bhi complaint|baar baar complaint', re.I)
NUM = r'\d[\d,]*(?:\.\d+)?'
UNIT = r'(?:lakhs?|k|rupees|rupaye)'
MONEY = rf'(?:Rs\.?\s*|INR\s*|₹\s*)?{NUM}(?:\s*{UNIT}\b)?'
MONEY_PATTERNS = [
    re.compile(rf'(?:Rs\.?\s*|INR\s*|₹\s*)(?P<n>{NUM})(?:\s*(?P<u>{UNIT})\b)?', re.I),
    re.compile(rf'(?<![\w.])(?P<n>{NUM})\s*(?P<u>{UNIT})\b', re.I),
    re.compile(rf'(?:amount\s*(?:is|was|:)?|total|rakam)\s*(?P<n>{NUM})(?:\s*(?P<u>{UNIT})\b)?', re.I),
    re.compile(rf'(?<![\w.])(?P<n>{NUM})(?:\s*(?P<u>{UNIT})\b)?\s+(?:in total|was the value|is the amount)', re.I),
]
MONTHS = {name.lower(): i for i, names in enumerate([
    ('Jan', 'January'), ('Feb', 'February'), ('Mar', 'March'), ('Apr', 'April'),
    ('May', 'May'), ('Jun', 'June'), ('Jul', 'July'), ('Aug', 'August'),
    ('Sep', 'September'), ('Oct', 'October'), ('Nov', 'November'), ('Dec', 'December')], 1) for name in names}
MONTH_RE = '|'.join(sorted(MONTHS, key=len, reverse=True))


def load(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def dump_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def customer_body(ticket):
    body = ticket.split('\n\n', 1)[-1]
    # A quoted-email boundary starts an earlier thread; it is not current evidence.
    body = re.split(r'(?m)^\s*>', body, maxsplit=1)[0]
    return body


def amount_evidence(body):
    text = re.sub(r'My account balance is .*? right now\.|Account mein abhi .*? pade hain\.', '', body, flags=re.I)
    text = re.sub(rf'FYI my daily limit is {MONEY}\.', '', text, flags=re.I)
    matches = []
    for pattern in MONEY_PATTERNS:
        for m in pattern.finditer(text):
            unit = (m.groupdict().get('u') or '').lower()
            value = Decimal(m['n'].replace(',', ''))
            factor = 100_000 if unit.startswith('lakh') else 1_000 if unit == 'k' else 1
            matches.append((m.start(), m.end(), int(value * factor * 100), m.group()))
    # Prefer longest overlapping match so '1.5 lakh' is not read as 1.5 rupees.
    selected = []
    for m in sorted(matches, key=lambda x: -(x[1] - x[0])):
        if not any(m[0] < s[1] and m[1] > s[0] for s in selected):
            selected.append(m)
    values = {m[2] for m in selected}
    return (next(iter(values)) if len(values) == 1 else None), selected


def date_evidence(ticket, body):
    received = dt.date.fromisoformat(re.search(r'Received:\s*(\d{4}-\d{2}-\d{2})', ticket)[1])
    values = set()
    for m in re.finditer(r'\b(\d+)\s*(?:days? ago|din pehle)\b', body, re.I):
        values.add((received - dt.timedelta(days=int(m[1]))).isoformat())
    for pattern, days in [(r'\b(?:day before yesterday|parso)\b', 2), (r'\b(?:yesterday|kal)\b', 1), (r'\b(?:today|aaj)\b', 0)]:
        # Prevent matching 'yesterday' inside 'day before yesterday'.
        text = re.sub(r'\bday before yesterday\b', '', body, flags=re.I) if days == 1 else body
        if re.search(pattern, text, re.I):
            values.add((received - dt.timedelta(days=days)).isoformat())

    def add(day, month, year=None):
        try:
            yr = int(year) if year else received.year
            if yr < 100:
                yr += 2000
            date = dt.date(yr, int(month), int(day))
            if not year and date > received:
                date = date.replace(year=yr - 1)
            values.add(date.isoformat())
        except ValueError:
            pass

    for m in re.finditer(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2}|\d{4})\b', body):
        add(m[1], m[2], m[3])
    for m in re.finditer(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})(?:\s+(\d{{4}}))?\b', body, re.I):
        add(m[1], MONTHS[m[2].lower()], m[3])
    for m in re.finditer(rf'\b({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b', body, re.I):
        add(m[2], MONTHS[m[1].lower()], m[3])
    return values


def priority(category, amount, body):
    if category == 'fraud' or (amount is not None and amount >= 5_000_000) or LEGAL.search(body):
        return 'P1'
    if category == 'account_access' or (category in {'refund', 'payment_failed'} and amount is not None and amount >= 500_000):
        return 'P2'
    return 'P3'


def valid_label(obj):
    if not isinstance(obj, dict) or set(obj) != set(KEYS):
        return False
    return (obj['category'] in CATEGORIES and obj['priority'] in {'P1','P2','P3'}
            and (obj['amount_paise'] is None or type(obj['amount_paise']) is int)
            and (obj['txn_date'] is None or isinstance(obj['txn_date'], str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', obj['txn_date']) is not None)
            and (obj['txn_id'] is None or isinstance(obj['txn_id'], str) and re.fullmatch(r'NP\d{10}', obj['txn_id']) is not None)
            and obj['channel'] in {None, 'upi', 'card', 'netbanking', 'wallet'}
            and obj['language'] in {'en', 'hi-en'} and type(obj['needs_human']) is bool)


def audit(train_path, output_dir):
    path, out = Path(train_path), Path(output_dir)
    if path.name != 'train.jsonl':
        raise ValueError('Cleaner only accepts the provided train.jsonl')
    rows = load(path)
    if any(not r['id'].startswith('tr-') for r in rows):
        raise ValueError('Non-training id detected')
    out.mkdir(parents=True, exist_ok=True)
    cleaned, log, evidence, review = [], [], [], []
    seen, counts = {}, collections.Counter()
    for row in rows:
        r = copy.deepcopy(row)
        ticket = r['messages'][1]['content']
        body = customer_body(ticket)
        if re.fullmatch(r'\s*(?:test(?: test 123)?|asdf asdf)?\s*', body, re.I):
            log.append({'id':r['id'], 'action':'drop', 'reason':'empty_or_placeholder_customer_message; label_has_no_evidence'})
            counts['placeholder_ticket'] += 1
            continue
        signature = tuple(m['content'] for m in r['messages'][:2])
        if signature in seen:
            log.append({'id': r['id'], 'action': 'drop', 'reason': f'exact_duplicate_of:{seen[signature]}'})
            counts['duplicate'] += 1
            continue
        seen[signature] = r['id']
        try:
            original = json.loads(r['messages'][-1]['content'])
        except (ValueError, TypeError):
            log.append({'id': r['id'], 'action': 'drop', 'reason': 'truncated_or_invalid_assistant_json; no invented replacement label'})
            counts['invalid_json'] += 1
            continue
        if not isinstance(original, dict) or original.get('category') not in CATEGORIES:
            log.append({'id': r['id'], 'action': 'drop', 'reason': 'unsupported_label_structure_or_category'})
            counts['unsupported_label'] += 1
            continue
        label, reasons = copy.deepcopy(original), []
        if 'prio' in label:
            label['priority'] = {'low':'P3','medium':'P2','high':'P1'}.get(label.pop('prio'))
            reasons.append('migrate_legacy_prio_to_priority')
        if label.get('txn_date') and re.fullmatch(r'\d{2}/\d{2}/\d{4}', label['txn_date']):
            label['txn_date'] = dt.datetime.strptime(label['txn_date'], '%d/%m/%Y').date().isoformat()
            reasons.append('migrate_legacy_date_to_ISO')
        if label['category'] in NULL_CATEGORIES:
            for k in ['amount_paise','txn_date','txn_id','channel']:
                if label.get(k) is not None:
                    label[k] = None
                    reasons.append(f'null_transaction_field:{k}')
        else:
            amount, candidates = amount_evidence(body)
            if len({m[2] for m in candidates}) == 1 and label.get('amount_paise') != amount:
                old = label.get('amount_paise')
                label['amount_paise'] = amount
                reasons.append(f'amount_conversion:{old}->{amount}; evidence:{candidates[0][3]}')
            elif len({m[2] for m in candidates}) > 1:
                review.append({'id':r['id'], 'issue':'multiple_amount_candidates_retained', 'candidates':candidates})
            dates = date_evidence(ticket, body)
            if len(dates) == 1 and label.get('txn_date') not in dates:
                old = label.get('txn_date')
                label['txn_date'] = next(iter(dates))
                reasons.append(f'transaction_date_evidence:{old}->{label["txn_date"]}')
            if not dates and label.get('txn_date') is not None:
                review.append({'id':r['id'], 'issue':'label_date_not_recognized_in_current_body', 'value':label['txn_date']})
            if label.get('txn_id') is not None:
                norm = re.sub(r'[^A-Z0-9]', '', str(label['txn_id']).upper())
                if norm != label['txn_id'] and re.fullmatch(r'NP\d{10}', norm):
                    label['txn_id'] = norm
                    reasons.append('normalize_transaction_id')
        expected_priority = priority(label['category'], label.get('amount_paise'), body)
        if label.get('priority') != expected_priority:
            reasons.append(f'priority_business_rule:{label.get("priority")}->{expected_priority}')
            label['priority'] = expected_priority
        human = label['priority'] == 'P1' or REPEAT.search(body) is not None
        if type(label.get('needs_human')) is not bool or label.get('needs_human') != human:
            reasons.append(f'human_escalation_rule:{label.get("needs_human")}->{human}')
            label['needs_human'] = human
        if not valid_label(label):
            log.append({'id':r['id'], 'action':'drop', 'reason':'remaining_schema_violation'})
            counts['remaining_schema_violation'] += 1
            continue
        if reasons:
            log.append({'id':r['id'], 'action':'fix', 'reason':'; '.join(reasons)})
            evidence.append({'id':r['id'], 'before':original, 'after':label, 'reasons':reasons})
            for reason in reasons:
                counts[reason.split(':')[0]] += 1
        r['messages'][-1]['content'] = json.dumps({k:label[k] for k in KEYS}, ensure_ascii=False)
        cleaned.append(r)
    dump_jsonl(out / 'train_clean.jsonl', cleaned)
    with (out / 'cleaning_log.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id','action','reason'])
        writer.writeheader()
        writer.writerows(log)
    dump_json(out / 'cleaning_evidence.json', evidence)
    dump_json(out / 'audit_review.json', review)
    summary = {'input_rows':len(rows), 'retained_rows':len(cleaned),
               'dropped_rows':sum(r['action']=='drop' for r in log),
               'fixed_rows':sum(r['action']=='fix' for r in log),
               'findings':dict(counts), 'review_rows':len(review),
               'quoted_thread_rows':sum(bool(re.search(r'(?m)^\s*>', r['messages'][1]['content'])) for r in rows),
               'source_counts':dict(collections.Counter(re.search(r'Source:\s*(\S+)', r['messages'][1]['content'])[1] for r in rows)),
               'category_counts':dict(collections.Counter(json.loads(r['messages'][-1]['content'])['category'] for r in cleaned)),
               'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    dump_json(out / 'audit_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/train.jsonl')
    parser.add_argument('--out', default='artifacts')
    args = parser.parse_args()
    print(json.dumps(audit(args.train, args.out), indent=2))
