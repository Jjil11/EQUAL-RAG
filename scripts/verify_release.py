"""Publication checks; do not print matching sensitive content."""
import ast,hashlib,json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BAD_BINARY={'.pt','.bin','.safetensors','.ckpt','.pem','.key','.zip','.gz','.tar','.log'}
PATTERNS={
    'private IPv4 address':r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b',
    'common service token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,})\b',
    'private key header':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'literal credential assignment':r'(?i)\b(?:password|passwd|api_key|access_token)\s*=\s*[\x22\x27][^\x22\x27\r\n]{6,}[\x22\x27]',
    'original machine path':r'/(?:home|common_data)/liang'+'jj(?:/|\b)',
}

def files():
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(part in {'.git','__pycache__','.venv','workspace'} for part in rel.parts):continue
        if p.is_file():yield p

def verify():
    errors=[];count=0;syntax=0
    manifest=json.loads((ROOT/'release_manifest.json').read_text(encoding='utf-8'))
    for category in ['sources','results']:
        for rel,record in manifest[category].items():
            p=ROOT/rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['release_sha256']:
                errors.append(rel+': release hash mismatch')
            count+=1
    publication=ROOT/'publication_20261017.json'
    if publication.exists():
        for record in json.loads(publication.read_text(encoding='utf-8'))['files']:
            p=ROOT/record['path']
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
                errors.append(record['path']+': 20261017 publication hash mismatch')
            count+=1
    for p in files():
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix in BAD_BINARY or p.name in {'.env','id_rsa','id_ed25519'}:errors.append(rel+': forbidden artifact type')
        if p.stat().st_size>10_000_000:errors.append(rel+': file too large for this source-only release')
        try:text=p.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:errors.append(rel+': unreviewed non-text file');continue
        for label,pattern in PATTERNS.items():
            if re.search(pattern,text):errors.append(rel+': '+label)
        if p.suffix=='.py':
            try:ast.parse(text);syntax+=1
            except SyntaxError:errors.append(rel+': invalid Python syntax')
        if p.suffix=='.json' and rel.startswith('results/'):
            def inspect_json(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if key in {'prompt','output','short','chosen','rejected','aliases','docs','documents','premise','claim','naive_ids','joint_ids','naive_boundary_window','joint_boundary_window','prompt_token_ids','generated_token_ids'}:
                            errors.append(rel+': unexpected raw-example field '+key)
                        inspect_json(item)
                elif isinstance(value,list):
                    for item in value:inspect_json(item)
            inspect_json(json.loads(text))
        if p.suffix=='.md':
            for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',text):
                target=target.split('#',1)[0].strip('<>')
                if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):continue
                if not (p.parent/target).exists():errors.append(rel+': broken local Markdown link '+target)
    return errors,count,syntax

def main():
    errors,count,syntax=verify()
    print(json.dumps({'status':'FAIL' if errors else 'PASS','manifest_files_verified':count,'python_sources_parsed':syntax,'errors':errors},ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
