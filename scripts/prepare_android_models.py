"""Download publisher-converted whisper.cpp weights; never rename CTranslate2 weights."""
from pathlib import Path
import argparse,hashlib,json
from prepare_models import download

REVISION='5359861c739e955e79d9a303bcbc70fb988958b1'
MODELS={
 'q5_1':('ggml-small-q5_1.bin',190085487,'ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb'),
 'f16':('ggml-small.bin',487601967,'1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b')}
def main():
    p=argparse.ArgumentParser(description='ONLINE preparation only; no audio is read or sent')
    p.add_argument('--variant',choices=['q5_1','f16','both'],default='q5_1')
    p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'models/android')
    args=p.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    files=[]
    for key in (list(MODELS) if args.variant=='both' else [args.variant]):
        name,size,digest=MODELS[key];path=args.directory/name
        url=f'https://huggingface.co/ggerganov/whisper.cpp/resolve/{REVISION}/{name}'
        if not path.exists():download(url,path)
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if path.stat().st_size!=size or actual!=digest:raise ValueError('Model checksum or size mismatch: '+name)
        files.append({'file':name,'variant':key,'format':'whisper.cpp custom GGML','bytes':size,'sha256':actual,'source_url':url})
        print('Verified '+name)
    (args.directory/'manifest.json').write_text(json.dumps({'source_revision':REVISION,'files':files},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
