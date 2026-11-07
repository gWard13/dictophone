"""Compile pure Kotlin with an explicit local JRE/compiler, compare outputs to Python fixtures."""
from pathlib import Path
import argparse,subprocess,json,os,hashlib

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--java',type=Path,required=True)
    p.add_argument('--kotlin-home',type=Path,required=True)
    args=p.parse_args()
    port=ROOT/'android-port';build=port/'build';build.mkdir(exist_ok=True)
    sources=list((port/'kotlin').glob('*.kt'))+[build/'ParityRunner.kt']
    compiler=[str(args.java),'-Dfile.encoding=UTF-8','-Dstdout.encoding=UTF-8','-Dstderr.encoding=UTF-8','-cp',str(args.kotlin_home/'lib/*'),
              'org.jetbrains.kotlin.cli.jvm.K2JVMCompiler','-kotlin-home',str(args.kotlin_home),
              '-jvm-target','17','-include-runtime','-d',str(build/'port-tests.jar'),*map(str,sources)]
    r=subprocess.run(compiler,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if r.returncode:
        print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
    actual=subprocess.run([str(args.java),'-Dfile.encoding=UTF-8','-Dstdout.encoding=UTF-8','-Dstderr.encoding=UTF-8','-jar',str(build/'port-tests.jar')],
                           capture_output=True,text=True,encoding='utf-8',check=True)
    outputs={r['id']:r['result'] for r in map(json.loads,actual.stdout.splitlines())}
    fixtures=json.loads((port/'fixtures.json').read_text(encoding='utf-8'))
    failures=[{'id':c['id'],'expected':c['expected'],'actual':outputs.get(c['id'])}
              for c in fixtures['cases'] if outputs.get(c['id'])!=c['expected']]
    report={'scope':'pure Kotlin JVM17 versus Python reference; NOT Android execution',
            'fixtures':len(fixtures['cases']),'passed':len(fixtures['cases'])-len(failures),'failed':failures,
            'unicode_version':fixtures['unicode_version'],
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.parent.name=='kotlin'},
            'native_inference_on_android_tested':False}
    (port/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
