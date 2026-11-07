"""Export synthetic cross-language fixtures and exact Python Unicode behavior."""
from pathlib import Path
from dataclasses import asdict
import argparse,inspect,json,hashlib,unicodedata
from dictophone.core import Word,Turn,Utterance,asr_windows,choose_speaker,align,normalized_words,guarded_text
from dictophone.pipeline import add_unintelligible
from dictophone.refinement import refine_labels

ROOT=Path(__file__).resolve().parents[1]
PORT=ROOT/'android-port'
def literal(value):
    return json.dumps(value,ensure_ascii=False).replace('$','\\$')
def turn(t): return f"Turn({t['start_ms']}L,{t['end_ms']}L,{literal(t['speaker'])})"
def word(w): return f"Word({w['start_ms']}L,{w['end_ms']}L,{literal(w['text'])})"
def row(r): return f"Utterance({literal(r['speaker_id'])},{r['start_ms']}L,{r['end_ms']}L,{literal(r['text'])})"
def items(values,fn,kind): return f"listOf<{kind}>("+','.join(fn(v) for v in values)+')'
def write(name,value):
    p=PORT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value,encoding='utf-8',newline='\n')

def export():
    # Pin Python's code-point semantics, rather than depending on Android's ICU version.
    ranges=[];start=None
    for cp in range(0x110000):
        yes=chr(cp).isalnum()
        if yes and start is None:start=cp
        if not yes and start is not None:ranges.extend([start,cp-1]);start=None
    if start is not None:ranges.extend([start,0x10ffff])
    folds={cp:chr(cp).casefold() for cp in range(0x110000) if chr(cp).casefold()!=chr(cp)}
    spaces=[cp for cp in range(0x110000) if chr(cp).isspace()]
    code='''package org.dictophone.port

/** Generated from the Python reference; regenerate with scripts/export_android_port.py. */
internal object UnicodeCompat {
'''
    code+='    const val unicodeVersion='+literal(unicodedata.unidata_version)+'\n'
    code+='    private val ranges=intArrayOf('+','.join(map(str,ranges))+')\n'
    code+='    private val spaces=setOf('+','.join(map(str,spaces))+')\n'
    code+='    private val folds=mapOf('+','.join(f'{cp} to {literal(v)}' for cp,v in folds.items())+')\n'
    code+='''    fun isSpace(cp:Int)=cp in spaces
    fun isAlnum(cp:Int):Boolean {
        var left=0;var right=ranges.size/2-1
        while(left<=right) { val m=(left+right)/2
            if(cp<ranges[m*2]) right=m-1 else if(cp>ranges[m*2+1]) left=m+1 else return true
        }
        return false
    }
    fun caseFold(text:String):String=buildString {
        text.codePoints().forEach { cp -> val mapped=folds[cp];if(mapped==null) appendCodePoint(cp) else append(mapped) }
    }
    fun strip(text:String):String {
        val cps=text.codePoints().toArray();var a=0;var b=cps.size
        while(a<b && isSpace(cps[a])) a++
        while(b>a && isSpace(cps[b-1])) b--
        return String(cps,a,b-a)
    }
    fun collapseSpace(text:String):String=buildString {
        var pending=false
        text.codePoints().forEach { cp ->
            if(isSpace(cp)) { if(isNotEmpty()) pending=true }
            else { if(pending) append(' ');appendCodePoint(cp);pending=false }
        }
    }
}
'''
    write(Path('kotlin/UnicodeCompat.kt'),code)

    cases=[]
    def add(id,op,**data):cases.append({'id':id,'operation':op,'input':data})
    a={'start_ms':1000,'end_ms':2000,'speaker':'a'}
    b={'start_ms':2200,'end_ms':3000,'speaker':'b'}
    for name,turns,duration in [('empty',[],5000),('single',[a],5000),('switch',[a,b],5000),
        ('same-pause',[a,dict(b,speaker='a')],5000),('overlap',[a,dict(b,start_ms=1500)],5000),
        ('zero',[dict(a,end_ms=1000)],5000),('invalid',[dict(a,end_ms=900)],5000),
        ('edges',[dict(a,start_ms=0),dict(b,start_ms=4500,end_ms=5000)],5000)]:
        add('window-'+name,'windows',turns=turns,duration_ms=duration)
    for name,words,turns,duration in [
        ('empty',[],[],5000),
        ('merge',[Word(1100,1300,'Привет'),Word(1400,1600,'друг')],[Turn(**a)],5000),
        ('switch',[Word(1100,1300,'Да'),Word(2300,2600,'Нет')],[Turn(**a),Turn(**b)],5000),
        ('overlap',[Word(1700,1900,'Да')],[Turn(**a),Turn(1500,2500,'b')],5000),
        ('unknown',[Word(4000,4200,'Да')],[Turn(**a)],5000),
        ('duplicate',[Word(1700,1900,'Да')],[Turn(**a),Turn(1500,2500,'a')],5000),
        ('zero-word',[Word(1200,1200,'Да')],[Turn(**a)],5000),
        ('long-pause',[Word(1100,1200,'Да'),Word(2300,2400,'Нет')],[Turn(0,5000,'a')],5000),
        ('invalid',[Word(1000,500,'Да')],[],5000)]:
        add('align-'+name,'align',words=[asdict(w) for w in words],turns=[asdict(t) for t in turns],duration_ms=duration)
    for i,text in enumerate(['Привет, мир!','Straße STRASSE','Σ σ ς','İ I ı','по-русски isn’t don\'t',
                              'Ёж ёж 12,5','a_b a—b','\U00010400\U00010428 ½ ²','текст\u00a0ещё','a\u0301b']):
        add(f'normalize-{i}','normalize',text=text)
    for i,(original,candidate) in enumerate([
        ('привет как дела','Привет! Как дела?'),('Straße','STRASSE'),('Σ','ς'),
        ('привет','Привет друг'),('два слова','слова два'),('число 12','число 13'),
        ('да','да <>'),('да',''),('да',None),('да нет','\u00a0Да\tнет!\u00a0'),
        ('да','да'+'.'*100),('по-русски','по русски'),('\U00010400','\U00010428')]):
        add(f'guard-{i}','guard',original=original,candidate=candidate)
    for i,(word_,turns) in enumerate([
        (Word(400,700,'да'),[Turn(0,500,'a'),Turn(500,1000,'b')]),
        (Word(400,700,'да'),[Turn(0,1000,'a'),Turn(300,800,'b')]),
        (Word(400,700,'да'),[Turn(0,1000,'a'),Turn(300,800,'a')]),
        (Word(400,700,'да'),[]),
        (Word(400,600,'да'),[Turn(0,500,'speaker_2'),Turn(500,1000,'speaker_10')])]):
        add(f'choose-{i}','choose',word=asdict(word_),turns=[asdict(t) for t in turns])
    base=[Turn(0,5000,'a'),Turn(6000,11000,'b'),Turn(12000,13000,'b')]
    for i,vectors in enumerate([[[1,0],[0,1],[1,0]],[[1,0],[0.99,0.01],[1,0]],[[1,0],[0,1],None],[[0,0],[0,1],[1,0]]]):
        add(f'refine-{i}','refine',turns=[asdict(t) for t in base],embeddings=vectors)
    for i,rows in enumerate([[],[Utterance('speaker_1',0,1000,'Он сказал: "Да".\nДалее. 😀')],
                             [Utterance('unknown',0,0,'\u0001\tда')],[Utterance('bad',0,100,'да')]]):
        add(f'export-{i}','export',rows=[asdict(r) for r in rows],duration_ms=1000)
    for i,(words,turns) in enumerate([([],[]),([],[Turn(0,1000,'a')]),
        ([Word(100,200,'да')],[Turn(0,1000,'a')]),([],[Turn(0,199,'a')])]):
        add(f'missing-{i}','missing',words=[asdict(w) for w in words],turns=[asdict(t) for t in turns],duration_ms=5000)

    runner=['package org.dictophone.port','',
      '''fun encoded(x:Any?):String=when(x) {
        null->"null"
        is String->AudioPipelineLogic.jsonQuote(x)
        is Number,is Boolean->x.toString()
        is Map<*,*>->x.entries.joinToString(",","{","}") { encoded(it.key.toString())+":"+encoded(it.value) }
        is Iterable<*>->x.joinToString(",","[","]") { encoded(it) }
        else->error("Unsupported test output")
      }
      fun rows(values:List<Utterance>)=values.map { mapOf("speaker_id" to it.speakerId,"start_ms" to it.startMs,"end_ms" to it.endMs,"text" to it.text) }
      fun turns(values:List<Turn>)=values.map { mapOf("start_ms" to it.startMs,"end_ms" to it.endMs,"speaker" to it.speaker) }
      fun words(values:List<Word>)=values.map { mapOf("start_ms" to it.startMs,"end_ms" to it.endMs,"text" to it.text) }
      fun main() {''']
    for case in cases:
        d=case['input'];op=case['operation']
        ts=[Turn(**v) for v in d.get('turns',[])];ws=[Word(**v) for v in d.get('words',[])]
        kt=items(d.get('turns',[]),turn,'Turn');kw=items(d.get('words',[]),word,'Word')
        try:
            if op=='windows':
                expr=f'AudioPipelineLogic.asrWindows({kt},{d["duration_ms"]}L).map {{ listOf(it.startMs,it.endMs) }}'
                result=[list(x) for x in asr_windows(ts,d['duration_ms'])]
            elif op=='align':
                expr=f'rows(AudioPipelineLogic.align({kw},{kt},{d["duration_ms"]}L))'
                result=[asdict(x) for x in align(ws,ts,d['duration_ms'])]
            elif op=='normalize':result=normalized_words(d['text']);expr=f'AudioPipelineLogic.normalizedWords({literal(d["text"])})'
            elif op=='guard':
                expr=f'AudioPipelineLogic.guardedText({literal(d["original"])},{"null" if d["candidate"] is None else literal(d["candidate"])})'
                result=guarded_text(d['original'],d['candidate'])
            elif op=='choose':result=choose_speaker(Word(**d['word']),ts);expr=f'AudioPipelineLogic.chooseSpeaker({word(d["word"])},{kt})'
            elif op=='refine':
                kv='listOf<DoubleArray?>('+','.join('null' if v is None else 'doubleArrayOf('+','.join(str(float(x)) for x in v)+')' for v in d['embeddings'])+')'
                expr=f'AudioPipelineLogic.refineLabels({kt},{kv}).let {{ mapOf("turns" to turns(it.turns),"changes" to it.changes) }}'
                out,count=refine_labels(ts,d['embeddings']);result={'turns':[asdict(t) for t in out],'changes':count}
            elif op=='export':
                from dictophone.core import validate_transcript
                expr=f'AudioPipelineLogic.toJson({items(d["rows"],row,"Utterance")},{d["duration_ms"]}L)'
                result={'transcript':d['rows']};validate_transcript(result,d['duration_ms'])
                result=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
            elif op=='missing':
                result=[asdict(w) for w in add_unintelligible(ws,ts,d['duration_ms'])]
                expr=f'words(AudioPipelineLogic.addUnintelligible({kw},{kt}))'
            case['expected']=result
        except (ValueError,TypeError):case['expected']={'error':True}
        runner.append('run { val result:Any?=try { '+expr+' } catch(e:Exception) { mapOf("error" to true) }; println(encoded(mapOf("id" to '+literal(case['id'])+',"result" to result))) }')
    expected_rows=[asdict(r) for r in align([Word(100,400,'привет'),Word(1200,1500,'привет')],
                         [Turn(0,900,'a'),Turn(1100,2000,'b')],2000)]
    for r in expected_rows:r['text']=guarded_text(r['text'],'Привет!')
    cases.append({'id':'pipeline-native-contract','operation':'native_interface_simulation',
       'input':{'description':'Synthetic adapters, no actual speech models'},
       'expected':{'rows':expected_rows,'accepted':2,'warnings':[],
                   'order':['diarizer_open','diarizer_close','asr_open','asr_close','llm_open','llm_close']}})
    runner.append('''run {
      val events=mutableListOf<String>()
      val pipeline=AudioPipeline(
        asrFactory={ events.add("asr_open");object:OfflineAsr {
          override fun transcribe(pcm16kMono:FloatArray,language:String)=listOf(Word(100,400,"привет"))
          override fun close() { events.add("asr_close") }
        } },
        diarizerFactory={ events.add("diarizer_open");object:OfflineDiarizer {
          override fun diarize(pcm16kMono:FloatArray)=listOf(Turn(0,900,"a"),Turn(1100,2000,"b"))
          override fun close() { events.add("diarizer_close") }
        } },
        textFactory={ events.add("llm_open");object:OfflineTextModel {
          override fun punctuate(text:String)="Привет!"
          override fun close() { events.add("llm_close") }
        } },settings=PipelineSettings(refineShortTurns=false,padMs=0))
      val r=pipeline.runDecoded(FloatArray(32000),"ru")
      println(encoded(mapOf("id" to "pipeline-native-contract","result" to mapOf(
        "rows" to rows(r.transcript),"accepted" to r.acceptedLlmTexts,"warnings" to r.warnings,"order" to events))))
    }''')
    runner.append('}')
    build=PORT/'build';build.mkdir(exist_ok=True)
    (build/'ParityRunner.kt').write_text('\n'.join(runner),encoding='utf-8')
    write(Path('fixtures.json'),json.dumps({'origin':'synthetic algorithm tests, not model transcripts','unicode_version':unicodedata.unidata_version,'cases':cases},ensure_ascii=False,indent=2)+'\n')
    params=json.loads((ROOT/'config/pipeline.short-turns.json').read_text(encoding='utf-8'))
    for section in params:
        if isinstance(params[section],dict):
            params[section]={k:v for k,v in params[section].items() if k not in ['model_dir','model_path','segmentation_model','embedding_model']}
    params.update({'pcm':{'sample_rate_hz':16000,'channels':1,'sample_format':'float32','samples_per_ms':16},
      'speaker_assignment':{'min_overlap_ms':40,'min_overlap_ratio':0.2},
      'missing_speech_marker':{'min_duration_ms':200},
      'time_rounding':'Python round: ties to even; milliseconds integer',
      'unicode_version':unicodedata.unidata_version,
      'target_requested':{'profile':'Pixel 10 emulator','api':37,'ram_mib':4096,'abi_requested':'arm64-v8a','abi_verified':False},
      'android_proposed_only':{'initial_threads':2,'llm_context_tokens':1024,'sequential_model_loading':True,'measured_on_android':False},
      'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/dictophone/core.py',ROOT/'src/dictophone/refinement.py',ROOT/'src/dictophone/backends.py',ROOT/'src/dictophone/pipeline.py']}})
    write(Path('parameters.json'),json.dumps(params,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'fixtures':len(cases),'unicode_version':unicodedata.unidata_version,'casefold_entries':len(folds),'alnum_ranges':len(ranges)//2}))

if __name__=='__main__':export()
