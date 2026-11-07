package org.dictophone.port

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class Word(val startMs: Long, val endMs: Long, val text: String)
data class Turn(val startMs: Long, val endMs: Long, val speaker: String)
data class Utterance(val speakerId: String, val startMs: Long, val endMs: Long, val text: String)
data class AsrWindow(val startMs: Long, val endMs: Long)
data class RefinementResult(val turns: List<Turn>, val changes: Int)

/** Pure Kotlin logic. Native inference and Android UI/storage are deliberately separate adapters. */
object AudioPipelineLogic {
    private fun valid(a: Long, b: Long, duration: Long) = a >= 0 && a <= b && b <= duration
    private fun scalarCompare(a: String, b: String): Int {
        val aa=a.codePoints().toArray(); val bb=b.codePoints().toArray()
        for (i in 0 until min(aa.size,bb.size)) if (aa[i]!=bb[i]) return aa[i].compareTo(bb[i])
        return aa.size.compareTo(bb.size)
    }
    private val turnOrder=Comparator<Turn> { a,b ->
        val x=a.startMs.compareTo(b.startMs)
        if(x!=0) x else { val y=a.endMs.compareTo(b.endMs); if(y!=0) y else scalarCompare(a.speaker,b.speaker) }
    }

    fun asrWindows(turns: List<Turn>, durationMs: Long, padMs: Long=500, mergeGapMs: Long=1500): List<AsrWindow> {
        require(padMs>=0 && mergeGapMs>=0)
        val grouped=mutableListOf<Turn>()
        for(t in turns.sortedWith(compareBy<Turn> { it.startMs }.thenBy { it.endMs })) {
            require(valid(t.startMs,t.endMs,durationMs))
            if(t.startMs==t.endMs) continue
            val p=grouped.lastOrNull()
            if(p!=null && (t.startMs<=p.endMs || (t.speaker==p.speaker && t.startMs-p.endMs<=mergeGapMs))) {
                grouped[grouped.lastIndex]=Turn(p.startMs,max(p.endMs,t.endMs),if(p.speaker==t.speaker) p.speaker else "overlap")
            } else grouped.add(t)
        }
        if(grouped.isEmpty()) return listOf(AsrWindow(0,durationMs))
        return grouped.mapIndexed { i,t ->
            val left=if(i>0) grouped[i-1].endMs+(t.startMs-grouped[i-1].endMs)/2 else 0
            val right=if(i+1<grouped.size) t.endMs+(grouped[i+1].startMs-t.endMs)/2 else durationMs
            AsrWindow(max(left,t.startMs-padMs),min(right,t.endMs+padMs))
        }
    }

    fun canonicalTurns(turns: List<Turn>, durationMs: Long): List<Turn> {
        val ids=linkedMapOf<String,String>()
        return turns.sortedWith(turnOrder).mapNotNull { t ->
            require(valid(t.startMs,t.endMs,durationMs) && t.speaker.isNotEmpty())
            if(t.startMs==t.endMs) null else t.copy(speaker=ids.getOrPut(t.speaker) { "speaker_${ids.size+1}" })
        }
    }

    fun chooseSpeaker(word: Word, turns: List<Turn>, minOverlapMs: Long=40, minOverlapRatio: Double=0.2): String {
        val start=word.startMs; val end=max(word.endMs,start+1)
        val events=sortedMapOf<Long,MutableList<Pair<String,Int>>>()
        for(t in turns) {
            val a=max(start,t.startMs); val b=min(end,t.endMs)
            if(b>a) {
                events.getOrPut(a) { mutableListOf() }.add(t.speaker to 1)
                events.getOrPut(b) { mutableListOf() }.add(t.speaker to -1)
            }
        }
        val active=mutableMapOf<String,Int>(); val coverage=mutableMapOf<String,Long>()
        var overlap=0L; var previous=start
        for((at,changes) in events) {
            val width=at-previous; val speakers=active.filterValues { it>0 }.keys
            for(s in speakers) coverage[s]=(coverage[s]?:0)+width
            if(speakers.size>1) overlap+=width
            for((s,delta) in changes) active[s]=(active[s]?:0)+delta
            previous=at
        }
        if(overlap>=minOverlapMs && overlap.toDouble()/(end-start)>=minOverlapRatio) return "overlap"
        if(coverage.isEmpty()) return "unknown"
        return coverage.keys.minWithOrNull(Comparator { a,b ->
            val score=coverage[b]!!.compareTo(coverage[a]!!)
            if(score!=0) score else scalarCompare(a,b)
        })!!
    }

    fun align(words: List<Word>, turns: List<Turn>, durationMs: Long,
              pauseMs: Long=700,maxUtteranceMs: Long=15000): List<Utterance> {
        val normalized=canonicalTurns(turns,durationMs)
        val result=mutableListOf<Utterance>()
        var cursor=0; var active=mutableListOf<Turn>()
        for(w in words.sortedWith(compareBy<Word> { it.startMs }.thenBy { it.endMs })) {
            require(valid(w.startMs,w.endMs,durationMs))
            val text=UnicodeCompat.strip(w.text)
            if(text.isEmpty()) continue
            while(cursor<normalized.size && normalized[cursor].startMs<max(w.endMs,w.startMs+1)) active.add(normalized[cursor++])
            active=active.filter { it.endMs>w.startMs }.toMutableList()
            val speaker=chooseSpeaker(w,active); val p=result.lastOrNull()
            if(p!=null && p.speakerId==speaker && w.startMs-p.endMs<=pauseMs && w.endMs-p.startMs<=maxUtteranceMs)
                result[result.lastIndex]=p.copy(endMs=max(p.endMs,w.endMs),text=p.text+" "+text)
            else result.add(Utterance(speaker,w.startMs,w.endMs,text))
        }
        return result
    }

    fun normalizedWords(text: String): List<String> {
        val cps=UnicodeCompat.caseFold(text).codePoints().toArray()
        val result=mutableListOf<String>(); var i=0
        while(i<cps.size) {
            if(!UnicodeCompat.isAlnum(cps[i])) { i++; continue }
            val word=StringBuilder()
            while(i<cps.size && UnicodeCompat.isAlnum(cps[i])) word.appendCodePoint(cps[i++])
            while(i+1<cps.size && cps[i] in intArrayOf(45,0x2019,39) && UnicodeCompat.isAlnum(cps[i+1])) {
                word.appendCodePoint(cps[i++])
                while(i<cps.size && UnicodeCompat.isAlnum(cps[i])) word.appendCodePoint(cps[i++])
            }
            result.add(word.toString())
        }
        return result
    }

    fun guardedText(original: String, candidate: String?): String {
        require(candidate!=null && UnicodeCompat.strip(candidate).isNotEmpty())
        require(normalizedWords(original)==normalizedWords(candidate)) { "LLM changed words" }
        require(candidate.codePointCount(0,candidate.length)<=max(64,original.codePointCount(0,original.length)*3))
        val allowed=".,!?;:…—–-()[]«»\"'’ \t\n\r".codePoints().toArray().toSet()
        require(candidate.codePoints().allMatch { UnicodeCompat.isAlnum(it)||UnicodeCompat.isSpace(it)||it in allowed })
        return UnicodeCompat.collapseSpace(candidate)
    }

    fun addUnintelligible(words: List<Word>, turns: List<Turn>): List<Word> {
        val regions=mutableListOf<AsrWindow>()
        for(t in turns.filter { it.endMs>it.startMs }.sortedWith(compareBy<Turn> { it.startMs }.thenBy { it.endMs })) {
            val p=regions.lastOrNull()
            if(p!=null && t.startMs<=p.endMs) regions[regions.lastIndex]=p.copy(endMs=max(p.endMs,t.endMs))
            else regions.add(AsrWindow(t.startMs,t.endMs))
        }
        return words+regions.filter { r -> r.endMs-r.startMs>=200 && words.none { it.startMs<r.endMs && it.endMs>r.startMs } }
            .map { Word(it.startMs,it.endMs,"[Неразборчиво]") }
    }

    fun validate(rows: List<Utterance>, durationMs: Long) {
        var previous=-1L
        for(r in rows) {
            require(Regex("speaker_[1-9][0-9]*|unknown|overlap").matches(r.speakerId))
            require(UnicodeCompat.strip(r.text).isNotEmpty() && valid(r.startMs,r.endMs,durationMs) && r.startMs>=previous)
            previous=r.startMs
        }
    }

    fun toJson(rows: List<Utterance>, durationMs: Long): String {
        validate(rows,durationMs)
        if(rows.isEmpty()) return "{\n  \"transcript\": []\n}\n"
        return "{\n  \"transcript\": [\n"+rows.joinToString(",\n") { r ->
            "    {\n      \"speaker_id\": ${jsonQuote(r.speakerId)},\n      \"start_ms\": ${r.startMs},\n      \"end_ms\": ${r.endMs},\n      \"text\": ${jsonQuote(r.text)}\n    }"
        }+"\n  ]\n}\n"
    }

    fun refineLabels(turns: List<Turn>, embeddings: List<DoubleArray?>,
                     maxDurationMs: Long=3000,minReferenceMs: Long=3500,
                     minSimilarity: Double=0.6,minMargin: Double=0.15): RefinementResult {
        require(turns.size==embeddings.size && maxDurationMs>0 && minReferenceMs>0)
        require(minSimilarity.isFinite() && minSimilarity in -1.0..1.0 && minMargin.isFinite() && minMargin in 0.0..2.0)
        var dimension:Int?=null
        val vectors=embeddings.map { v ->
            if(v==null) null else {
                require(v.isNotEmpty() && v.all { it.isFinite() })
                val norm=sqrt(v.sumOf { it*it }); require(norm>0)
                require(dimension==null || v.size==dimension); dimension=v.size
                DoubleArray(v.size) { v[it]/norm }
            }
        }
        val clean=BooleanArray(turns.size) { true }; var active=mutableListOf<Int>()
        for(i in turns.indices.sortedBy { turns[it].startMs }) {
            active=active.filter { turns[it].endMs>turns[i].startMs }.toMutableList()
            for(j in active) if(turns[j].speaker!=turns[i].speaker) { clean[i]=false;clean[j]=false }
            active.add(i)
        }
        val sums=mutableMapOf<String,DoubleArray>();val durations=mutableMapOf<String,Long>()
        for(j in turns.indices) {
            val t=turns[j];val weight=t.endMs-t.startMs;val v=vectors[j]
            if(!clean[j] || v==null || weight<=0) continue
            val sum=sums.getOrPut(t.speaker) { DoubleArray(v.size) }
            for(k in v.indices) sum[k]+=v[k]*weight
            durations[t.speaker]=(durations[t.speaker]?:0)+weight
        }
        val result=turns.toMutableList();var changed=0
        for(i in turns.indices) {
            val t=turns[i];val v=vectors[i]
            if(t.endMs-t.startMs>maxDurationMs || v==null || !clean[i]) continue
            val scores=mutableListOf<Pair<Double,String>>()
            for((label,total) in sums) {
                val vector=total.copyOf();var duration=durations[label]!!
                if(label==t.speaker) {
                    val weight=t.endMs-t.startMs;duration-=weight
                    for(k in vector.indices) vector[k]-=v[k]*weight
                }
                val norm=sqrt(vector.sumOf { it*it })
                if(duration>=minReferenceMs && norm>0) scores.add(v.indices.sumOf { v[it]*vector[it]/norm } to label)
            }
            scores.sortWith(Comparator { a,b -> val x=b.first.compareTo(a.first);if(x!=0) x else scalarCompare(a.second,b.second) })
            if(scores.size>=2) {
                val best=scores[0];val second=scores[1]
                if(best.second!=t.speaker && best.first>=minSimilarity && best.first-second.first>=minMargin) {
                    result[i]=t.copy(speaker=best.second);changed++
                }
            }
        }
        return RefinementResult(result,changed)
    }

    fun jsonQuote(text: String): String = buildString {
        append('"')
        for(c in text) when(c) {
            '"'->append("\\\"");'\\'->append("\\\\");'\b'->append("\\b");'\u000C'->append("\\f")
            '\n'->append("\\n");'\r'->append("\\r");'\t'->append("\\t")
            else->if(c.code<32) append("\\u"+c.code.toString(16).padStart(4,'0')) else append(c)
        }
        append('"')
    }
}
