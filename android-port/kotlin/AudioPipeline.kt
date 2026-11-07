package org.dictophone.port

import java.io.Closeable
import kotlin.math.max
import kotlin.math.min

/** Implement with local JNI/SDK calls. All timestamps returned by ASR are WINDOW-relative ms. */
interface OfflineAsr : Closeable { fun transcribe(pcm16kMono: FloatArray, language: String): List<Word> }
interface OfflineDiarizer : Closeable { fun diarize(pcm16kMono: FloatArray): List<Turn> }
interface OfflineEmbeddings : Closeable { fun extract(pcm16kMono: FloatArray): DoubleArray? }
interface OfflineTextModel : Closeable { fun punctuate(text: String): String }

data class PipelineSettings(
    val speechGuided: Boolean=true,
    val refineShortTurns: Boolean=true,
    val padMs: Long=500,
    val mergeGapMs: Long=1500,
    val pauseMs: Long=700,
    val maxUtteranceMs: Long=15000,
    val shortTurnMaxMs: Long=3000,
    val shortTurnReferenceMs: Long=3500,
    val shortTurnMinSimilarity: Double=0.6,
    val shortTurnMinMargin: Double=0.15
)

data class PipelineResult(
    val transcript: List<Utterance>,
    val rawTranscript: List<Utterance>,
    val durationMs: Long,
    val warnings: List<String>,
    val acceptedLlmTexts: Int,
    val refinedTurns: Int
) {
    fun json()=AudioPipelineLogic.toJson(transcript,durationMs)
}

/** Model factories are invoked sequentially, so native models need not coexist in RAM.
 * MP3 decoding, SAF export, progress UI, threading and actual native bindings belong to the app.
 */
class AudioPipeline(
    private val asrFactory: () -> OfflineAsr,
    private val diarizerFactory: () -> OfflineDiarizer,
    private val textFactory: () -> OfflineTextModel,
    private val embeddingsFactory: (() -> OfflineEmbeddings)?=null,
    private val settings: PipelineSettings=PipelineSettings()
) {
    fun runDecoded(pcm16kMono: FloatArray, language: String="ru"): PipelineResult {
        require(pcm16kMono.isNotEmpty() && pcm16kMono.all { it.isFinite() })
        require(language in setOf("ru","en"))
        val durationMs=(pcm16kMono.size.toLong()*1000+15999)/16000
        val warnings=mutableListOf<String>()
        var turns=try { diarizerFactory().use { it.diarize(pcm16kMono) }.also {
            AudioPipelineLogic.canonicalTurns(it,durationMs)
        } } catch(e:Exception) { warnings.add("diarization_failed:unknown_speakers");emptyList() }
        var changed=0
        if(settings.refineShortTurns && embeddingsFactory!=null) {
            try {
                val vectors=embeddingsFactory.invoke().use { extractor -> turns.map { t ->
                    extractor.extract(pcm16kMono.copyOfRange((t.startMs*16).toInt(),min(pcm16kMono.size.toLong(),t.endMs*16).toInt()))
                } }
                val refined=AudioPipelineLogic.refineLabels(turns,vectors,settings.shortTurnMaxMs,
                    settings.shortTurnReferenceMs,settings.shortTurnMinSimilarity,settings.shortTurnMinMargin)
                turns=refined.turns;changed=refined.changes
            } catch(e:Exception) { warnings.add("short_turn_refinement_failed:original_diarization_preserved") }
        }
        val windows=if(settings.speechGuided) AudioPipelineLogic.asrWindows(turns,durationMs,settings.padMs,settings.mergeGapMs)
                    else listOf(AsrWindow(0,durationMs))
        val words=asrFactory().use { asr -> windows.flatMap { window ->
            val segment=pcm16kMono.copyOfRange((window.startMs*16).toInt(),min(pcm16kMono.size.toLong(),window.endMs*16).toInt())
            asr.transcribe(segment,language).map { w ->
                require(w.endMs>=w.startMs)
                val a=min(window.endMs,max(window.startMs,window.startMs+w.startMs))
                val b=min(window.endMs,max(a,window.startMs+w.endMs))
                w.copy(startMs=a,endMs=b)
            }
        } }
        if(words.isNotEmpty() && turns.isEmpty() && warnings.none { it.startsWith("diarization_failed") })
            warnings.add("diarization_empty:unknown_speakers")
        val raw=AudioPipelineLogic.align(AudioPipelineLogic.addUnintelligible(words,turns),turns,durationMs,
                                          settings.pauseMs,settings.maxUtteranceMs)
        val (rows,accepted)=postprocess(raw,warnings)
        return PipelineResult(rows,raw,durationMs,warnings.toList(),accepted,changed)
    }

    /** Re-run punctuation without decoding, ASR or diarization. Keep raw data in the app session. */
    fun retryText(previous: PipelineResult): PipelineResult {
        val warnings=mutableListOf<String>()
        val (rows,accepted)=postprocess(previous.rawTranscript,warnings)
        return previous.copy(transcript=rows,warnings=warnings,acceptedLlmTexts=accepted)
    }

    private fun postprocess(raw: List<Utterance>, warnings: MutableList<String>): Pair<List<Utterance>,Int> {
        if(raw.isEmpty()) return emptyList<Utterance>() to 0
        var accepted=0
        return try {
            val rows=textFactory().use { model -> raw.mapIndexed { i,r ->
                if(r.text=="[Неразборчиво]") r else try {
                    val text=AudioPipelineLogic.guardedText(r.text,model.punctuate(r.text))
                    accepted++;r.copy(text=text)
                } catch(e:Exception) { warnings.add("llm_rejected_or_failed:utterance_${i+1}");r }
            } }
            rows to accepted
        } catch(e:Exception) { warnings.add("llm_unavailable:original_asr_preserved");raw to 0 }
    }
}
