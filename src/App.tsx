/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileText,
  Languages,
  CheckCircle2,
  AlertCircle,
  Download,
  Loader2,
  Sparkles,
  RefreshCw,
  Clock,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { SubtitleViewer, PreviewItem } from './components/SubtitleViewer';
import { SAMPLE_JAPANESE_ASS, SAMPLE_ENGLISH_SRT } from './data/sampleSubtitles';

export default function App() {
  const [file, setFile] = useState<{ name: string; size: number; content: string } | null>(null);
  const [sourceLanguage, setSourceLanguage] = useState<'auto' | 'ja' | 'en'>('auto');
  const [modelChoice, setModelChoice] = useState<'gemini-4-hydrogen' | 'gemini-3.8-flash' | 'gemini-3.1-flash-lite'>('gemini-4-hydrogen');
  const [isDragging, setIsDragging] = useState(false);

  // Translation State
  const [status, setStatus] = useState<'idle' | 'translating' | 'completed' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [progressPercent, setProgressPercent] = useState(0);
  const [currentBatchInfo, setCurrentBatchInfo] = useState<{ current: number; total: number } | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');

  // Results
  const [translatedAss, setTranslatedAss] = useState<string>('');
  const [downloadFileName, setDownloadFileName] = useState<string>('');
  const [previewItems, setPreviewItems] = useState<PreviewItem[]>([]);
  const [stats, setStats] = useState<{ totalEvents: number; translatableCount: number } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Manage elapsed timer
  useEffect(() => {
    if (status === 'translating') {
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => {
        setElapsedSeconds((sec) => sec + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [status]);

  const handleFileLoad = (name: string, content: string, size: number) => {
    // Validate extension
    const lowerName = name.toLowerCase();
    if (!lowerName.endsWith('.ass') && !lowerName.endsWith('.srt')) {
      setErrorMessage('Unsupported file format. Please upload an ASS (.ass) or SRT (.srt) subtitle file.');
      return;
    }

    if (!content.trim()) {
      setErrorMessage('The uploaded subtitle file is empty.');
      return;
    }

    setErrorMessage('');
    setFile({ name, size, content });
    setStatus('idle');
    setTranslatedAss('');
    setPreviewItems([]);
    setStats(null);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      handleFileLoad(selected.name, content, selected.size);
    };
    reader.readAsText(selected);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (!droppedFile) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      handleFileLoad(droppedFile.name, content, droppedFile.size);
    };
    reader.readAsText(droppedFile);
  };

  const loadSample = (type: 'ass' | 'srt') => {
    if (type === 'ass') {
      const content = SAMPLE_JAPANESE_ASS;
      handleFileLoad('evangelion_ep01_sample.ass', content, new Blob([content]).size);
      setSourceLanguage('ja');
    } else {
      const content = SAMPLE_ENGLISH_SRT;
      handleFileLoad('anime_adventure_ep01.srt', content, new Blob([content]).size);
      setSourceLanguage('en');
    }
  };

  const startTranslation = async () => {
    if (!file) return;

    setStatus('translating');
    setStatusMessage('Reading and preparing subtitle dialogue events...');
    setProgressPercent(5);
    setPreviewItems([]);
    setErrorMessage('');

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('/api/translate/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fileContent: file.content,
          fileName: file.name,
          sourceLanguage,
          modelChoice,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported on this browser response');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let didComplete = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\r\n/g, '\n');
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data:')) {
            const jsonStr = trimmed.slice(5).trim();
            let eventData: Record<string, any> | null = null;
            try {
              eventData = JSON.parse(jsonStr);
            } catch (parseErr) {
              console.error('Error parsing SSE JSON:', parseErr);
              continue;
            }

            if (!eventData) continue;

            if (eventData.type === 'init') {
              setStats({
                totalEvents: eventData.totalEvents,
                translatableCount: eventData.translatableCount,
              });
              setStatusMessage(`Initialized: ${eventData.translatableCount} dialogue lines identified.`);
            } else if (eventData.type === 'status') {
              setStatusMessage(eventData.statusMessage);
              if (eventData.batchIndex && eventData.totalBatches) {
                setCurrentBatchInfo({ current: eventData.batchIndex, total: eventData.totalBatches });
              }
              if (typeof eventData.percent === 'number') {
                setProgressPercent((prev) => Math.max(prev, eventData!.percent));
              }
            } else if (eventData.type === 'batch_complete') {
              if (eventData.preview && Array.isArray(eventData.preview)) {
                setPreviewItems((prev) => {
                  const combined = [...prev, ...eventData!.preview];
                  combined.sort((a, b) => a.index - b.index);
                  return combined;
                });
              }
              if (eventData.completedBatches && eventData.totalBatches) {
                setCurrentBatchInfo({ current: eventData.completedBatches, total: eventData.totalBatches });
              }
              if (typeof eventData.percent === 'number') {
                setProgressPercent((prev) => Math.max(prev, eventData!.percent));
              }
              setStatusMessage(
                `Completed ${eventData.processedCount}/${eventData.totalCount} lines (${eventData.completedBatches}/${eventData.totalBatches} batches).`
              );
            } else if (eventData.type === 'complete') {
              didComplete = true;
              if (Array.isArray(eventData.finalPreview)) {
                setPreviewItems(eventData.finalPreview);
              }
              setTranslatedAss(eventData.assContent);
              setDownloadFileName(eventData.outputFileName);
              setProgressPercent(100);
              setStatus('completed');
              setStatusMessage('Translation and validation completed successfully!');
            } else if (eventData.type === 'error') {
              throw new Error(eventData.message || 'Translation failed on server');
            }
          }
        }
      }

      if (!didComplete) {
        throw new Error('Translation stream closed unexpectedly before completion. Please try again.');
      }
    } catch (err: unknown) {
      if ((err as Error)?.name === 'AbortError') {
        setStatus('idle');
        setStatusMessage('Translation cancelled.');
      } else {
        setStatus('error');
        setErrorMessage(err instanceof Error ? err.message : 'Unknown translation failure.');
      }
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setStatus('idle');
      setStatusMessage('Translation stopped.');
    }
  };

  const handleDownload = () => {
    if (!translatedAss || !downloadFileName) return;
    const blob = new Blob([translatedAss], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = downloadFileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-600 selection:text-white">
      {/* Top Navigation */}
      <header className="border-b border-slate-800/80 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 min-h-16 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 shrink-0 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="font-bold text-base sm:text-lg text-white tracking-tight">Anime Subtitle Translator</h1>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                  ASS & SRT
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Preserves anime styles, override tags, and line breaks into Burmese (မြန်မာဘာသာ)
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
              <Zap className="w-3.5 h-3.5 text-indigo-400" />
              Gemini 4 Hydrogen • XHigh Thinking ⚡
            </span>
            <span className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <ShieldCheck className="w-3.5 h-3.5" />
              Tag & Timing Safe
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Upload Zone & Config Grid */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* File Upload Box (Spans 2 columns) */}
          <div className="lg:col-span-2 space-y-4">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 group ${
                isDragging
                  ? 'border-indigo-400 bg-indigo-950/20 shadow-lg shadow-indigo-500/10'
                  : file
                  ? 'border-slate-700 bg-slate-900/60 hover:border-slate-600'
                  : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/70'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".ass,.srt"
                onChange={handleFileInputChange}
                className="hidden"
              />

              <div className="w-14 h-14 rounded-2xl bg-slate-800 group-hover:bg-slate-700 flex items-center justify-center mb-4 transition-colors">
                <UploadCloud className="w-7 h-7 text-indigo-400 group-hover:text-indigo-300" />
              </div>

              {file ? (
                <div className="space-y-1">
                  <div className="flex items-center justify-center gap-2">
                    <FileText className="w-4 h-4 text-indigo-400" />
                    <span className="font-semibold text-white text-sm sm:text-base">{file.name}</span>
                    <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      {file.name.toLowerCase().endsWith('.srt') ? 'SRT' : 'ASS'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">
                    File size: <span className="font-mono text-slate-300">{formatFileSize(file.size)}</span> • Click or drag to replace
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <p className="text-sm sm:text-base font-semibold text-slate-200">
                    Drop your anime subtitle file here, or{' '}
                    <span className="text-indigo-400 underline decoration-indigo-400/40 underline-offset-4">
                      browse
                    </span>
                  </p>
                  <p className="text-xs text-slate-400 max-w-sm">
                    Supports <strong className="text-slate-300">.ass</strong> (Advanced SubStation Alpha) and{' '}
                    <strong className="text-slate-300">.srt</strong>. No video file required.
                  </p>
                </div>
              )}
            </div>

            {/* Quick Sample Selector */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 bg-slate-900/40 border border-slate-800/80 rounded-xl text-xs">
              <span className="text-slate-400 flex items-center gap-1.5 font-medium">
                <Zap className="w-4 h-4 text-amber-400" />
                Quick test with anime samples:
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => loadSample('ass')}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition border border-slate-700"
                >
                  Load Japanese ASS (Neon Genesis)
                </button>
                <button
                  type="button"
                  onClick={() => loadSample('srt')}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition border border-slate-700"
                >
                  Load English SRT (Adventure)
                </button>
              </div>
            </div>
          </div>

          {/* Translation Settings & Action (1 Column) */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between space-y-6">
            <div className="space-y-5">
              <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Languages className="w-4 h-4 text-indigo-400" />
                Translation Settings
              </h2>

              {/* AI Engine Selector */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label htmlFor="ai-model-select" className="block text-xs font-medium text-slate-300">
                    AI Translation Engine
                  </label>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    Google AI Studio
                  </span>
                </div>
                <select
                  id="ai-model-select"
                  aria-label="AI Translation Engine"
                  value={modelChoice}
                  onChange={(e) => setModelChoice(e.target.value as 'gemini-4-hydrogen' | 'gemini-3.8-flash' | 'gemini-3.1-flash-lite')}
                  disabled={status === 'translating'}
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium"
                >
                  <option value="gemini-4-hydrogen">Gemini 4 Hydrogen (XHigh Thinking)</option>
                  <option value="gemini-3.8-flash">Gemini 3.8 Flash (Balanced • Low)</option>
                  <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash-Lite (Ultra Fast)</option>
                </select>
                <p className="text-[11px] text-slate-400 px-1">
                  {modelChoice === 'gemini-4-hydrogen'
                    ? '🧠 Powered by Gemini 4 Hydrogen with XHigh Thinking for deep anime context accuracy.'
                    : modelChoice === 'gemini-3.8-flash'
                    ? '⚡ Powered by Gemini 3.8 Flash with 3x parallel acceleration and auto-fallback.'
                    : '⚡ Ultra-fast lightweight model for instant translation.'}
                </p>
              </div>

              {/* Source Language */}
              <div className="space-y-1.5">
                <label htmlFor="source-language-select" className="block text-xs font-medium text-slate-300">
                  Source Language
                </label>
                <select
                  id="source-language-select"
                  aria-label="Source Language"
                  value={sourceLanguage}
                  onChange={(e) => setSourceLanguage(e.target.value as 'auto' | 'ja' | 'en')}
                  disabled={status === 'translating'}
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium"
                >
                  <option value="auto">Auto detect</option>
                  <option value="ja">Japanese (日本語)</option>
                  <option value="en">English</option>
                </select>
              </div>

              {/* Target Language (Fixed to Burmese) */}
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-slate-300">Target Language</label>
                <div className="p-3 bg-slate-800/80 border border-slate-700/80 rounded-xl flex items-center justify-between">
                  <div>
                    <div className="text-xs font-semibold text-white">Burmese / Myanmar</div>
                    <div className="text-[11px] text-emerald-400 font-sans mt-0.5">မြန်မာဘာသာ (Anime Subtitle Format)</div>
                  </div>
                  <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/40">
                    Fixed
                  </span>
                </div>
              </div>

              {/* Strict Subtitle Rules summary */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 text-[11px] text-slate-400 space-y-1.5">
                <div className="text-slate-200 font-semibold flex items-center justify-between">
                  <span>Guaranteed Subtitle Integrity:</span>
                  <span className="text-[10px] font-mono text-emerald-400 font-normal">Target CPS &lt; 10</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-300">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span><strong>CPS &lt; 10 Limit:</strong> Spoken lines compressed for fast reading</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-300">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span><strong>Signs & Titles:</strong> Exempt from character shortening</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-300">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span><strong>Nearby Duplicates:</strong> Matched text kept identical (CPS exempt)</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span>Preserves {'{\\an8}'}, {'{\\pos}'}, colors, &amp; literal \N breaks</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span>Zero timestamp modification (Timecode Safe)</span>
                </div>
              </div>
            </div>

            {/* Translate Button */}
            <div>
              {status === 'translating' ? (
                <button
                  type="button"
                  onClick={handleCancel}
                  className="w-full py-3 px-4 rounded-xl font-semibold text-xs text-rose-300 bg-rose-950/60 hover:bg-rose-900/60 border border-rose-800/50 transition flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-4 h-4 animate-spin text-rose-400" />
                  Stop / Cancel Translation
                </button>
              ) : (
                <button
                  type="button"
                  onClick={startTranslation}
                  disabled={!file}
                  className={`w-full py-3 px-4 rounded-xl font-semibold text-sm transition flex items-center justify-center gap-2 shadow-lg ${
                    file
                      ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white shadow-indigo-600/25 active:scale-[0.99]'
                      : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                  }`}
                >
                  <Sparkles className="w-4 h-4" />
                  Translate to Burmese ASS
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Error Alert */}
        {errorMessage && (
          <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-xs flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold text-rose-300">Translation Notice</div>
              <div>{errorMessage}</div>
            </div>
          </div>
        )}

        {/* Translation Progress Indicator */}
        {status === 'translating' && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
                <div>
                  <h3 className="text-sm font-semibold text-white">Translating Subtitles</h3>
                  <p className="text-xs text-slate-400">{statusMessage}</p>
                </div>
              </div>

              <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
                {currentBatchInfo && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
                    <Layers className="w-3.5 h-3.5 text-indigo-400" />
                    Batch {currentBatchInfo.current} / {currentBatchInfo.total}
                  </span>
                )}
                <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
                  <Clock className="w-3.5 h-3.5 text-cyan-400" />
                  {elapsedSeconds}s elapsed
                </span>
                <span className="font-semibold text-indigo-400 text-sm">{progressPercent}%</span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden p-0.5">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 via-indigo-400 to-cyan-400 rounded-full transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            {/* Live translation streaming preview if lines arrived */}
            {previewItems.length > 0 && (
              <div className="pt-2">
                <div className="text-[11px] uppercase tracking-wider font-semibold text-slate-400 mb-2">
                  Live Translated Lines ({previewItems.length}{stats ? ` / ${stats.translatableCount}` : ''}):
                </div>
                <div className="max-h-48 overflow-y-auto space-y-2 pr-1 divide-y divide-slate-800/60">
                  {previewItems.slice(-5).map((item) => (
                    <div key={item.index} className="pt-2 text-xs flex flex-col sm:flex-row gap-2 justify-between">
                      <div className="text-slate-400 text-[11px] font-mono shrink-0">
                        [{item.time}] {item.style}:
                      </div>
                      <div className="flex-1 text-slate-300">{item.original}</div>
                      <div className="flex-1 text-emerald-400 font-medium">{item.translated}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Translation Completed Banner & Downloader */}
        {status === 'completed' && (
          <div className="bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-500/30 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-start gap-3.5">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 border border-emerald-500/30">
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    Translation Complete & Verified
                    <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      100% Timing Preserved
                    </span>
                  </h3>
                  <p className="text-xs text-slate-300 mt-1">
                    All original ASS styles, positioning tags, colors, and line breaks are intact. Output verified as a valid ASS subtitle file.
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={handleDownload}
                className="px-6 py-3 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 transition shadow-lg shadow-emerald-600/25 flex items-center justify-center gap-2 shrink-0 active:scale-[0.99]"
              >
                <Download className="w-4 h-4" />
                Download Translated ASS
              </button>
            </div>

            {/* Subtitle Viewer with side-by-side table & raw code preview */}
            <SubtitleViewer
              items={previewItems}
              rawAssContent={translatedAss}
              downloadFileName={downloadFileName}
              onDownload={handleDownload}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500">
        <p>Anime Subtitle Translator • Advanced ASS & SRT Subtitle Localization into Burmese</p>
      </footer>
    </div>
  );
}
