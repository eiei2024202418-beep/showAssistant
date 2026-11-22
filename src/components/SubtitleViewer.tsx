import React, { useState } from 'react';
import { Copy, Check, Search, Download, FileCode, List } from 'lucide-react';
import {
  type DialogueCpsMode,
  getCpsModeConfig,
  isCpsCompliantForMode,
} from '../utils/cpsMode';

export interface PreviewItem {
  index: number;
  time: string;
  start?: string;
  end?: string;
  durationSec?: number;
  cps?: number;
  isSignOrTitle?: boolean;
  actor?: string;
  style: string;
  original: string;
  translated: string;
}

interface SubtitleViewerProps {
  items: PreviewItem[];
  rawAssContent: string;
  downloadFileName: string;
  onDownload: () => void;
  cpsMode?: DialogueCpsMode;
}

export const SubtitleViewer: React.FC<SubtitleViewerProps> = ({
  items,
  rawAssContent,
  downloadFileName,
  onDownload,
  cpsMode = 'balanced_natural',
}) => {
  const [activeTab, setActiveTab] = useState<'table' | 'raw'>('table');
  const [filterMode, setFilterMode] = useState<'all' | 'dialogue' | 'signs'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(rawAssContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  // Stats calculation matching selected Dialogue CPS Mode
  const cpsConfig = getCpsModeConfig(cpsMode);
  const dialogueItems = items.filter((item) => !item.isSignOrTitle);
  const signItems = items.filter((item) => item.isSignOrTitle);
  const compliantCpsCount = dialogueItems.filter((item) => isCpsCompliantForMode(item.cps ?? 0, cpsMode)).length;
  const compliantPercent = dialogueItems.length > 0 ? Math.round((compliantCpsCount / dialogueItems.length) * 100) : 100;

  const filteredItems = items.filter((item) => {
    if (filterMode === 'dialogue' && item.isSignOrTitle) return false;
    if (filterMode === 'signs' && !item.isSignOrTitle) return false;

    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.original.toLowerCase().includes(q) ||
      item.translated.toLowerCase().includes(q) ||
      item.style.toLowerCase().includes(q) ||
      (item.actor && item.actor.toLowerCase().includes(q))
    );
  });

  return (
    <div className="bg-slate-900/95 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-slate-800 bg-slate-900/80">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('table')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === 'table'
                ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <List className="w-3.5 h-3.5" />
            Dialogue Table ({items.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('raw')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === 'raw'
                ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            Raw ASS Output
          </button>
        </div>

        {/* CPS Compliance Indicator */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/40 text-[11px] text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          {cpsConfig.viewerComplianceLimit === null ? (
            <span>
              Dialogue CPS: <strong>Unlimited</strong> ({dialogueItems.length} dialogue lines)
            </span>
          ) : (
            <span>
              Dialogue CPS &le; {cpsConfig.viewerComplianceLimit}: <strong>{compliantPercent}%</strong> ({compliantCpsCount}/{dialogueItems.length} lines)
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {activeTab === 'table' && (
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                placeholder="Search actor, style, dialogue..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-xs bg-slate-800/90 border border-slate-700/80 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 w-52"
              />
            </div>
          )}

          {activeTab === 'raw' && (
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 transition border border-slate-700"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied ASS' : 'Copy ASS'}
            </button>
          )}

          <button
            type="button"
            onClick={onDownload}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition shadow-sm"
          >
            <Download className="w-3.5 h-3.5" />
            Download {downloadFileName}
          </button>
        </div>
      </div>

      {/* Filter Sub-bar */}
      {activeTab === 'table' && (
        <div className="flex items-center justify-between px-5 py-2 bg-slate-950/40 border-b border-slate-800/60 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 text-[11px] mr-1">Filter:</span>
            <button
              type="button"
              onClick={() => setFilterMode('all')}
              className={`px-2.5 py-0.5 rounded text-[11px] font-medium transition ${
                filterMode === 'all'
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              All ({items.length})
            </button>
            <button
              type="button"
              onClick={() => setFilterMode('dialogue')}
              className={`px-2.5 py-0.5 rounded text-[11px] font-medium transition ${
                filterMode === 'dialogue'
                  ? 'bg-emerald-800/60 text-emerald-200 border border-emerald-600/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              Spoken Dialogue ({dialogueItems.length})
            </button>
            <button
              type="button"
              onClick={() => setFilterMode('signs')}
              className={`px-2.5 py-0.5 rounded text-[11px] font-medium transition ${
                filterMode === 'signs'
                  ? 'bg-indigo-800/60 text-indigo-200 border border-indigo-600/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              Signs & Titles ({signItems.length})
            </button>
          </div>

          <div className="text-[11px] text-slate-400">
            {cpsConfig.viewerComplianceLimit === null
              ? 'Aegisub Layout • Pacing Mode: Natural Unlimited'
              : `Aegisub Layout • Standard CPS Target: <= ${cpsConfig.viewerComplianceLimit}`}
          </div>
        </div>
      )}

      {/* Tab 1: Dialogue Comparison Table matching Aegisub format */}
      {activeTab === 'table' && (
        <div className="overflow-x-auto max-h-[520px] overflow-y-auto divide-y divide-slate-800/60">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 sticky top-0 text-slate-400 uppercase tracking-wider text-[11px] backdrop-blur-md z-10 border-b border-slate-800">
              <tr>
                <th className="py-2 px-3 w-10 text-center font-mono">#</th>
                <th className="py-2 px-3 w-24 font-mono">Start</th>
                <th className="py-2 px-3 w-24 font-mono">End</th>
                <th className="py-2 px-3 w-16 text-center font-mono">CPS</th>
                <th className="py-2 px-3 w-28">Style</th>
                <th className="py-2 px-3 w-24">Actor</th>
                <th className="py-2 px-4 w-[38%]">Original Text (Source)</th>
                <th className="py-2 px-4 w-[38%]">Burmese Translation (မြန်မာ)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/40 text-slate-300 font-sans">
              {filteredItems.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-10 text-center text-slate-500">
                    No matching subtitle lines found.
                  </td>
                </tr>
              ) : (
                filteredItems.map((item, idx) => {
                  const startTime = item.start || (item.time ? item.time.split('-')[0]?.trim() : '');
                  const endTime = item.end || (item.time ? item.time.split('-')[1]?.trim() : '');
                  const cpsVal = item.cps ?? 0;
                  const isSign = item.isSignOrTitle ?? false;

                  return (
                    <tr key={item.index} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2.5 px-3 text-center font-mono text-slate-500 text-[11px]">
                        {idx + 1}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-400 text-[11px] whitespace-nowrap">
                        {startTime}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-400 text-[11px] whitespace-nowrap">
                        {endTime}
                      </td>
                      <td className="py-2.5 px-3 text-center whitespace-nowrap">
                        {isSign ? (
                          <span
                            className="inline-block px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-400 border border-slate-700/60"
                            title="Sign / Text / Title (Exempt from CPS limit)"
                          >
                            SIGN
                          </span>
                        ) : cpsConfig.viewerBadgeGreenLimit === null ? (
                          <span
                            className="inline-block px-1.5 py-0.5 rounded text-[11px] font-mono font-medium bg-slate-800/80 text-indigo-300 border border-slate-700/80"
                            title={`CPS: ${cpsVal} (Natural Unlimited — No CPS Limit)`}
                          >
                            {Math.round(cpsVal)}
                          </span>
                        ) : cpsVal <= cpsConfig.viewerBadgeGreenLimit ? (
                          <span
                            className="inline-block px-1.5 py-0.5 rounded text-[11px] font-mono font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/50"
                            title={`CPS: ${cpsVal} (Good - Within ${cpsConfig.viewerComplianceLimit} CPS)`}
                          >
                            {Math.round(cpsVal)}
                          </span>
                        ) : (
                          <span
                            className="inline-block px-1.5 py-0.5 rounded text-[11px] font-mono font-medium bg-amber-950/60 text-amber-300 border border-amber-800/50"
                            title={`CPS: ${cpsVal} (Exceeds ${cpsConfig.viewerComplianceLimit} CPS)`}
                          >
                            {Math.round(cpsVal)}
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span className="inline-block px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-indigo-300 font-medium">
                          {item.style || 'Default'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span className="text-[11px] font-semibold text-slate-300 font-mono">
                          {item.actor || '—'}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 leading-relaxed text-xs">
                        {renderTaggedSubtitle(item.original)}
                      </td>
                      <td className="py-2.5 px-4 leading-relaxed text-xs text-emerald-300 font-medium">
                        {renderTaggedSubtitle(item.translated)}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Tab 2: Raw ASS Text */}
      {activeTab === 'raw' && (
        <div className="p-4 bg-slate-950/90 max-h-[520px] overflow-y-auto">
          <pre className="font-mono text-xs text-slate-300 leading-relaxed whitespace-pre-wrap selection:bg-indigo-600/40">
            {rawAssContent}
          </pre>
        </div>
      )}
    </div>
  );
};

/**
 * Highlights ASS override tags like {\...} and \N with subtle styles
 */
function renderTaggedSubtitle(text: string): React.ReactNode {
  if (!text) return null;

  // Split by ASS tags {...} and \N tokens
  const parts = text.split(/(\{[^}]*\}|\\N|\\n)/g);

  return (
    <span>
      {parts.map((part, i) => {
        if (part.startsWith('{') && part.endsWith('}')) {
          return (
            <span
              key={i}
              className="inline-block px-1 py-0.5 mx-0.5 text-[10px] font-mono text-amber-300/80 bg-amber-950/40 rounded border border-amber-800/30"
              title="Preserved ASS tag"
            >
              {part}
            </span>
          );
        }
        if (part === '\\N' || part === '\\n') {
          return (
            <span
              key={i}
              className="inline-block px-1 py-0.5 mx-0.5 text-[10px] font-mono text-cyan-300/80 bg-cyan-950/40 rounded border border-cyan-800/30"
              title="Preserved line break"
            >
              \N
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}
