import { BookOpen, ExternalLink, X } from "lucide-react";
import type { FinalResponse } from "../../lib/types";

type EvidencePanelProps = {
  isOpen: boolean;
  onClose: () => void;
  response: FinalResponse | null;
};

export function EvidencePanel({ isOpen, onClose, response }: EvidencePanelProps) {
  const citations = response?.citations ?? [];
  const isPrediction = response?.evidence_kind === "model_prediction";
  return (
    <aside className={`evidence-panel ${isOpen ? "is-open" : ""}`} aria-label="证据来源">
      <div className="evidence-header">
        <div><BookOpen size={18} aria-hidden="true" /><span>证据来源</span></div>
        <button className="icon-control" onClick={onClose} title="关闭证据面板" type="button"><X size={18} /></button>
      </div>
      {isPrediction ? (
        <div className="evidence-empty">
          <strong>模型候选不包含文献引用</strong>
          <p>本轮内容来自本地路线预测模型，需在实验前独立核查可行性和安全性。</p>
        </div>
      ) : citations.length === 0 ? (
        <div className="evidence-empty">当前回答没有可展示的文献记录。</div>
      ) : (
        <div className="citation-list">
          {citations.map((citation) => (
            <article className="citation-entry" key={citation.record_id}>
              <div className="citation-record">{citation.record_id}</div>
              {citation.doi ? (
                <a className="doi-link" href={`https://doi.org/${citation.doi}`} rel="noreferrer" target="_blank">
                  <span>{citation.doi}</span><ExternalLink size={14} aria-hidden="true" />
                </a>
              ) : null}
              <p>{citation.source_text}</p>
            </article>
          ))}
        </div>
      )}
    </aside>
  );
}
